"""
Null-space constrained model merging.

This script mirrors the FREE-Merging entry point but injects AlphaEdit-style
null-space projection. It:
  1. Builds a preserve Jacobian basis from a prompt dataset.
  2. Computes per-layer singular vectors and applies null-space projection.
  3. Merges expert deltas while suppressing directions that would leak preserve
     knowledge.

The implementation targets autoregressive LLMs loaded via Hugging Face
transformers. It is intentionally modular so that SVD- or frequency-based
extensions can be composed downstream.
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


def load_prompts(path: Path, field: Optional[str], limit: Optional[int]) -> List[str]:
    """Load prompts from .txt or .jsonl file."""
    if not path.exists():
        raise FileNotFoundError(f"Preserve dataset not found: {path}")

    prompts: List[str] = []
    if path.suffix == ".txt":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append(line)
    elif path.suffix in {".jsonl", ".ndjson"}:
        if field is None:
            raise ValueError("--preserve_field is required for JSONL datasets.")
        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                record = json.loads(raw)
                if field not in record:
                    raise KeyError(f"Field '{field}' missing in record: {record.keys()}")
                value = str(record[field]).strip()
                if value:
                    prompts.append(value)
    else:
        raise ValueError(f"Unsupported dataset format: {path.suffix}")

    if limit is not None:
        prompts = prompts[:limit]
    if not prompts:
        raise ValueError("Loaded preserve prompts are empty.")
    return prompts


def compile_layer_filter(patterns: Iterable[str]) -> List[re.Pattern]:
    compiled: List[re.Pattern] = []
    for pattern in patterns:
        compiled.append(re.compile(pattern))
    return compiled


@torch.no_grad()
def extract_state_dict(model: PreTrainedModel) -> Dict[str, torch.Tensor]:
    state = {}
    for name, param in model.named_parameters():
        state[name] = param.detach().clone()
    return state


def filter_parameters(
    model: PreTrainedModel, filters: List[re.Pattern]
) -> Dict[str, torch.nn.Parameter]:
    if not filters:
        return dict(model.named_parameters())

    selected: Dict[str, torch.nn.Parameter] = {}
    for name, param in model.named_parameters():
        if any(p.search(name) for p in filters):
            selected[name] = param
    if not selected:
        raise ValueError("No parameters matched the provided --layer_regex patterns.")
    return selected


def batchify(iterable: List[str], batch_size: int) -> Iterable[List[str]]:
    for idx in range(0, len(iterable), batch_size):
        yield iterable[idx : idx + batch_size]


def gather_gradients(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: List[str],
    parameter_pool: Dict[str, torch.nn.Parameter],
    device: torch.device,
    batch_size: int,
) -> Dict[str, List[torch.Tensor]]:
    model.eval()
    gradients: Dict[str, List[torch.Tensor]] = {name: [] for name in parameter_pool}
    loss_count = 0

    for batch in tqdm(
        list(batchify(prompts, batch_size)),
        desc="Collecting preserve gradients",
        leave=False,
    ):
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        model.zero_grad(set_to_none=True)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="mean",
        )
        loss.backward()
        loss_count += 1

        for name, param in parameter_pool.items():
            grad = param.grad
            if grad is None:
                continue
            gradients[name].append(grad.detach().flatten().cpu())

    if not loss_count:
        raise RuntimeError("Failed to compute gradients for preserve prompts.")

    return gradients


def compute_nullspace_bases(
    gradients: Dict[str, List[torch.Tensor]],
    max_rank: Optional[int],
    svd_threshold: float,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Return per-parameter (basis, singular_values)."""
    bases: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for name, grads in gradients.items():
        if not grads:
            continue
        G = torch.stack(grads, dim=0).float()
        try:
            U, S, Vh = torch.linalg.svd(G, full_matrices=False)
        except RuntimeError as exc:  # pragma: no cover - rare SVD failure
            raise RuntimeError(f"SVD failed for layer {name}") from exc

        if svd_threshold > 0:
            keep = (S / S.max()) > svd_threshold
        else:
            keep = torch.ones_like(S, dtype=torch.bool)

        if max_rank is not None:
            keep = keep & (torch.arange(len(S)) < max_rank)

        rank = int(keep.sum().item())
        if rank == 0:
            continue

        basis = Vh[:rank, :].contiguous()
        bases[name] = (basis, S[:rank])
    return bases


def project_onto_nullspace(
    delta: torch.Tensor,
    basis: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    if basis is None or basis.numel() == 0:
        return delta

    device = delta.device
    flat = delta.view(-1)
    B = basis.to(device)
    coeffs = torch.matmul(B, flat)
    correction = torch.matmul(coeffs, B)
    flat_proj = flat - strength * correction
    return flat_proj.view_as(delta)


def merge_with_nullspace(
    base_state: Dict[str, torch.Tensor],
    deltas: List[Tuple[Dict[str, torch.Tensor], float]],
    bases: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    strength: float,
) -> Dict[str, torch.Tensor]:
    merged: Dict[str, torch.Tensor] = {}
    for name, base_tensor in base_state.items():
        base = base_tensor.to(torch.float32)
        update = torch.zeros_like(base)
        basis = bases.get(name, (None, None))[0] if name in bases else None

        for delta_state, weight in deltas:
            delta = delta_state[name].to(torch.float32) - base
            delta = project_onto_nullspace(delta, basis, strength)
            update = update + weight * delta

        merged[name] = (base + update).to(base_tensor.dtype)
    return merged


def load_model(
    model_name_or_path: str,
    device: torch.device,
    dtype: torch.dtype,
) -> PreTrainedModel:
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map=None,
        low_cpu_mem_usage=False,
    )
    return model.to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Null-space constrained model merging.")
    parser.add_argument("--base_model", required=True, help="Base HF path or checkpoint.")
    parser.add_argument(
        "--experts",
        nargs="+",
        required=True,
        help="List of expert model checkpoints to merge.",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        required=True,
        help="Merge weights aligned with --experts.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save the merged model.",
    )
    parser.add_argument(
        "--preserve_dataset",
        type=Path,
        required=True,
        help="Path to JSONL or TXT prompts describing preserved knowledge.",
    )
    parser.add_argument(
        "--preserve_field",
        type=str,
        default=None,
        help="Field name containing prompts when dataset is JSONL.",
    )
    parser.add_argument(
        "--max_prompts",
        type=int,
        default=64,
        help="Maximum number of preserve prompts to use.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for preserve gradient collection.",
    )
    parser.add_argument(
        "--nullspace_strength",
        type=float,
        default=1.0,
        help="Scaling applied to null-space projection cleanup.",
    )
    parser.add_argument(
        "--svd_threshold",
        type=float,
        default=1e-3,
        help="Relative singular value threshold for keeping directions.",
    )
    parser.add_argument(
        "--max_rank",
        type=int,
        default=None,
        help="Maximum rank kept for null-space basis per parameter.",
    )
    parser.add_argument(
        "--layer_regex",
        nargs="*",
        default=[
            r"transformer\.h\.\d+\.attn\.(c_attn|c_proj)\.weight",
            r"transformer\.h\.\d+\.mlp\.(c_fc|c_proj)\.weight",
        ],
        help="Regex patterns selecting parameters for null-space safeguards.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float16",
        help="Model load dtype.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for computations.",
    )
    args = parser.parse_args()

    if len(args.experts) != len(args.weights):
        raise ValueError("--experts and --weights must have identical lengths.")

    weight_sum = sum(args.weights)
    if math.isclose(weight_sum, 0.0):
        raise ValueError("Sum of merge weights must be non-zero.")
    weights = [w / weight_sum for w in args.weights]

    device = torch.device(args.device)
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    prompts = load_prompts(
        path=args.preserve_dataset,
        field=args.preserve_field,
        limit=args.max_prompts,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = load_model(args.base_model, device=device, dtype=torch.float32)
    parameter_filters = compile_layer_filter(args.layer_regex)
    parameter_pool = filter_parameters(base_model, parameter_filters)

    gradients = gather_gradients(
        model=base_model,
        tokenizer=tokenizer,
        prompts=prompts,
        parameter_pool=parameter_pool,
        device=device,
        batch_size=args.batch_size,
    )
    bases = compute_nullspace_bases(
        gradients=gradients,
        max_rank=args.max_rank,
        svd_threshold=args.svd_threshold,
    )

    base_state = extract_state_dict(base_model)
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    expert_states: List[Tuple[Dict[str, torch.Tensor], float]] = []
    for expert_path, weight in zip(args.experts, weights):
        expert_model = load_model(expert_path, device=device, dtype=dtype)
        expert_states.append((extract_state_dict(expert_model), weight))
        del expert_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    merged_state = merge_with_nullspace(
        base_state=base_state,
        deltas=expert_states,
        bases=bases,
        strength=args.nullspace_strength,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map=None,
        low_cpu_mem_usage=False,
    )
    for name, param in merged_model.named_parameters():
        if name in merged_state:
            param.data = merged_state[name].to(param.device, dtype=param.dtype)
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    meta_path = output_dir / "nullspace_meta.json"
    report = {
        "base_model": args.base_model,
        "experts": args.experts,
        "weights": weights,
        "nullspace_layers": list(bases.keys()),
        "rank_per_layer": {
            layer: int(basis.shape[0]) for layer, (basis, _) in bases.items()
        },
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
