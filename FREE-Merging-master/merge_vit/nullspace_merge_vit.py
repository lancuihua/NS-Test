"""
Null-space constrained merging for vision transformers.

This script mirrors the FREE-Merging ViT pipeline, but projects expert
updates onto the null space of a preserve dataset before fusing them into the
base checkpoint. The idea follows AlphaEdit: directions that would change the
outputs on critical samples are removed prior to merging.

Workflow:
  1. Load the base image encoder checkpoint (e.g. zeroshot.pt) and compute
     per-layer gradients on a preserve image set.
  2. Build per-parameter null-space bases via SVD on the stacked gradients.
  3. Load finetuned expert checkpoints, project their deltas into the null
     space, and blend them with user-specified weights.
  4. Save the merged model and metadata for further evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
import open_clip
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm

# Allow importing helper utilities from merge_vit's src package.
THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import (
    safe_load_state_dict,
    read_config,
    get_dataset_name,
)  # type: ignore  # noqa: E402
from eval import eval_single_dataset_30  # type: ignore  # noqa: E402


def compile_layer_filter(patterns: Iterable[str]) -> List[re.Pattern]:
    compiled: List[re.Pattern] = []
    for pattern in patterns:
        compiled.append(re.compile(pattern))
    return compiled


def _flatten_like(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(1, -1)


def _reshape_like(row: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return row.reshape(like.shape)


def filter_parameters(
    model: torch.nn.Module, filters: List[re.Pattern]
) -> Dict[str, torch.nn.Parameter]:
    """Select parameters whose names match any regex pattern."""
    selected: Dict[str, torch.nn.Parameter] = {}
    for name, param in model.named_parameters():
        if not torch.is_floating_point(param):
            continue
        if filters and not any(p.search(name) for p in filters):
            continue
        selected[name] = param
    if filters and not selected:
        raise ValueError(
            "No parameters matched the provided --layer_regex patterns. "
            "Consider relaxing the regex expressions."
        )
    return selected


def load_image_dataset(
    dataset_path: Path,
    transform,
    max_samples: Optional[int],
    seed: int,
) -> ImageFolder | Subset[ImageFolder]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Preserve dataset path not found: {dataset_path}")
    dataset = ImageFolder(str(dataset_path), transform=transform)
    if len(dataset) == 0:
        raise ValueError(f"No images found under {dataset_path}.")
    if max_samples is not None and max_samples < len(dataset):
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=generator)[:max_samples]
        dataset = Subset(dataset, indices.tolist())
    return dataset


def create_dataloader(
    dataset,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )


def gather_preserve_gradients(
    model: torch.nn.Module,
    dataloader: DataLoader,
    parameter_pool: Dict[str, torch.nn.Parameter],
    device: torch.device,
) -> Dict[str, List[torch.Tensor]]:
    model.eval()
    gradients: Dict[str, List[torch.Tensor]] = {name: [] for name in parameter_pool}

    for images, _ in tqdm(dataloader, desc="Collecting preserve gradients", leave=False):
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            targets = model(images).detach()

        model.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = F.mse_loss(outputs, targets)
        loss.backward()

        for name, param in parameter_pool.items():
            grad = param.grad
            if grad is None:
                continue
            gradients[name].append(grad.detach().flatten().cpu())

    return gradients


def compute_nullspace_bases(
    gradients: Dict[str, List[torch.Tensor]],
    max_rank: Optional[int],
    svd_threshold: float,
) -> Dict[str, torch.Tensor]:
    """Compute per-parameter null-space basis via SVD."""
    bases: Dict[str, torch.Tensor] = {}
    for name, grad_list in gradients.items():
        if not grad_list:
            continue
        G = torch.stack(grad_list, dim=0).float()
        try:
            _, S, Vh = torch.linalg.svd(G, full_matrices=False)
        except RuntimeError as exc:  # pragma: no cover - unlikely SVD failure
            raise RuntimeError(f"SVD failed for parameter {name}") from exc

        keep_mask = torch.ones_like(S, dtype=torch.bool)
        if svd_threshold > 0:
            keep_mask &= (S / S.max()) > svd_threshold
        if max_rank is not None:
            keep_mask &= torch.arange(len(S)) < max_rank

        rank = int(keep_mask.sum().item())
        if rank == 0:
            continue
        bases[name] = Vh[:rank, :].contiguous()
    return bases


def compute_svd_preserve_bases(
    model: torch.nn.Module,
    parameter_pool: Dict[str, torch.nn.Parameter],
    max_rank: int,
) -> Dict[str, torch.Tensor]:
    """Build preservation bases from dominant singular directions of parameters."""
    bases: Dict[str, torch.Tensor] = {}
    state = dict(model.named_parameters())
    for name in parameter_pool:
        tensor = state[name].detach()
        if tensor.ndim < 2:
            continue
        flat = tensor.view(tensor.shape[0], -1).float()
        rank = min(max_rank, flat.shape[0], flat.shape[1])
        if rank <= 0:
            continue
        try:
            _, _, Vh = torch.linalg.svd(flat, full_matrices=False)
        except RuntimeError:
            continue
        bases[name] = Vh[:rank].contiguous()
    return bases


def project_delta(
    delta: torch.Tensor,
    basis: Optional[torch.Tensor],
    strength: float,
) -> torch.Tensor:
    if basis is None or basis.numel() == 0:
        return delta
    # Match the reshaping used in compute_svd_preserve_bases
    original_shape = delta.shape
    if delta.ndim < 2:
        flat = delta.view(-1).unsqueeze(0)  # Make it 2D for consistency
    else:
        flat = delta.view(delta.shape[0], -1)  # Match SVD computation: (shape[0], -1)
    
    B = basis.to(flat.device)  # B shape: (rank, feature_dim)
    # Project each row of flat
    coeffs = torch.matmul(flat, B.T)  # (batch, rank)
    correction = torch.matmul(coeffs, B)  # (batch, feature_dim)
    flat_proj = flat - strength * correction
    return flat_proj.view(original_shape)

def _project_onto_basis(row: torch.Tensor, basis: Optional[torch.Tensor]) -> torch.Tensor:
    if basis is None or basis.numel() == 0:
        return row
    B = basis.to(row.device)
    return (row @ B.t()) @ B


def _remove_basis_component(row: torch.Tensor, basis: Optional[torch.Tensor]) -> torch.Tensor:
    if basis is None or basis.numel() == 0:
        return row
    B = basis.to(row.device)
    return row - (row @ B.t()) @ B


def _proj_guard_and_shared(row: torch.Tensor, guard_basis: Optional[torch.Tensor], shared_basis: Optional[torch.Tensor]) -> torch.Tensor:
    v = row
    v = _project_onto_basis(v, shared_basis)
    v = _remove_basis_component(v, guard_basis)
    return v


def _build_shared_subspace(rows: List[torch.Tensor], rank: int) -> Optional[torch.Tensor]:
    if rank <= 0 or not rows:
        return None
    X = torch.cat(rows, dim=0)
    if X.shape[0] == 1:
        vec = F.normalize(X[0], dim=0, eps=1e-6).unsqueeze(0)
        return vec
    Xc = X - X.mean(dim=0, keepdim=True)
    try:
        _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
    except RuntimeError:
        return None
    if Vh.shape[0] == 0:
        return None
    r = min(rank, Vh.shape[0])
    if r <= 0:
        return None
    return Vh[:r].contiguous()


def merge_with_nullspace(
    base_state: Dict[str, torch.Tensor],
    expert_states: List[Tuple[Dict[str, torch.Tensor], float]],
    bases: Dict[str, torch.Tensor],
    strength: float,
    apgd_steps: int,
    apgd_lr: float,
    shared_rank: int,
    alpha_mode: str,
    alpha_beta: float,
) -> Dict[str, torch.Tensor]:
    merged: Dict[str, torch.Tensor] = {}
    for name, base_tensor in base_state.items():
        if not torch.is_floating_point(base_tensor):
            merged[name] = base_tensor.clone()
            continue

        base_f32 = base_tensor.to(torch.float32)
        basis = bases.get(name)
        rows: List[torch.Tensor] = []
        row_weights: List[float] = []

        for expert_state, weight in expert_states:
            if name not in expert_state:
                continue
            expert_tensor = expert_state[name].to(torch.float32)
            if expert_tensor.shape != base_f32.shape:
                print(
                    f"[warn] Shape mismatch for parameter '{name}': base {tuple(base_f32.shape)} vs expert {tuple(expert_tensor.shape)}. Skipping this expert.",
                    flush=True,
                )
                continue
            delta = expert_tensor - base_f32
            delta = project_delta(delta, basis, strength)
            row = _flatten_like(delta)
            rows.append(row)
            row_weights.append(weight)

        if not rows:
            merged[name] = base_tensor.clone()
            continue

        rows_tensor = torch.cat(rows, dim=0)  # K x F
        weight_tensor = torch.tensor(row_weights, device=rows_tensor.device, dtype=rows_tensor.dtype).view(-1, 1)
        weight_tensor = weight_tensor / weight_tensor.sum().clamp_min(1e-12)
        delta_row = (weight_tensor * rows_tensor).sum(dim=0, keepdim=True)

        shared_basis = _build_shared_subspace(rows, shared_rank)

        if apgd_steps > 0:
            for _ in range(apgd_steps):
                if alpha_mode == "uniform":
                    lam = torch.ones(rows_tensor.shape[0], device=rows_tensor.device, dtype=rows_tensor.dtype)
                else:
                    d2 = torch.sum((rows_tensor - delta_row) ** 2, dim=1)
                    if alpha_mode == "distance":
                        lam = torch.softmax(alpha_beta * d2, dim=0)
                    elif alpha_mode == "nci_balanced":
                        lam = torch.softmax(alpha_beta * d2, dim=0)
                        mu = rows_tensor.mean(dim=0, keepdim=True)
                        cos = (F.normalize(rows_tensor, dim=1) @ F.normalize(mu, dim=1).t()).squeeze(1).clamp_min(1e-6)
                        lam = lam / cos
                    else:
                        lam = torch.ones(rows_tensor.shape[0], device=rows_tensor.device, dtype=rows_tensor.dtype)
                lam = lam / lam.sum().clamp_min(1e-12)
                grad = 2.0 * torch.sum(lam.view(-1, 1) * (delta_row - rows_tensor), dim=0, keepdim=True)
                delta_row = delta_row - apgd_lr * grad
                delta_row = _proj_guard_and_shared(delta_row, basis, shared_basis)
        else:
            delta_row = _proj_guard_and_shared(delta_row, basis, shared_basis)

        delta_tensor = _reshape_like(delta_row, base_f32)
        merged[name] = (base_f32 + delta_tensor).to(base_tensor.dtype)
    return merged


def prepare_eval_args(
    config_root: Optional[str],
    model_name: Optional[str],
    task_name: Optional[str],
    method_tag: str,
    device: torch.device,
    dataset_override: Optional[List[str]],
    batch_size_override: Optional[int],
    num_workers_override: Optional[int],
) -> Tuple[Optional[argparse.Namespace], Optional[List[str]]]:
    if config_root is None:
        return None, None
    if model_name is None or task_name is None:
        raise ValueError(
            "Both --eval_model and --eval_task must be provided when --eval_config_root is set."
        )

    eval_namespace = argparse.Namespace(
        config_root_path=config_root,
        model=model_name,
        task=task_name,
        method=method_tag,
    )
    eval_args = read_config(eval_namespace)
    eval_args.device = str(device)

    if batch_size_override is not None:
        eval_args.batch_size = batch_size_override
    if num_workers_override is not None:
        eval_args.num_workers = num_workers_override

    datasets = dataset_override or get_dataset_name(eval_args)
    if not datasets:
        raise ValueError(
            "No evaluation datasets were found. Provide --eval_datasets to specify a list."
        )
    eval_args.eval_datasets = datasets
    return eval_args, datasets


def evaluate_merged_encoder(
    merged_state: Dict[str, torch.Tensor],
    base_checkpoint: str,
    eval_args: argparse.Namespace,
    datasets: List[str],
    device: torch.device,
) -> Dict[str, float]:
    image_encoder = torch.load(base_checkpoint, map_location="cpu")
    image_encoder.load_state_dict(merged_state, strict=False)
    image_encoder = image_encoder.to(device)
    image_encoder.eval()

    results: Dict[str, float] = {}
    for dataset in datasets:
        metrics = eval_single_dataset_30(image_encoder, dataset, eval_args)
        top1 = metrics.get("top1")
        if top1 is None:
            raise ValueError(f"Evaluation metrics for {dataset} missing 'top1' key: {metrics}")
        results[dataset] = float(top1)
    return results


def load_base_model(
    checkpoint: str,
    device: torch.device,
    dtype: torch.dtype,
    clip_arch: str,
    clip_pretrained: Optional[str],
) -> torch.nn.Module:
    obj = torch.load(checkpoint, map_location="cpu")
    if isinstance(obj, torch.nn.Module):
        model = obj
    else:
        state_dict = obj.get("state_dict", obj)
        model, _, _ = open_clip.create_model_and_transforms(
            clip_arch,
            pretrained=clip_pretrained,
            device=torch.device("cpu"),
        )
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[warn] Missing keys when loading base checkpoint: {missing}")
        if unexpected:
            print(f"[warn] Unexpected keys when loading base checkpoint: {unexpected}")
    return model.to(device=device, dtype=dtype)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Null-space constrained merging for ViT checkpoints."
    )
    parser.add_argument(
        "--base_checkpoint",
        required=True,
        help="Path to the base image encoder checkpoint (e.g., zeroshot.pt).",
    )
    parser.add_argument(
        "--experts",
        nargs="+",
        required=True,
        help="List of finetuned expert checkpoints to merge.",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        required=True,
        help="Merge weights (must align with --experts).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save merged artifacts.",
    )
    parser.add_argument(
        "--preserve_data",
        type=Path,
        default=None,
        help="Directory containing preserve images structured for torchvision ImageFolder.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=512,
        help="Maximum number of preserve images used to build the null space.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Mini-batch size for preserve gradient collection.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Dataloader worker count.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Seed used when sub-sampling preserve data.",
    )
    parser.add_argument(
        "--nullspace_strength",
        type=float,
        default=1.0,
        help="Scale of the null-space correction (1.0 fully removes preserved directions).",
    )
    parser.add_argument(
        "--svd_threshold",
        type=float,
        default=1e-3,
        help="Relative singular value threshold for retaining basis vectors.",
    )
    parser.add_argument(
        "--max_rank",
        type=int,
        default=None,
        help="Optional cap on null-space rank per parameter.",
    )
    parser.add_argument(
        "--preserve_mode",
        choices=["dataset", "svd"],
        default="dataset",
        help="Strategy for constructing preservation bases. 'dataset' builds AlphaEdit-style bases from gradients; 'svd' protects dominant singular directions of weights (data-free).",
    )
    parser.add_argument(
        "--svd_rank",
        type=int,
        default=8,
        help="Number of dominant singular directions to preserve when --preserve_mode=svd.",
    )
    parser.add_argument(
        "--apgd_steps",
        type=int,
        default=0,
        help="Number of projected gradient refinement steps in the null-space (0 disables APGD).",
    )
    parser.add_argument(
        "--apgd_lr",
        type=float,
        default=0.5,
        help="Step size for APGD refinement.",
    )
    parser.add_argument(
        "--shared_rank",
        type=int,
        default=8,
        help="Rank of task-shared subspace used during APGD refinement.",
    )
    parser.add_argument(
        "--alpha_mode",
        choices=["uniform", "distance", "nci_balanced"],
        default="distance",
        help="Adaptive coefficient strategy for APGD.",
    )
    parser.add_argument(
        "--alpha_beta",
        type=float,
        default=1.0,
        help="Sharpness parameter for adaptive coefficients when alpha_mode != 'uniform'.",
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default="ViT-B-32",
        help="CLIP vision backbone name used when instantiating models from state dict checkpoints.",
    )
    parser.add_argument(
        "--clip_pretrained",
        type=str,
        default=None,
        help="Optional CLIP pretrained identifier; set to 'openai' for official weights when reconstructing models from state dicts.",
    )
    parser.add_argument(
        "--layer_regex",
        nargs="*",
        default=[
            r".*attn.*weight",
            r".*mlp.*weight",
            r".*attn.*bias",
            r".*mlp.*bias",
        ],
        help="Regex patterns selecting parameters to protect. Empty list keeps all parameters.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for computing preserve gradients.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float16",
        help="Data type used when loading expert checkpoints.",
    )
    parser.add_argument(
        "--save_state_only",
        action="store_true",
        help="If set, only save the merged state dict instead of a serialized model.",
    )
    parser.add_argument(
        "--eval_config_root",
        type=str,
        default=None,
        help="Optional config directory used to build evaluation args (e.g., merge_vit/config).",
    )
    parser.add_argument(
        "--eval_model",
        type=str,
        default=None,
        help="Model name (e.g., ViT-B-32) for evaluation config lookup.",
    )
    parser.add_argument(
        "--eval_task",
        type=str,
        default=None,
        help="Task identifier matching the YAML config (e.g., 8 or 30).",
    )
    parser.add_argument(
        "--eval_method",
        type=str,
        default="FREE",
        help="Method tag used when resolving the evaluation YAML file.",
    )
    parser.add_argument(
        "--eval_datasets",
        nargs="*",
        default=None,
        help="Optional subset of datasets to evaluate on; defaults to config-defined set.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=None,
        help="Override evaluation batch size.",
    )
    parser.add_argument(
        "--eval_num_workers",
        type=int,
        default=None,
        help="Override evaluation dataloader worker count.",
    )
    parser.add_argument(
        "--eval_output",
        type=Path,
        default=None,
        help="Path to store evaluation metrics JSON; defaults to <output_dir>/nullspace_eval.json.",
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip running accuracy evaluation after merging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if len(args.experts) != len(args.weights):
        raise ValueError("--experts and --weights must have matching lengths.")

    weight_sum = sum(args.weights)
    if math.isclose(weight_sum, 0.0):
        raise ValueError("Sum of merge weights must be non-zero.")
    weights = [w / weight_sum for w in args.weights]

    device = torch.device(args.device)

    # Load base model for gradient collection (float32 for stability).
    base_model = load_base_model(
        checkpoint=args.base_checkpoint,
        device=device,
        dtype=torch.float32,
        clip_arch=args.clip_model,
        clip_pretrained=args.clip_pretrained,
    )
    base_model.eval()

    # Determine preserve transform from the model if available.
    if hasattr(base_model, "val_preprocess"):
        transform = base_model.val_preprocess
    elif hasattr(getattr(base_model, "image_encoder", None), "val_preprocess"):
        transform = base_model.image_encoder.val_preprocess
    else:
        # Fallback CLIP-style preprocessing.
        transform = T.Compose(
            [
                T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    parameter_filters = compile_layer_filter(args.layer_regex)
    parameter_pool = filter_parameters(base_model, parameter_filters)
    bases: Dict[str, torch.Tensor]

    if args.preserve_mode == "dataset":
        if not args.preserve_data:
            raise ValueError("--preserve_data is required when --preserve_mode=dataset.")
        dataset = load_image_dataset(
            dataset_path=args.preserve_data,
            transform=transform,
            max_samples=args.max_samples,
            seed=args.seed,
        )
        dataloader = create_dataloader(
            dataset=dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        gradients = gather_preserve_gradients(
            model=base_model,
            dataloader=dataloader,
            parameter_pool=parameter_pool,
            device=device,
        )
        bases = compute_nullspace_bases(
            gradients=gradients,
            max_rank=args.max_rank,
            svd_threshold=args.svd_threshold,
        )
    else:  # svd mode
        rank = args.svd_rank
        if args.max_rank is not None:
            rank = min(rank, args.max_rank)
        bases = compute_svd_preserve_bases(
            model=base_model,
            parameter_pool=parameter_pool,
            max_rank=rank,
        )

    base_state = {
        name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()
    }

    # Offload model before loading expert checkpoints to save memory.
    del parameter_pool
    base_model_cpu = base_model.to("cpu")
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    expert_states: List[Tuple[Dict[str, torch.Tensor], float]] = []
    for expert_path, weight in zip(args.experts, weights):
        state = safe_load_state_dict(expert_path)
        for key, tensor in state.items():
            if torch.is_floating_point(tensor):
                state[key] = tensor.to(torch.float32)
        expert_states.append((state, weight))

    merged_state = merge_with_nullspace(
        base_state=base_state,
        expert_states=expert_states,
        bases=bases,
        strength=args.nullspace_strength,
        apgd_steps=args.apgd_steps,
        apgd_lr=args.apgd_lr,
        shared_rank=args.shared_rank,
        alpha_mode=args.alpha_mode,
        alpha_beta=args.alpha_beta,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_results: Optional[Dict[str, float]] = None
    eval_datasets: Optional[List[str]] = None
    eval_output_path: Optional[Path] = None
    if not args.skip_eval and args.eval_config_root:
        eval_args, eval_datasets = prepare_eval_args(
            config_root=args.eval_config_root,
            model_name=args.eval_model,
            task_name=args.eval_task,
            method_tag=args.eval_method,
            device=device,
            dataset_override=args.eval_datasets,
            batch_size_override=args.eval_batch_size,
            num_workers_override=args.eval_num_workers,
        )
        if eval_args is not None and eval_datasets is not None:
            eval_results = evaluate_merged_encoder(
                merged_state=merged_state,
                base_checkpoint=args.base_checkpoint,
                eval_args=eval_args,
                datasets=eval_datasets,
                device=device,
            )
            eval_output_path = args.eval_output or (output_dir / "nullspace_eval.json")
            with eval_output_path.open("w", encoding="utf-8") as f:
                json.dump(eval_results, f, indent=2)
            print("Post-merge evaluation (top-1 accuracy):")
            for dataset, score in eval_results.items():
                print(f"  {dataset}: {score * 100:.2f}%")
        else:
            print("Evaluation configuration was requested but could not be prepared; skipping accuracy.")

    merged_state_path = output_dir / "nullspace_merged_state.pt"
    torch.save(merged_state, merged_state_path)

    metadata = {
        "base_checkpoint": args.base_checkpoint,
        "experts": args.experts,
        "weights": weights,
        "nullspace_strength": args.nullspace_strength,
        "svd_threshold": args.svd_threshold,
        "max_rank": args.max_rank,
        "preserve_mode": args.preserve_mode,
        "layer_regex": args.layer_regex,
        "protected_layers": {k: int(v.shape[0]) for k, v in bases.items()},
        "apgd_steps": args.apgd_steps,
        "apgd_lr": args.apgd_lr,
        "shared_rank": args.shared_rank,
        "alpha_mode": args.alpha_mode,
        "alpha_beta": args.alpha_beta,
    }
    if args.preserve_mode == "dataset":
        metadata["preserve_data"] = str(args.preserve_data)
        metadata["max_samples"] = args.max_samples
        metadata["batch_size"] = args.batch_size
    else:
        metadata["svd_rank"] = args.svd_rank
    if eval_results is not None and eval_datasets is not None and eval_output_path is not None:
        metadata["evaluation"] = {
            "datasets": eval_datasets,
            "results_path": str(eval_output_path),
            "top1": eval_results,
        }
    with (output_dir / "nullspace_meta.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if not args.save_state_only:
        # Reload a template model, apply the merged weights, and serialize.
        template_model = torch.load(args.base_checkpoint, map_location="cpu")
        template_model.load_state_dict(merged_state, strict=False)
        torch.save(template_model, output_dir / "nullspace_merged.pt")

    # Restore original base model to avoid side-effects.
    base_model_cpu.load_state_dict(base_state, strict=False)


if __name__ == "__main__":
    main()
