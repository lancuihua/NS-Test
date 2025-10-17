"""
Layer-wise task vector diagnostics.

This script compares a base (e.g., zeroshot) checkpoint against multiple
finetuned “expert” checkpoints and produces per-layer statistics that answer:
  1. Which layers changed the most (energy fraction)?
  2. Do different experts modify the same set of layers (participation score)?
  3. Are the update directions aligned or conflicting (pairwise cosine)?
  4. Are the updates low-rank (entropy-based effective rank)?

Outputs a JSON report (and optional CSV) and prints concise summaries,
helping decide per-layer merge strategies before running actual model merges.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from utils import safe_load_state_dict  # type: ignore  # noqa: E402


@dataclass
class LayerStats:
    layer: str
    norms: List[float]
    energy_fraction: List[float]
    participation: float
    mean_cosine: Optional[float]
    min_cosine: Optional[float]
    max_cosine: Optional[float]
    effective_ranks: List[float]


def compile_layer_filter(patterns: Iterable[str]) -> List[re.Pattern]:
    return [re.compile(p) for p in patterns]


def filter_parameter_names(
    base_state: Dict[str, torch.Tensor], filters: List[re.Pattern]
) -> List[str]:
    names: List[str] = []
    for name, tensor in base_state.items():
        if not torch.is_floating_point(tensor):
            continue
        if filters and not any(p.search(name) for p in filters):
            continue
        names.append(name)
    return names


def load_state(path: Path) -> Dict[str, torch.Tensor]:
    state = safe_load_state_dict(str(path))
    result: Dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        if torch.is_floating_point(tensor):
            result[key] = tensor.detach().to(torch.float32)
    return result


def flatten_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 1:
        return tensor
    return tensor.reshape(-1)


def cosine_between(a: torch.Tensor, b: torch.Tensor, eps: float) -> Optional[float]:
    an = torch.linalg.vector_norm(a)
    bn = torch.linalg.vector_norm(b)
    if an <= eps or bn <= eps:
        return None
    return float(torch.dot(a, b) / (an * bn))


def effective_rank(tensor: torch.Tensor, eps: float) -> float:
    if tensor.ndim <= 1:
        return float(tensor.count_nonzero().item())
    try:
        singular_vals = torch.linalg.svdvals(tensor)
    except RuntimeError:
        return float("nan")
    if singular_vals.numel() == 0:
        return 0.0
    powers = singular_vals.square()
    total = powers.sum().item()
    if total <= eps:
        return 0.0
    probs = (powers / (total + eps)).clamp_min(eps)
    entropy = -(probs * torch.log(probs)).sum().item()
    return float(math.exp(entropy))


def build_report(
    base_path: Path,
    expert_paths: Sequence[Path],
    layer_regex: List[str],
    energy_threshold: float,
    eps: float,
) -> Tuple[List[LayerStats], Dict[str, float]]:
    base_state = load_state(base_path)
    experts = [load_state(p) for p in expert_paths]

    layer_filters = compile_layer_filter(layer_regex)
    layer_names = filter_parameter_names(base_state, layer_filters)
    if not layer_names:
        raise ValueError("No parameters matched the provided layer patterns.")

    num_experts = len(experts)

    # Pre-compute per-expert energy denominators.
    total_energy: List[float] = [0.0 for _ in range(num_experts)]
    layer_norms: Dict[str, List[float]] = {name: [0.0] * num_experts for name in layer_names}
    layer_vectors: Dict[str, List[Optional[torch.Tensor]]] = {
        name: [None] * num_experts for name in layer_names
    }

    for idx, expert in enumerate(experts):
        energy_sum = 0.0
        for name in layer_names:
            base_tensor = base_state[name]
            expert_tensor = expert.get(name)
            if expert_tensor is None or expert_tensor.shape != base_tensor.shape:
                continue
            delta = (expert_tensor - base_tensor).to(torch.float32)
            norm_sq = float(torch.linalg.vector_norm(delta).square().item())
            energy_sum += norm_sq
            layer_norms[name][idx] = math.sqrt(norm_sq)
            layer_vectors[name][idx] = flatten_tensor(delta).cpu()
        total_energy[idx] = energy_sum

    layer_stats: List[LayerStats] = []
    summary_energy: Dict[str, float] = {}

    for name in layer_names:
        norms = layer_norms[name]
        vecs = layer_vectors[name]

        energy_fraction: List[float] = []
        mask: List[int] = []
        for idx, norm_val in enumerate(norms):
            denom = total_energy[idx]
            frac = 0.0
            if denom > 0.0:
                frac = (norm_val ** 2) / denom
            energy_fraction.append(frac)
            mask.append(1 if frac >= energy_threshold else 0)

        participation = sum(mask) / max(num_experts, 1)
        summary_energy[name] = sum(energy_fraction) / max(num_experts, 1)

        cos_values: List[float] = []
        for i in range(num_experts):
            vi = vecs[i]
            if vi is None:
                continue
            for j in range(i + 1, num_experts):
                vj = vecs[j]
                if vj is None:
                    continue
                cos = cosine_between(vi, vj, eps=eps)
                if cos is not None:
                    cos_values.append(cos)

        mean_cos = float(sum(cos_values) / len(cos_values)) if cos_values else None
        min_cos = float(min(cos_values)) if cos_values else None
        max_cos = float(max(cos_values)) if cos_values else None

        ranks: List[float] = []
        for idx, tensor in enumerate(layer_vectors[name]):
            if tensor is None:
                ranks.append(float("nan"))
                continue
            original = layer_norms[name][idx]
            if original <= eps:
                ranks.append(0.0)
            else:
                expert_tensor = experts[idx].get(name)
                if expert_tensor is None or expert_tensor.shape != base_state[name].shape:
                    ranks.append(float("nan"))
                    continue
                shaped = (expert_tensor - base_state[name]).to(torch.float32)
                ranks.append(effective_rank(shaped, eps=eps))

        stats = LayerStats(
            layer=name,
            norms=norms,
            energy_fraction=energy_fraction,
            participation=participation,
            mean_cosine=mean_cos,
            min_cosine=min_cos,
            max_cosine=max_cos,
            effective_ranks=ranks,
        )
        layer_stats.append(stats)

    layer_stats.sort(key=lambda s: summary_energy[s.layer], reverse=True)
    return layer_stats, summary_energy


def write_report_json(path: Path, layers: List[LayerStats], summary: Dict[str, float]) -> None:
    payload = {
        "layer_order": [stat.layer for stat in layers],
        "layers": [
            {
                "layer": stat.layer,
                "norms": stat.norms,
                "energy_fraction": stat.energy_fraction,
                "participation": stat.participation,
                "mean_cosine": stat.mean_cosine,
                "min_cosine": stat.min_cosine,
                "max_cosine": stat.max_cosine,
                "effective_ranks": stat.effective_ranks,
            }
            for stat in layers
        ],
        "mean_energy_fraction": summary,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_report_csv(path: Path, layers: List[LayerStats], expert_names: Sequence[str]) -> None:
    headers = [
        "layer",
        *[f"norm_{name}" for name in expert_names],
        *[f"energy_fraction_{name}" for name in expert_names],
        "participation",
        "mean_cosine",
        "min_cosine",
        "max_cosine",
        *[f"effective_rank_{name}" for name in expert_names],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for stat in layers:
            row: List[object] = [stat.layer]
            row.extend(f"{val:.6f}" for val in stat.norms)
            row.extend(f"{val:.6f}" for val in stat.energy_fraction)
            row.append(f"{stat.participation:.4f}")
            row.append("" if stat.mean_cosine is None else f"{stat.mean_cosine:.4f}")
            row.append("" if stat.min_cosine is None else f"{stat.min_cosine:.4f}")
            row.append("" if stat.max_cosine is None else f"{stat.max_cosine:.4f}")
            row.extend("" if math.isnan(rank) else f"{rank:.4f}" for rank in stat.effective_ranks)
            writer.writerow(row)


def summarize_to_stdout(layers: List[LayerStats], top_k: int) -> None:
    print("=== Layer change summary (energy fraction sorted) ===")
    for stat in layers[:top_k]:
        avg_energy = sum(stat.energy_fraction) / max(len(stat.energy_fraction), 1)
        mean_cos_str = "n/a" if stat.mean_cosine is None else f"{stat.mean_cosine:.3f}"
        print(
            f"{stat.layer}: avg_energy={avg_energy:.4f}, "
            f"participation={stat.participation:.2f}, mean_cosine={mean_cos_str}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layer-wise weight delta diagnostics.")
    parser.add_argument("--base", required=True, help="Path to the base (e.g., zeroshot) checkpoint.")
    parser.add_argument("--experts", nargs="+", required=True, help="Paths to finetuned expert checkpoints.")
    parser.add_argument(
        "--layer_regex",
        nargs="*",
        default=[
            r".*attn.*weight",
            r".*mlp.*weight",
        ],
        help="Regex filters selecting parameters to include.",
    )
    parser.add_argument(
        "--energy_threshold",
        type=float,
        default=0.05,
        help="Threshold on per-task energy fraction for participation (default: 0.05).",
    )
    parser.add_argument("--eps", type=float, default=1e-8, help="Numerical epsilon for norms and ranks.")
    parser.add_argument("--output_json", type=Path, default=None, help="Optional path to store JSON report.")
    parser.add_argument("--output_csv", type=Path, default=None, help="Optional path to store CSV table.")
    parser.add_argument("--top_k_print", type=int, default=15, help="Number of layers to print in the summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.energy_threshold < 0 or args.energy_threshold > 1:
        raise ValueError("--energy_threshold must be between 0 and 1.")

    layers, summary = build_report(
        base_path=Path(args.base),
        expert_paths=[Path(p) for p in args.experts],
        layer_regex=args.layer_regex,
        energy_threshold=args.energy_threshold,
        eps=args.eps,
    )

    summarize_to_stdout(layers, top_k=max(args.top_k_print, 1))

    if args.output_json is not None:
        write_report_json(args.output_json, layers, summary)
        print(f"JSON report written to {args.output_json}")
    if args.output_csv is not None:
        write_report_csv(args.output_csv, layers, [Path(p).stem for p in args.experts])
        print(f"CSV table written to {args.output_csv}")


if __name__ == "__main__":
    main()
