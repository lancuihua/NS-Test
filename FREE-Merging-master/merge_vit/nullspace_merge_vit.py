"""
Shared subspace constrained merging for vision transformers.

This script implements the "A method" workflow:
  1. Load a base checkpoint and a list of expert checkpoints with the same
     architecture.
  2. For every selected parameter tensor, build task vectors
     DeltaW = W_expert - W_base and reshape them to [out_dim, -1].
  3. Stack task vectors column-wise, compute their output-side covariance,
     and extract the top-k eigenvectors to form the shared left subspace U_k.
  4. Project every non-anchor task vector onto the orthogonal complement of U_k
     with either a fixed or adaptive strength alpha, then merge with user-specified
     weights.
  5. Optionally remove input-side shared modes and apply spectral balancing to the
     merged parameters for additional stabilization.
  6. Save the merged weights (and optional serialized model) alongside
     diagnostic metadata and optional evaluation metrics.

No preserve dataset is required; all computations are data-free.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import open_clip
import torch

# Allow importing helper utilities from merge_vit's src package.
THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import (  # type: ignore  # noqa: E402
    get_dataset_name,
    read_config,
    safe_load_state_dict,
)
from eval import eval_single_dataset_30  # type: ignore  # noqa: E402


def compile_layer_filter(patterns: Iterable[str]) -> List[re.Pattern]:
    return [re.compile(pattern) for pattern in patterns]


@dataclass
class LayerProjectionInfo:
    left_basis: Optional[torch.Tensor]
    left_rank: int
    left_alpha: float
    left_removed_fraction: float
    left_projected_norm: float
    left_stacked_norm: float
    left_total_energy: float
    right_basis: Optional[torch.Tensor] = None
    right_rank: int = 0
    right_alpha: float = 0.0
    right_removed_fraction: float = 0.0
    right_projected_norm: float = 0.0
    right_stacked_norm: float = 0.0
    num_experts: int = 0
    out_dim: int = 0
    flat_dim: int = 0


def filter_parameter_names(model: torch.nn.Module, filters: List[re.Pattern]) -> List[str]:
    """Return parameter names that are float tensors satisfying the regex filters."""
    selected: List[str] = []
    for name, param in model.named_parameters():
        if not torch.is_floating_point(param):
            continue
        if param.ndim < 2:
            continue  # skip biases and scalar parameters
        if filters and not any(pattern.search(name) for pattern in filters):
            continue
        selected.append(name)
    if filters and not selected:
        raise ValueError(
            "No parameters matched the provided --layer_regex patterns. "
            "Consider relaxing the regex expressions."
        )
    return selected


def normalize_weights(weights: Sequence[float]) -> List[float]:
    if len(weights) == 0:
        raise ValueError("At least one merge weight must be provided.")
    total = float(sum(weights))
    if math.isclose(total, 0.0):
        raise ValueError("Merge weights must sum to a non-zero value.")
    return [float(w) / total for w in weights]


def collect_task_vectors(
    base_state: Dict[str, torch.Tensor],
    expert_states: Sequence[Dict[str, torch.Tensor]],
    parameter_names: Sequence[str],
) -> Dict[str, List[torch.Tensor]]:
    task_vectors: Dict[str, List[torch.Tensor]] = {name: [] for name in parameter_names}
    for name in parameter_names:
        base_tensor = base_state[name]
        out_dim = base_tensor.shape[0]
        for expert_state in expert_states:
            expert_tensor = expert_state.get(name)
            if expert_tensor is None:
                continue
            if expert_tensor.shape != base_tensor.shape:
                raise ValueError(
                    f"Shape mismatch for parameter '{name}': base {tuple(base_tensor.shape)} vs "
                    f"expert {tuple(expert_tensor.shape)}."
                )
            delta = expert_tensor - base_tensor
            task_vectors[name].append(delta.reshape(out_dim, -1))
    return task_vectors


def determine_rank(
    eigenvalues: torch.Tensor,
    max_rank: Optional[int],
    energy_threshold: Optional[float],
) -> int:
    values = eigenvalues.clamp_min(0.0)
    total_energy = float(values.sum().item())
    if total_energy <= 0.0:
        return 0

    if energy_threshold is None:
        rank = values.numel()
    else:
        if not 0.0 < energy_threshold <= 1.0:
            raise ValueError("--shared_energy must be in the interval (0, 1].")
        descending = values.flip(0)
        cumulative = torch.cumsum(descending, dim=0)
        target = torch.tensor(
            energy_threshold * total_energy, device=values.device, dtype=values.dtype
        )
        idx = int(torch.searchsorted(cumulative, target, right=False).item())
        rank = min(idx + 1, descending.numel())

    if max_rank is not None:
        rank = min(rank, max_rank)
    return max(rank, 0)


def compute_alpha(
    mode: str,
    basis: Optional[torch.Tensor],
    stacked: torch.Tensor,
    alpha_value: float,
    alpha_scale: float,
    alpha_min: float,
    alpha_max: float,
    eps: float,
) -> float:
    if basis is None or basis.numel() == 0:
        return 0.0
    if mode == "fixed":
        alpha = alpha_value
    else:
        projected = torch.linalg.vector_norm(basis.t() @ stacked).item()
        total = torch.linalg.vector_norm(stacked).item()
        ratio = 0.0 if total <= eps else projected / (total + eps)
        alpha = alpha_scale * ratio
    alpha = max(alpha_min, min(alpha_max, alpha))
    return float(alpha)


def build_shared_bases(
    task_vectors: Dict[str, List[torch.Tensor]],
    left_max_rank: Optional[int],
    left_energy_threshold: Optional[float],
    left_alpha_mode: str,
    left_alpha_value: float,
    left_alpha_scale: float,
    left_alpha_min: float,
    left_alpha_max: float,
    right_max_rank: Optional[int],
    right_energy_threshold: Optional[float],
    right_alpha_mode: str,
    right_alpha_value: float,
    right_alpha_scale: float,
    right_alpha_min: float,
    right_alpha_max: float,
    eps: float,
) -> Dict[str, LayerProjectionInfo]:
    projection_infos: Dict[str, LayerProjectionInfo] = {}
    for name, matrices in task_vectors.items():
        if not matrices:
            continue
        stacked = torch.cat(matrices, dim=1)  # [out_dim, num_experts * flat_dim]
        covariance = stacked @ stacked.t()
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        except RuntimeError as exc:  # pragma: no cover - numerical fallback
            raise RuntimeError(f"Failed to compute eigendecomposition for '{name}'.") from exc

        left_rank = determine_rank(eigenvalues, left_max_rank, left_energy_threshold)
        left_basis: Optional[torch.Tensor]
        left_projected_norm = 0.0
        left_removed_fraction = 0.0
        stacked_norm = float(torch.linalg.vector_norm(stacked).item())
        if left_rank > 0:
            left_basis = eigenvectors[:, -left_rank:].contiguous()
            projected_component = left_basis @ (left_basis.t() @ stacked)
            left_projected_norm = float(torch.linalg.vector_norm(projected_component).item())
            left_removed_fraction = 0.0 if stacked_norm <= eps else left_projected_norm / (stacked_norm + eps)
        else:
            left_basis = None
        left_alpha = compute_alpha(
            mode=left_alpha_mode,
            basis=left_basis,
            stacked=stacked,
            alpha_value=left_alpha_value,
            alpha_scale=left_alpha_scale,
            alpha_min=left_alpha_min,
            alpha_max=left_alpha_max,
            eps=eps,
        )

        right_basis: Optional[torch.Tensor] = None
        right_rank = 0
        right_projected_norm = 0.0
        right_removed_fraction = 0.0
        right_stacked_norm = 0.0
        if right_max_rank is not None or right_energy_threshold is not None:
            base_matrix = matrices[0]
            flat_dim = base_matrix.shape[1]
            covariance_right = torch.zeros(
                flat_dim,
                flat_dim,
                dtype=base_matrix.dtype,
                device=base_matrix.device,
            )
            for mat in matrices:
                covariance_right += mat.t() @ mat
            try:
                eigenvalues_right, eigenvectors_right = torch.linalg.eigh(covariance_right)
            except RuntimeError as exc:  # pragma: no cover
                raise RuntimeError(f"Failed to compute right-side eigendecomposition for '{name}'.") from exc
            right_rank = determine_rank(eigenvalues_right, right_max_rank, right_energy_threshold)
            if right_rank > 0:
                right_basis = eigenvectors_right[:, -right_rank:].contiguous()
                stacked_right = torch.cat([mat.t() for mat in matrices], dim=1)
                right_stacked_norm = float(torch.linalg.vector_norm(stacked_right).item())
                projected_right = right_basis @ (right_basis.t() @ stacked_right)
                right_projected_norm = float(torch.linalg.vector_norm(projected_right).item())
                right_removed_fraction = (
                    0.0 if right_stacked_norm <= eps else right_projected_norm / (right_stacked_norm + eps)
                )
                right_alpha = compute_alpha(
                    mode=right_alpha_mode,
                    basis=right_basis,
                    stacked=stacked_right,
                    alpha_value=right_alpha_value,
                    alpha_scale=right_alpha_scale,
                    alpha_min=right_alpha_min,
                    alpha_max=right_alpha_max,
                    eps=eps,
                )
            else:
                right_basis = None
                right_alpha = 0.0
        else:
            right_alpha = 0.0

        projection_infos[name] = LayerProjectionInfo(
            left_basis=left_basis,
            left_rank=int(left_rank),
            left_alpha=float(left_alpha),
            left_removed_fraction=float(left_removed_fraction),
            left_projected_norm=left_projected_norm,
            left_stacked_norm=stacked_norm,
            left_total_energy=float(eigenvalues.clamp_min(0.0).sum().item()),
            right_basis=right_basis,
            right_rank=int(right_rank),
            right_alpha=float(right_alpha),
            right_removed_fraction=float(right_removed_fraction),
            right_projected_norm=right_projected_norm,
            right_stacked_norm=right_stacked_norm,
            num_experts=len(matrices),
            out_dim=matrices[0].shape[0],
            flat_dim=matrices[0].shape[1],
        )
    return projection_infos


def project_left(delta: torch.Tensor, info: LayerProjectionInfo) -> torch.Tensor:
    basis = info.left_basis
    if basis is None or info.left_alpha == 0.0:
        return delta
    basis = basis.to(delta.device)
    reshaped = delta.reshape(info.out_dim, -1)
    correction = basis @ (basis.t() @ reshaped)
    projected = reshaped - info.left_alpha * correction
    return projected.reshape_as(delta)


def project_right(delta: torch.Tensor, info: LayerProjectionInfo) -> torch.Tensor:
    basis = info.right_basis
    if basis is None or info.right_alpha == 0.0:
        return delta
    basis = basis.to(delta.device)
    reshaped = delta.reshape(info.out_dim, -1)
    correction = (reshaped @ basis) @ basis.t()
    projected = reshaped - info.right_alpha * correction
    return projected.reshape_as(delta)


def apply_spectral_balance(
    merged_tensor: torch.Tensor,
    expert_weights: List[torch.Tensor],
    mode: str,
    clip_lo: float,
    clip_hi: float,
    eps: float,
) -> torch.Tensor:
    if mode == "none" or merged_tensor.ndim < 2 or not expert_weights:
        return merged_tensor

    out_dim = merged_tensor.shape[0]
    mat_merged = merged_tensor.reshape(out_dim, -1).to(torch.float32)
    min_dim = min(mat_merged.shape[0], mat_merged.shape[1])

    singular_lists: List[torch.Tensor] = []
    for weight in expert_weights:
        if weight.ndim < 2:
            continue
        mat = weight.reshape(out_dim, -1).to(torch.float32)
        try:
            _, s, _ = torch.linalg.svd(mat, full_matrices=False)
        except RuntimeError:
            continue
        singular_lists.append(s[:min_dim])

    if not singular_lists:
        return merged_tensor

    stacked_s = torch.stack(singular_lists, dim=0)

    try:
        U_m, S_m, Vh_m = torch.linalg.svd(mat_merged, full_matrices=False)
    except RuntimeError:
        return merged_tensor

    S_new = S_m.clone()
    if mode == "gmean":
        safe_vals = stacked_s.clamp_min(eps)
        log_vals = torch.log(safe_vals)
        log_mean = log_vals.mean(dim=0)
        target = torch.exp(log_mean)
        S_new[: target.shape[0]] = target
    elif mode == "clip":
        lo = torch.quantile(stacked_s, clip_lo, dim=0)
        hi = torch.quantile(stacked_s, clip_hi, dim=0)
        lo = lo.clamp_min(eps)
        hi = torch.maximum(hi, lo + eps)
        S_new = torch.minimum(torch.maximum(S_new, lo), hi)
    else:
        return merged_tensor

    recon = (U_m * S_new.unsqueeze(0)) @ Vh_m
    return recon.reshape_as(merged_tensor).to(merged_tensor.dtype)


def merge_expert_states(
    base_state: Dict[str, torch.Tensor],
    base_state_f32: Dict[str, torch.Tensor],
    expert_states: Sequence[Dict[str, torch.Tensor]],
    projection_infos: Dict[str, LayerProjectionInfo],
    protected_names: Set[str],
    normalized_weights: Sequence[float],
    anchor_index: Optional[int],
    anchor_scale: float,
    project_anchor: bool,
    spectral_mode: str,
    spectral_clip_lo: float,
    spectral_clip_hi: float,
    spectral_eps: float,
) -> Dict[str, torch.Tensor]:
    merged: Dict[str, torch.Tensor] = {}
    num_experts = len(expert_states)
    if anchor_index is not None and not (0 <= anchor_index < num_experts):
        raise ValueError(f"--anchor_index {anchor_index} is out of range for {num_experts} experts.")

    other_indices: List[int] = (
        [idx for idx in range(num_experts) if idx != anchor_index] if anchor_index is not None else list(range(num_experts))
    )

    for name, base_tensor in base_state.items():
        if not torch.is_floating_point(base_tensor):
            merged[name] = base_tensor.clone()
            continue
        base_f32 = base_state_f32[name]

        deltas: List[Optional[torch.Tensor]] = []
        expert_actuals: List[Optional[torch.Tensor]] = []
        for state in expert_states:
            tensor = state.get(name)
            if tensor is None:
                deltas.append(None)
                expert_actuals.append(None)
                continue
            if tensor.shape != base_tensor.shape:
                raise ValueError(
                    f"Shape mismatch for parameter '{name}': base {tuple(base_tensor.shape)} vs "
                    f"expert {tuple(tensor.shape)}."
                )
            tensor_f32 = tensor.to(torch.float32)
            expert_actuals.append(tensor_f32)
            deltas.append(tensor_f32 - base_f32)

        info = projection_infos.get(name)
        apply_projection = info is not None and name in protected_names
        combined = torch.zeros_like(base_f32)

        if anchor_index is None:
            for idx, weight in enumerate(normalized_weights):
                delta = deltas[idx]
                if delta is None:
                    continue
                if apply_projection and info is not None:
                    delta = project_left(delta, info)
                    delta = project_right(delta, info)
                combined += weight * delta
        else:
            anchor_delta = deltas[anchor_index]
            if anchor_delta is not None and anchor_scale != 0.0:
                if project_anchor and apply_projection and info is not None:
                    projected_anchor = project_left(anchor_delta, info)
                    projected_anchor = project_right(projected_anchor, info)
                    combined += anchor_scale * projected_anchor
                else:
                    combined += anchor_scale * anchor_delta
            for idx in other_indices:
                delta = deltas[idx]
                if delta is None:
                    continue
                weight = float(normalized_weights[idx])
                if apply_projection and info is not None:
                    delta = project_left(delta, info)
                    delta = project_right(delta, info)
                combined += weight * delta

        merged_tensor = base_f32 + combined
        if spectral_mode != "none":
            expert_weights = [w for w in expert_actuals if w is not None]
            merged_tensor = apply_spectral_balance(
                merged_tensor=merged_tensor,
                expert_weights=expert_weights,
                mode=spectral_mode,
                clip_lo=spectral_clip_lo,
                clip_hi=spectral_clip_hi,
                eps=spectral_eps,
            )

        merged[name] = merged_tensor.to(base_tensor.dtype)
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
        raise ValueError("Both --eval_model and --eval_task must be provided when --eval_config_root is set.")

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
    if isinstance(image_encoder, torch.nn.Module):
        image_encoder.load_state_dict(merged_state, strict=False)
    else:
        state_dict = image_encoder.get("state_dict", image_encoder)
        pretrained_id = getattr(eval_args, "clip_pretrained", getattr(eval_args, "pretrained", None))
        image_encoder, _, _ = open_clip.create_model_and_transforms(
            eval_args.model,
            pretrained=pretrained_id,
            device=torch.device("cpu"),
        )
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


def parse_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype '{dtype_name}'. Choose from {list(mapping)}.")
    return mapping[dtype_name]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Shared subspace constrained merging for ViT checkpoints."
    )
    parser.add_argument("--base_checkpoint", required=True, help="Path to the base image encoder checkpoint.")
    parser.add_argument("--experts", nargs="+", required=True, help="List of finetuned expert checkpoints.")
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Merge weights (align with experts). Defaults to uniform weights when omitted.",
    )
    parser.add_argument("--output_dir", required=True, help="Directory to save merged artifacts.")
    parser.add_argument(
        "--layer_regex",
        nargs="*",
        default=[
            r".*attn.*q_proj.weight",
            r".*attn.*k_proj.weight",
            r".*attn.*v_proj.weight",
            r".*attn.*out_proj.weight",
            r".*mlp.*fc1.weight",
            r".*mlp.*fc2.weight",
        ],
        help="Regex patterns selecting parameters to protect. Empty list keeps all float parameters.",
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
        help="Optional CLIP pretrained identifier (e.g., 'openai') for reconstructing models from state dicts.",
    )
    parser.add_argument("--shared_rank", type=int, default=8, help="Maximum rank of the shared left subspace.")
    parser.add_argument(
        "--shared_energy",
        type=float,
        default=None,
        help="Optional cumulative energy threshold (0-1] for selecting shared modes.",
    )
    parser.add_argument(
        "--alpha_mode",
        choices=["fixed", "adaptive"],
        default="adaptive",
        help="Projection strength strategy.",
    )
    parser.add_argument("--alpha_value", type=float, default=1.0, help="Fixed projection strength when alpha_mode=fixed.")
    parser.add_argument("--alpha_scale", type=float, default=1.0, help="Global scale when alpha_mode=adaptive.")
    parser.add_argument("--alpha_min", type=float, default=0.0, help="Minimum projection strength (after scaling).")
    parser.add_argument("--alpha_max", type=float, default=1.0, help="Maximum projection strength (after scaling).")
    parser.add_argument("--right_rank", type=int, default=0, help="Maximum rank of the shared right subspace.")
    parser.add_argument(
        "--right_energy",
        type=float,
        default=None,
        help="Optional cumulative energy threshold (0-1] for selecting right-side modes.",
    )
    parser.add_argument(
        "--right_alpha_mode",
        choices=["fixed", "adaptive"],
        default="fixed",
        help="Projection strength strategy for the right subspace.",
    )
    parser.add_argument("--right_alpha_value", type=float, default=1.0, help="Fixed strength when right_alpha_mode=fixed.")
    parser.add_argument("--right_alpha_scale", type=float, default=1.0, help="Global scale when right_alpha_mode=adaptive.")
    parser.add_argument("--right_alpha_min", type=float, default=0.0, help="Minimum right projection strength.")
    parser.add_argument("--right_alpha_max", type=float, default=1.0, help="Maximum right projection strength.")
    parser.add_argument(
        "--spectral_balance",
        choices=["none", "gmean", "clip"],
        default="none",
        help="Optional spectral balancing post-process (geometric mean or quantile clipping).",
    )
    parser.add_argument("--spectral_clip_lo", type=float, default=0.05, help="Lower quantile for spectral clipping mode.")
    parser.add_argument("--spectral_clip_hi", type=float, default=0.95, help="Upper quantile for spectral clipping mode.")
    parser.add_argument("--eps", type=float, default=1e-6, help="Numerical epsilon for norm ratios.")
    parser.add_argument(
        "--anchor_index",
        type=int,
        default=None,
        help="Optional index of the anchor expert (0-based). Its weight in --weights is ignored during merging.",
    )
    parser.add_argument(
        "--anchor_scale",
        type=float,
        default=1.0,
        help="Multiplier applied to the anchor delta (default 1.0 keeps it unchanged).",
    )
    parser.add_argument(
        "--project_anchor",
        action="store_true",
        help="Project the anchor update as well (default keeps anchor unprojected).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for constructing the shared subspaces.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="Data type for instantiating the base model prior to CPU offload.",
    )
    parser.add_argument("--save_state_only", action="store_true", help="Only save the merged state dict.")
    parser.add_argument("--eval_config_root", type=str, default=None, help="Optional evaluation config directory.")
    parser.add_argument("--eval_model", type=str, default=None, help="Model name used for evaluation configs.")
    parser.add_argument("--eval_task", type=str, default=None, help="Task identifier for evaluation configs.")
    parser.add_argument("--eval_method", type=str, default="FREE", help="Method tag for evaluation config lookup.")
    parser.add_argument(
        "--eval_datasets",
        nargs="*",
        default=None,
        help="Optional subset of datasets to evaluate on; defaults to config-defined set.",
    )
    parser.add_argument("--eval_batch_size", type=int, default=None, help="Override evaluation batch size.")
    parser.add_argument("--eval_num_workers", type=int, default=None, help="Override evaluation dataloader workers.")
    parser.add_argument("--eval_output", type=Path, default=None, help="Path to store evaluation metrics JSON.")
    parser.add_argument("--skip_eval", action="store_true", help="Skip post-merge evaluation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.weights is None:
        weights = [1.0] * len(args.experts)
    else:
        if len(args.experts) != len(args.weights):
            raise ValueError("--experts and --weights must have matching lengths.")
        weights = [float(w) for w in args.weights]
    if not weights:
        raise ValueError("At least one expert checkpoint must be provided.")
    if args.shared_energy is not None and not (0.0 < args.shared_energy <= 1.0):
        raise ValueError("--shared_energy must lie in the interval (0, 1].")
    if args.right_energy is not None and not (0.0 < args.right_energy <= 1.0):
        raise ValueError("--right_energy must lie in the interval (0, 1].")
    if args.alpha_min > args.alpha_max:
        raise ValueError("--alpha_min must be less than or equal to --alpha_max.")
    if args.right_alpha_min > args.right_alpha_max:
        raise ValueError("--right_alpha_min must be less than or equal to --right_alpha_max.")
    if args.spectral_balance == "clip":
        if not (0.0 <= args.spectral_clip_lo < args.spectral_clip_hi <= 1.0):
            raise ValueError("--spectral_clip_lo and --spectral_clip_hi must satisfy 0 <= lo < hi <= 1.")

    normalized_weights = normalize_weights(weights)
    device = torch.device(args.device)
    dtype = parse_dtype(args.dtype)

    base_model = load_base_model(
        checkpoint=args.base_checkpoint,
        device=device,
        dtype=dtype,
        clip_arch=args.clip_model,
        clip_pretrained=args.clip_pretrained,
    )
    base_model.eval()

    parameter_filters = compile_layer_filter(args.layer_regex)
    protected_names = set(filter_parameter_names(base_model, parameter_filters))

    state_dict = base_model.state_dict()
    base_state = {name: tensor.detach().cpu() for name, tensor in state_dict.items()}
    base_state_f32 = {
        name: tensor.to(torch.float32)
        for name, tensor in base_state.items()
        if torch.is_floating_point(tensor)
    }

    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    expert_states: List[Dict[str, torch.Tensor]] = []
    for expert_path in args.experts:
        state = safe_load_state_dict(expert_path)
        cast_state: Dict[str, torch.Tensor] = {}
        for key, tensor in state.items():
            if torch.is_floating_point(tensor):
                cast_state[key] = tensor.detach().to(torch.float32).cpu()
            else:
                cast_state[key] = tensor
        expert_states.append(cast_state)

    task_vectors = collect_task_vectors(base_state_f32, expert_states, sorted(protected_names))
    left_max_rank = args.shared_rank if args.shared_rank > 0 else None
    right_max_rank = args.right_rank if args.right_rank > 0 else None
    shared_energy = args.shared_energy if args.shared_energy is None else float(args.shared_energy)
    right_energy = args.right_energy if args.right_energy is None else float(args.right_energy)
    projection_infos = build_shared_bases(
        task_vectors=task_vectors,
        left_max_rank=left_max_rank,
        left_energy_threshold=shared_energy,
        left_alpha_mode=args.alpha_mode,
        left_alpha_value=args.alpha_value,
        left_alpha_scale=args.alpha_scale,
        left_alpha_min=args.alpha_min,
        left_alpha_max=args.alpha_max,
        right_max_rank=right_max_rank,
        right_energy_threshold=right_energy,
        right_alpha_mode=args.right_alpha_mode,
        right_alpha_value=args.right_alpha_value,
        right_alpha_scale=args.right_alpha_scale,
        right_alpha_min=args.right_alpha_min,
        right_alpha_max=args.right_alpha_max,
        eps=args.eps,
    )

    if projection_infos:
        print("Shared subspace summary:")
        for name in sorted(projection_infos):
            info = projection_infos[name]
            left_pct = info.left_removed_fraction * 100.0
            message = (
                f"  {name}: left_rank={info.left_rank}, left_alpha={info.left_alpha:.3f}, "
                f"left_removed={left_pct:.2f}% (||stack||={info.left_stacked_norm:.3f})"
            )
            if info.right_basis is not None and info.right_rank > 0:
                right_pct = info.right_removed_fraction * 100.0
                message += (
                    f", right_rank={info.right_rank}, right_alpha={info.right_alpha:.3f}, "
                    f"right_removed={right_pct:.2f}%"
                )
            print(message)
    else:
        print("No shared subspaces were constructed; proceeding with linear merging.")

    merged_state = merge_expert_states(
        base_state=base_state,
        base_state_f32=base_state_f32,
        expert_states=expert_states,
        projection_infos=projection_infos,
        protected_names=protected_names,
        normalized_weights=normalized_weights,
        anchor_index=args.anchor_index,
        anchor_scale=args.anchor_scale,
        project_anchor=args.project_anchor,
        spectral_mode=args.spectral_balance,
        spectral_clip_lo=args.spectral_clip_lo,
        spectral_clip_hi=args.spectral_clip_hi,
        spectral_eps=args.eps,
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
            eval_output_path = args.eval_output or (output_dir / "shared_subspace_eval.json")
            with eval_output_path.open("w", encoding="utf-8") as f:
                json.dump(eval_results, f, indent=2)
            print("Post-merge evaluation (top-1 accuracy):")
            for dataset, score in eval_results.items():
                print(f"  {dataset}: {score * 100:.2f}%")
        else:
            print("Evaluation configuration was requested but could not be prepared; skipping accuracy.")

    merged_state_path = output_dir / "shared_subspace_merged_state.pt"
    torch.save(merged_state, merged_state_path)

    layer_stats: Dict[str, Dict[str, object]] = {}
    for name, info in projection_infos.items():
        stats_entry: Dict[str, Optional[Dict[str, float]]] = {
            "left": {
                "rank": int(info.left_rank),
                "alpha": float(info.left_alpha),
                "removed_fraction": float(info.left_removed_fraction),
                "projected_norm": float(info.left_projected_norm),
                "stacked_norm": float(info.left_stacked_norm),
                "total_energy": float(info.left_total_energy),
            },
            "right": None,
            "num_experts": int(info.num_experts),
            "out_dim": int(info.out_dim),
            "flat_dim": int(info.flat_dim),
        }
        if info.right_basis is not None and info.right_rank > 0:
            stats_entry["right"] = {
                "rank": int(info.right_rank),
                "alpha": float(info.right_alpha),
                "removed_fraction": float(info.right_removed_fraction),
                "projected_norm": float(info.right_projected_norm),
                "stacked_norm": float(info.right_stacked_norm),
            }
        layer_stats[name] = stats_entry

    metadata = {
        "base_checkpoint": args.base_checkpoint,
        "experts": args.experts,
        "weights": weights,
        "weights_normalized": normalized_weights,
        "clip_model": args.clip_model,
        "clip_pretrained": args.clip_pretrained,
        "layer_regex": args.layer_regex,
        "selected_parameters": sorted(protected_names),
        "shared_rank_limit": args.shared_rank,
        "shared_energy": args.shared_energy,
        "alpha_mode": args.alpha_mode,
        "alpha_value": args.alpha_value,
        "alpha_scale": args.alpha_scale,
        "alpha_min": args.alpha_min,
        "alpha_max": args.alpha_max,
        "right_rank_limit": args.right_rank,
        "right_energy": args.right_energy,
        "right_alpha_mode": args.right_alpha_mode,
        "right_alpha_value": args.right_alpha_value,
        "right_alpha_scale": args.right_alpha_scale,
        "right_alpha_min": args.right_alpha_min,
        "right_alpha_max": args.right_alpha_max,
        "eps": args.eps,
        "spectral_balance": args.spectral_balance,
        "spectral_clip_lo": args.spectral_clip_lo,
        "spectral_clip_hi": args.spectral_clip_hi,
        "anchor_index": args.anchor_index,
        "anchor_scale": args.anchor_scale,
        "project_anchor": args.project_anchor,
        "layer_stats": layer_stats,
    }
    if args.anchor_index is not None:
        metadata["anchor_weight_normalized"] = normalized_weights[args.anchor_index]
    if eval_results is not None and eval_datasets is not None and eval_output_path is not None:
        metadata["evaluation"] = {
            "datasets": eval_datasets,
            "results_path": str(eval_output_path),
            "top1": eval_results,
        }
    with (output_dir / "shared_subspace_meta.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if not args.save_state_only:
        template = load_base_model(
            checkpoint=args.base_checkpoint,
            device=torch.device("cpu"),
            dtype=torch.float32,
            clip_arch=args.clip_model,
            clip_pretrained=args.clip_pretrained,
        )
        template.load_state_dict(merged_state, strict=False)
        torch.save(template, output_dir / "shared_subspace_merged.pt")


if __name__ == "__main__":
    main()
