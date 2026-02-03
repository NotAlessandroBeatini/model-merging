import copy
import logging
import re
from typing import Dict, Iterable, List, Sequence

import torch

from model_merging.merger.merger import TaskVectorBasedMerger
from model_merging.merging.structured import (
    aggregate_decomposed_task_vectors,
    get_svd_dict,
)
from model_merging.model.encoder import ImageEncoder
from model_merging.utils.utils import (
    apply_dict_to_model,
    compute_task_dict,
    get_finetuning_accuracies,
)

pylogger = logging.getLogger(__name__)


def _pattern_to_regex(pattern: str) -> re.Pattern:
    escaped = re.escape(pattern).replace(r"\*", ".*")
    return re.compile(rf"^{escaped}$")


def _sort_layer_keys(keys: Iterable[str]) -> List[str]:
    def sort_key(k: str):
        match = re.search(r"resblocks\.(\d+)", k)
        layer_idx = int(match.group(1)) if match else -1
        return (layer_idx, k)

    return sorted(keys, key=sort_key)


def _mode_n_product(tensor: torch.Tensor, matrix: torch.Tensor, mode: int) -> torch.Tensor:
    """
    Multiply tensor by matrix along a given mode.
    tensor shape: (d0, d1, ..., dN)
    matrix shape: (new_dim, d_mode)
    result shape: (d0, ..., d_{mode-1}, new_dim, d_{mode+1}, ...)
    """
    result = torch.tensordot(matrix, tensor, dims=([1], [mode]))
    # Move new_dim (axis 0) to the desired mode position
    order = list(range(1, mode + 1)) + [0] + list(range(mode + 1, result.dim()))
    return result.permute(order)


def _unfold(tensor: torch.Tensor, mode: int) -> torch.Tensor:
    order = [mode] + [i for i in range(tensor.ndim) if i != mode]
    return tensor.permute(order).reshape(tensor.shape[mode], -1)


def _select_rank_by_energy(singular_values: torch.Tensor, threshold: float) -> int:
    if threshold >= 1.0:
        return singular_values.numel()

    energy = singular_values.pow(2)
    total = energy.sum()
    if total <= 0:
        return 1

    cumulative = torch.cumsum(energy, dim=0) / total
    idx = torch.nonzero(cumulative >= threshold, as_tuple=True)[0]
    if idx.numel() == 0:
        return singular_values.numel()
    return int(idx[0].item()) + 1


def _hosvd(
    tensor: torch.Tensor,
    ranks: Sequence[int],
    rank_strategy: str,
    energy_thresholds: Sequence[float],
) -> tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Compute HOSVD for a 4D tensor.
    Returns:
        core: compressed core tensor
        factors: list of factor matrices [U_task, U_layer, U_out, U_in]
    """
    assert tensor.ndim == 4
    dims = tensor.shape

    factors: List[torch.Tensor] = []
    for mode in range(4):
        unfolded = _unfold(tensor, mode)
        u, s, _ = torch.linalg.svd(unfolded, full_matrices=False)
        if rank_strategy == "energy":
            threshold = energy_thresholds[mode]
            r = _select_rank_by_energy(s, threshold)
        else:
            r = min(ranks[mode], dims[mode])
        factors.append(u[:, :r])

    core = tensor
    for mode, u in enumerate(factors):
        core = _mode_n_product(core, u.T, mode)

    return core, factors


class TensorDecompMerger(TaskVectorBasedMerger):
    """
    Cross-layer + cross-task tensor decomposition on selected layer families.
    For non-selected parameters, falls back to a per-layer aggregator (TSV or mean).
    """

    def __init__(
        self,
        layer_families: List[str],
        decomp: str = "hosvd",
        rank_strategy: str = "fixed",
        rank_task: int | None = None,
        rank_layer: int | None = None,
        rank_out: int | None = None,
        rank_in: int | None = None,
        energy_task: float = 0.95,
        energy_layer: float = 0.95,
        energy_out: float = 1.0,
        energy_in: float = 1.0,
        task_weighting: str = "uniform",
        task_weights: List[float] | None = None,
        normalize_task_weights: bool = True,
        fallback: str = "tsv",
        non_matrix_params_aggregation: str = "mean",
        svd_path: str | None = None,
        svd_compress_factor: float | None = None,
        finetuned_accuracy_path: str | None = None,
        model_name: str | None = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.layer_families = layer_families
        self.decomp = decomp
        self.rank_strategy = rank_strategy
        self.rank_task = rank_task
        self.rank_layer = rank_layer
        self.rank_out = rank_out
        self.rank_in = rank_in
        self.energy_task = energy_task
        self.energy_layer = energy_layer
        self.energy_out = energy_out
        self.energy_in = energy_in
        self.task_weighting = task_weighting
        self.task_weights = task_weights
        self.normalize_task_weights = normalize_task_weights
        self.fallback = fallback
        self.non_matrix_params_aggregation = non_matrix_params_aggregation
        self.svd_path = svd_path
        self.svd_compress_factor = svd_compress_factor
        self.finetuned_accuracy_path = finetuned_accuracy_path
        self.model_name = model_name
        self.device = device

    def _resolve_device(self) -> torch.device:
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            pylogger.warning("CUDA requested but not available; falling back to CPU.")
            return torch.device("cpu")
        return torch.device(self.device)

    def _task_names(self, tasks: Iterable) -> List[str]:
        names = []
        for t in tasks:
            if hasattr(t, "name"):
                names.append(t.name)
            else:
                names.append(str(t))
        return names

    def _get_task_weights(self, tasks: List) -> torch.Tensor:
        num_tasks = len(tasks)
        if self.task_weighting == "manual":
            if not self.task_weights:
                raise ValueError("task_weighting='manual' requires task_weights.")
            weights_list = list(self.task_weights)
            if len(weights_list) != num_tasks:
                raise ValueError(
                    f"task_weights length {len(weights_list)} != num_tasks {num_tasks}"
                )
            weights = torch.tensor(weights_list, dtype=torch.float32)
        elif self.task_weighting == "accuracy":
            if not self.finetuned_accuracy_path:
                raise ValueError("task_weighting='accuracy' requires finetuned_accuracy_path.")
            accs = get_finetuning_accuracies(self.finetuned_accuracy_path)
            if self.model_name and self.model_name in accs:
                accs = accs[self.model_name]
            task_names = self._task_names(tasks)
            missing = [n for n in task_names if n not in accs]
            if missing:
                raise ValueError(f"Missing accuracies for tasks: {missing}")
            weights = torch.tensor([accs[n] for n in task_names], dtype=torch.float32)
        else:  # uniform
            weights = torch.ones(num_tasks, dtype=torch.float32)

        if self.normalize_task_weights:
            weights = weights / weights.sum()
        return weights

    def _match_family_keys(self, pattern: str, keys: Iterable[str]) -> List[str]:
        regex = _pattern_to_regex(pattern)
        matched = [k for k in keys if regex.match(k)]
        return _sort_layer_keys(matched)

    def _build_family_tensor(
        self,
        task_dicts: Dict,
        keys: List[str],
        device: torch.device,
    ) -> torch.Tensor:
        tasks = list(task_dicts.keys())
        num_tasks = len(tasks)
        num_layers = len(keys)

        ref = task_dicts[tasks[0]][keys[0]]
        out_dim, in_dim = ref.shape
        tensor = torch.empty(
            num_tasks, num_layers, out_dim, in_dim, device=device, dtype=ref.dtype
        )

        for ti, task in enumerate(tasks):
            for li, key in enumerate(keys):
                tensor[ti, li] = task_dicts[task][key].to(device)

        return tensor

    def _merge_family(
        self,
        family_tensor: torch.Tensor,
        task_weights: torch.Tensor,
        ranks: Sequence[int],
    ) -> torch.Tensor:
        if self.decomp != "hosvd":
            raise ValueError(f"Unsupported decomp method: {self.decomp}")
        if self.rank_strategy not in {"fixed", "energy"}:
            raise ValueError(f"Unsupported rank_strategy: {self.rank_strategy}")

        energy_thresholds = [
            self.energy_task,
            self.energy_layer,
            self.energy_out,
            self.energy_in,
        ]

        core, factors = _hosvd(
            family_tensor,
            ranks=ranks,
            rank_strategy=self.rank_strategy,
            energy_thresholds=energy_thresholds,
        )
        u_task, u_layer, u_out, u_in = factors

        # Collapse task mode using weighted combination in the compressed space
        w = task_weights.to(core.device)
        w_r = u_task.T @ w  # project weights into task subspace
        core_collapsed = torch.tensordot(w_r, core, dims=([0], [0]))

        merged = _mode_n_product(core_collapsed, u_layer, mode=0)
        merged = _mode_n_product(merged, u_out, mode=1)
        merged = _mode_n_product(merged, u_in, mode=2)

        return merged

    def merge(self, base_model: ImageEncoder, finetuned_models: Dict) -> ImageEncoder:
        base_state = base_model.state_dict()
        datasets = list(finetuned_models.keys())

        task_dicts = {}
        for dataset in datasets:
            task_dicts[dataset] = compute_task_dict(
                base_state, finetuned_models[dataset]
            )

        # Fallback merged vector for non-family params
        if self.fallback == "tsv":
            svd_dict = get_svd_dict(
                task_dicts,
                datasets,
                self.svd_path,
                self.svd_compress_factor,
            )
            merged_vector = aggregate_decomposed_task_vectors(
                ref_state_dict=copy.deepcopy(base_state),
                decomposed_task_vectors=svd_dict,
                device=self.device,
                non_matrix_params_aggregation=self.non_matrix_params_aggregation,
            )
        elif self.fallback == "mean":
            merged_vector = {}
            for key in base_state.keys():
                deltas = [task_dicts[d][key] for d in datasets]
                merged_vector[key] = torch.stack(deltas, dim=0).mean(dim=0)
        else:
            raise ValueError(f"Unsupported fallback method: {self.fallback}")

        if not self.layer_families:
            pylogger.warning("No layer families specified; using fallback only.")
            merged_encoder = copy.deepcopy(base_model)
            return apply_dict_to_model(merged_vector, merged_encoder)

        device = self._resolve_device()
        task_weights = self._get_task_weights(datasets)

        for family_pattern in self.layer_families:
            family_keys = self._match_family_keys(family_pattern, base_state.keys())
            if not family_keys:
                raise ValueError(f"No parameters matched pattern: {family_pattern}")

            # Validate shapes
            shape = task_dicts[datasets[0]][family_keys[0]].shape
            if len(shape) != 2:
                raise ValueError(
                    f"Family {family_pattern} is not 2D (shape {shape})."
                )
            for key in family_keys:
                if task_dicts[datasets[0]][key].shape != shape:
                    raise ValueError(
                        f"Inconsistent shapes in family {family_pattern}: "
                        f"{key} has {task_dicts[datasets[0]][key].shape}, expected {shape}"
                    )

            family_tensor = self._build_family_tensor(task_dicts, family_keys, device)

            num_tasks, num_layers, out_dim, in_dim = family_tensor.shape
            ranks = [
                self.rank_task or num_tasks,
                self.rank_layer or num_layers,
                self.rank_out or out_dim,
                self.rank_in or in_dim,
            ]

            merged_family = self._merge_family(
                family_tensor=family_tensor,
                task_weights=task_weights,
                ranks=ranks,
            )

            # Assign back per-layer deltas
            for li, key in enumerate(family_keys):
                merged_vector[key] = merged_family[li].to(merged_vector[key].device)

        merged_encoder = copy.deepcopy(base_model)
        return apply_dict_to_model(merged_vector, merged_encoder)
