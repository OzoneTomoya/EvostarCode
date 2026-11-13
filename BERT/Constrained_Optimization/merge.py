import argparse
import datetime
import os

# Error: mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL is incompatible with libgomp-a34b3233.so.1 library.  # noqa: E501
#     Try to import numpy first or set the threading layer accordingly. Set MKL_SERVICE_FORCE_INTEL to force it.  # noqa: E501
import numpy  # noqa: F401
import torch
import torch.nn as nn
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments

@torch.no_grad()
def dare_ties_marge(
    base_model: nn.Module,
    models: list[nn.Module],
    weights: list[float] | None = None,
    densities: list[float] | None = None,
    scale: float | None = None,
    seed: int | None = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> None:
    if weights is None:
        weights = [1.0] * len(models)
    if densities is None:
        densities = [0.5] * len(models)
    scale = scale or 1.0
    seed = seed or 0
    weights = torch.tensor(weights, device=device, dtype=torch.float32)
    densities = torch.tensor(densities, device=device, dtype=torch.float32)
    assert len(models) == len(weights), f"{len(models)=}, {len(weights)=}"
    assert len(models) == len(densities), f"{len(models)=}, {len(densities)=}"

    base_state_dict = base_model.state_dict()
    model_state_dicts = [m.state_dict() for m in models]

    def align_dim(tensor: torch.Tensor, total_dim: int):
        return tensor.reshape(tensor.shape + (1,) * (total_dim - tensor.dim()))

    def sparsify(params: torch.Tensor) -> torch.Tensor:
        torch.manual_seed(seed)
        aligned_densities = align_dim(densities, params.dim())
        mask = torch.rand_like(params) < aligned_densities
        return params * mask / aligned_densities

    def elect_sign(params: torch.Tensor) -> torch.Tensor:
        return params.sign() == params.sum(dim=0).sign()

    def merge_params(key):
        base_param = base_state_dict[key].to(device, non_blocking=True)
        model_params = torch.stack(
            [s[key].to(device, non_blocking=True) for s in model_state_dicts]
        )

        task_vectors = (model_params - base_param) * align_dim(
            weights, model_params.dim()
        )

        # Trim with DARE
        task_vectors = sparsify(task_vectors)
        #task_vectors = weights * task_vectors

        #############
        weights_ = weights.to(task_vectors.device).to(task_vectors.dtype)

        while weights_.dim() < task_vectors.dim():
            weights_ = weights_.unsqueeze(-1)
        task_vectors = weights_ * task_vectors
        ##################

        # Elect
        mask = elect_sign(task_vectors)

        # Merge
        merged_vector = (task_vectors * mask).sum(dim=0) 

        base_param += scale * merged_vector
        return base_param.to("cpu")

    keys = base_state_dict.keys()
    for state_dict in model_state_dicts:
        keys |= state_dict.keys()

    for key in keys:
        #########
        if "classifier" in key or "score" in key or "head" in key or "pooler" in key:
            continue
        #########
        base_state_dict[key] = merge_params(key)

    base_model.load_state_dict(base_state_dict)


class ApiWrapper:
    def __init__(self):
        self.api = HfApi(token=os.environ["HF_AUTH_TOKEN"])

    def newest_commit(self, model_id) -> str | None:
        cutoff = datetime.datetime(
            year=2024, month=6, day=1, tzinfo=datetime.timezone.utc
        )
        for commit in self.api.list_repo_commits(model_id):
            if commit.created_at < cutoff:
                return commit.commit_id


class DareTies:

    def __init__(self):
        self.api = HfApi() 

    def _load_model(self, cache_dir) -> nn.Module:
        try:
            model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, "bert-base-uncased"))
        except:
            model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path="bert-base-uncased", cache_dir=cache_dir)
        return model
    
    def _load_local_model(self, model_id: str) -> nn.Module:
        return AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_id)

    def merge_and_save(
        self,
        base_model_name: str,
        model_names: list[str],
        x: list[float],
        cache_dir: str,
        save_dir: str,
        seed: int,
    ):
        base_model = self._load_model(cache_dir)
        model_paths = [os.path.join("../FT_Model", name) for name in model_names]
        models = [self._load_local_model(p) for p in model_paths]

        dare_ties_marge(
            base_model,
            models,
            weights=x[: len(models)],
            densities=x[len(models) : -1],
            scale=x[-1],
            seed=seed,
        )

        base_model.save_pretrained(save_dir)
        token = os.environ.get("HF_AUTH_TOKEN", None)
        AutoTokenizer.from_pretrained(
            base_model_name, token=token
        ).save_pretrained(save_dir)
