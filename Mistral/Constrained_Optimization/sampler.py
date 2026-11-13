from __future__ import annotations

import copy
from typing import Any

from vllm import LLM
from mergekit.config import MergeConfiguration
from optuna.distributions import BaseDistribution
from optuna.distributions import FloatDistribution, IntDistribution
from optuna._transform import _SearchSpaceTransform
from optuna.samplers import BaseSampler
from optuna.samplers import CmaEsSampler
from optuna.study import Study
from optuna.trial import FrozenTrial
import yaml
import numpy as np
from scipy.interpolate import interp1d, CubicSpline, lagrange

from trial import EvoMergeTrial

class EvoMergeSampler(BaseSampler):
    def __init__(
        self,
        optimization: str,
        base_layer_bits: dict | None = None,
        base_config: dict | None = None,
        seed: int | None = None,
    ) -> None:
        self.optimization = optimization
        self.seed = seed
        x0 = {}

        if optimization == "Sequential_MergeParameter":
            if base_config is None:
                raise ValueError("base_config is required for Sequential_MergeParameter")
            
            with open(base_config, "r", encoding="utf-8") as fp:
                self._merge_config = MergeConfiguration.model_validate(yaml.safe_load(fp))
                for model_cfg in self._merge_config.models:
                    model_name = model_cfg.model  
                    params = getattr(model_cfg, "parameters", None)
                    if params:
                        for k, v in params.items():
                            x0[f"{model_name}-{k}"] = v
            
            self._layer_bits = base_layer_bits
        elif optimization == "Sequential_BitWidth":
            self._merge_config = None

            if base_layer_bits is None:
                raise ValueError("base_layer_bits is required for Sequential_BitWidth")

            self._layer_bits = base_layer_bits
            x0 = {f"layer{i}-bit": self._layer_bits.get(i) for i in range(32)}
        else:
            if base_layer_bits is None or base_config is None:
                raise ValueError("Both base_layer_bits and base_config are required")

            with open(base_config, "r", encoding="utf-8") as fp:
                self._merge_config = MergeConfiguration.model_validate(yaml.safe_load(fp))
                for model_cfg in self._merge_config.models:
                    model_name = model_cfg.model  
                    params = getattr(model_cfg, "parameters", None)
                    if params:
                        for k, v in params.items():
                            x0[f"{model_name}-{k}"] = v
            
            self._layer_bits = base_layer_bits
            x0.update({f"layer{i}-bit": self._layer_bits.get(i) for i in range(32)})

        sigma0 = 1/6
        self._cmaes = CmaEsSampler(x0=x0, sigma0=sigma0, seed=seed, with_margin=True)
        self.search_space = None
        self.models = set()
        self.parameters = set()

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        return {}

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        return {}

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        return param_distribution._asdict()

    def reseed_rng(self, seed: int) -> None:
        self.seed = seed

    def _sample(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        return param_distribution._asdict()

    def sample_model(self, study: Study, trial: EvoMergeTrial) -> LLM:
        merge_config = copy.deepcopy(self._merge_config)
        layer_bits = copy.deepcopy(self._layer_bits)
        quamerge_search_space = {}
        model_name_to_params = {}

        if self.optimization == "Sequential_MergeParameter":
            for i, m in enumerate(merge_config.models):
                if i == 0:
                    # No parameters necessary for base model.
                    continue
                model_name = m.model
                search_space = {f"{model_name}-{k}": FloatDistribution(0, 1) for k in m.parameters}
                quamerge_search_space.update(search_space)
                model_name_to_params[model_name] = list(search_space.keys())

            if quamerge_search_space:
                params = self._cmaes.sample_relative(study, trial, quamerge_search_space)
            else:
                params = {}

            if self.search_space == None:
                self.search_space = quamerge_search_space

            for i, m in enumerate(merge_config.models):
                if i == 0:
                    continue
                model_name = m.model

                if len(params) == 0:
                    # trial = 0
                    for param in m.parameters:
                        param_value = m.parameters[param]
                        distribution = FloatDistribution(0, 1)
                        param_value_in_internal_repr = distribution.to_internal_repr(param_value)
                        study._storage.set_trial_param(
                            trial._trial_id, f"{model_name}-{param}", param_value_in_internal_repr, distribution
                        )
                else:
                    # trial > 0
                    for k in m.parameters:
                        param_name = f"{model_name}-{k}"
                        if param_name in params:
                            param_value = params[param_name]
                            distribution = quamerge_search_space[param_name]
                            param_value_in_internal_repr = distribution.to_internal_repr(param_value)
                            study._storage.set_trial_param(
                                trial._trial_id, param_name, param_value_in_internal_repr, distribution
                            )
                            m.parameters[k] = param_value
            return merge_config
        elif self.optimization == "Sequential_BitWidth":
            for i in range(32):
                search_space = {f"layer{i}-bit": IntDistribution(5, 16)}
                quamerge_search_space.update(search_space)

            if quamerge_search_space:
                params = self._cmaes.sample_relative(study, trial, quamerge_search_space)
            else:
                params = {}

            if self.search_space == None:
                self.search_space = quamerge_search_space

            for i in range(32):
                if len(params) == 0:
                    # trial = 0
                    param_value = layer_bits.get(i, 8)
                    distribution = IntDistribution(5, 16)
                    param_value_in_internal_repr = distribution.to_internal_repr(param_value)
                    study._storage.set_trial_param(
                        trial._trial_id, f"layer{i}-bit", param_value_in_internal_repr, distribution
                    )
                else:
                    # trial > 0
                    param_name = f"layer{i}-bit"
                    param_value = round(params[param_name])
                    distribution = quamerge_search_space[param_name]
                    param_value_in_internal_repr = distribution.to_internal_repr(param_value)
                    study._storage.set_trial_param(
                        trial._trial_id, param_name, param_value_in_internal_repr, distribution
                    )
                    layer_bits[i] = param_value

            return layer_bits
        else:
            for i, m in enumerate(merge_config.models):
                if i == 0:
                    # No parameters necessary for base model.
                    continue
                model_name = m.model
                search_space = {f"{model_name}-{k}": FloatDistribution(0, 1) for k in m.parameters}
                quamerge_search_space.update(search_space)
                model_name_to_params[model_name] = list(search_space.keys())
            
            for i in range(32):
                search_space = {f"layer{i}-bit": IntDistribution(5, 16)}
                quamerge_search_space.update(search_space)

            if quamerge_search_space:
                params = self._cmaes.sample_relative(study, trial, quamerge_search_space)
            else:
                params = {}

            if self.search_space == None:
                self.search_space = quamerge_search_space

            for i, m in enumerate(merge_config.models):
                if i == 0:
                    continue
                model_name = m.model

                if len(params) == 0:
                    # trial = 0
                    for param in m.parameters:
                        param_value = m.parameters[param]
                        distribution = FloatDistribution(0, 1)
                        param_value_in_internal_repr = distribution.to_internal_repr(param_value)
                        study._storage.set_trial_param(
                            trial._trial_id, f"{model_name}-{param}", param_value_in_internal_repr, distribution
                        )
                else:
                    # trial > 0
                    for k in m.parameters:
                        param_name = f"{model_name}-{k}"
                        if param_name in params:
                            param_value = params[param_name]
                            distribution = quamerge_search_space[param_name]
                            param_value_in_internal_repr = distribution.to_internal_repr(param_value)
                            study._storage.set_trial_param(
                                trial._trial_id, param_name, param_value_in_internal_repr, distribution
                            )
                            m.parameters[k] = param_value
            
            for i in range(32):
                if len(params) == 0:
                    # trial = 0
                    param_value = layer_bits.get(i, 8)
                    distribution = IntDistribution(5, 16)
                    param_value_in_internal_repr = distribution.to_internal_repr(param_value)
                    study._storage.set_trial_param(
                        trial._trial_id, f"layer{i}-bit", param_value_in_internal_repr, distribution
                    )
                else:
                    # trial > 0
                    param_name = f"layer{i}-bit"
                    param_value = round(params[param_name])
                    distribution = quamerge_search_space[param_name]
                    param_value_in_internal_repr = distribution.to_internal_repr(param_value)
                    study._storage.set_trial_param(
                        trial._trial_id, param_name, param_value_in_internal_repr, distribution
                    )
                    layer_bits[i] = param_value

            return merge_config, layer_bits
