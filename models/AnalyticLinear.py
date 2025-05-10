# -*- coding: utf-8 -*-
"""
Basic analytic linear modules for the analytic continual learning [1-5].

References:
[1] Zhuang, Huiping, et al.
    "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection."
    Advances in Neural Information Processing Systems 35 (2022): 11602-11614.
[2] Zhuang, Huiping, et al.
    "GKEAL: Gaussian Kernel Embedded Analytic Learning for Few-Shot Class Incremental Task."
    Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.
[3] Zhuang, Huiping, et al.
    "DS-AL: A Dual-Stream Analytic Learning for Exemplar-Free Class-Incremental Learning."
    Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 15. 2024.
[4] Zhuang, Huiping, et al.
    "G-ACIL: Analytic Learning for Exemplar-Free Generalized Class Incremental Learning"
    arXiv preprint arXiv:2403.15706 (2024).
[5] Fang, Di, et al.
    "AIR: Analytic Imbalance Rectifier for Continual Learning."
    arXiv preprint arXiv:2408.10349 (2024).
"""

import torch
from torch.nn import functional as F
from typing import Optional, Union
from abc import abstractmethod, ABCMeta
import numpy as np
import torch.distributed as dist

class AnalyticLinear(torch.nn.Linear, metaclass=ABCMeta):
    def __init__(
        self,
        in_features: int,
        gamma: float = 1e-1,
        bias: bool = False,
        device: Optional[Union[torch.device, str, int]] = None,
        dtype=torch.double,
    ) -> None:
        super(torch.nn.Linear, self).__init__()  # Skip the Linear class
        factory_kwargs = {"device": device, "dtype": dtype}
        self.gamma: float = gamma
        self.bias: bool = bias
        self.dtype = dtype

        # Linear Layer
        if bias:
            in_features += 1
        weight = torch.ones((in_features, 0), **factory_kwargs)
        self.register_buffer("weight", weight)

    @torch.inference_mode()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.weight)
        if self.bias:
            X = torch.cat((X, torch.ones(X.shape[0], 1).to(X)), dim=-1)
        return X @ self.weight

    @property
    def in_features(self) -> int:
        if self.bias:
            return self.weight.shape[0] - 1
        return self.weight.shape[0]

    @property
    def out_features(self) -> int:
        return self.weight.shape[1]

    def reset_parameters(self) -> None:
        # Following the equation (4) of ACIL, self.weight is set to \hat{W}_{FCN}^{-1}
        self.weight = torch.zeros((self.weight.shape[0], 0)).to(self.weight)

    @abstractmethod
    def fit(self, X: torch.Tensor, Y: torch.Tensor, return_updates: bool = False, is_distributed: bool = False) -> None or dict:
        raise NotImplementedError()

    def update(self) -> None:
        assert torch.isfinite(self.weight).all(), (
            "Pay attention to the numerical stability! "
            "A possible solution is to increase the value of gamma. "
            "Setting self.dtype=torch.double also helps."
        )

            

class RecursiveLinear(AnalyticLinear):
    def __init__(
        self,
        in_features: int,
        gamma: float = 1e-1,
        bias: bool = False,
        device: Optional[Union[torch.device, str, int]] = None,
        dtype=torch.double,
    ) -> None:
        super().__init__(in_features, gamma, bias, device, dtype)
        factory_kwargs = {"device": device, "dtype": dtype}

        # Regularized Feature Autocorrelation Matrix (RFAuM)
        self.R: torch.Tensor
        R = torch.eye(self.weight.shape[0], **factory_kwargs) / self.gamma
        self.register_buffer("R", R)
        
    @torch.no_grad()
    def fit(self, X: torch.Tensor, Y: torch.Tensor, return_updates: bool = False, is_distributed: bool = False) -> None or dict:
        """
        拟合线性分类器
        
        参数:
            X: torch.Tensor - 输入特征
            Y: torch.Tensor - 目标标签(one-hot编码)
            return_updates: bool - 是否返回更新值而不是直接应用
            is_distributed: bool - 是否处于分布式环境
        
        返回:
            None 或 包含更新参数的字典
        """
        # 保存当前参数状态
        if return_updates:
            old_weight = self.weight.clone() if hasattr(self, 'weight') else None
            old_bias = self.bias.clone() if hasattr(self, 'bias') else None
            old_R = self.R.clone() if hasattr(self, 'R') else None
        
        X = X.reshape(-1, X.shape[-1])
        Y = Y.reshape(-1, Y.shape[-1])
        X, Y = X.to(self.weight), Y.to(self.weight)
        if self.bias:
            X = torch.cat((X, torch.ones(X.shape[0], 1).to(X)), dim=-1)

        num_targets = Y.shape[1]
        if num_targets > self.out_features:
            increment_size = num_targets - self.out_features
            tail = torch.zeros((self.weight.shape[0], increment_size)).to(self.weight)
            self.weight = torch.cat((self.weight, tail), dim=1)
        elif num_targets < self.out_features:
            increment_size = self.out_features - num_targets
            tail = torch.zeros((Y.shape[0], increment_size)).to(Y)
            Y = torch.cat((Y, tail), dim=1)

        K = torch.inverse(torch.eye(X.shape[0]).to(X) + X @ self.R @ X.T)
        self.R -= self.R @ X.T @ K @ X @ self.R
        self.weight += self.R @ X.T @ (Y - X @ self.weight)

        if dist.is_initialized():
            torch.cuda.synchronize()
            
            dist.all_reduce(self.weight, op=dist.ReduceOp.SUM)
            self.weight /= dist.get_world_size()

            dist.all_reduce(self.R, op=dist.ReduceOp.SUM)
            self.R /= dist.get_world_size()
            
            torch.cuda.synchronize()
            dist.barrier()

        # 如果需要返回更新而非应用
        if return_updates:
            updates = {}
            if hasattr(self, 'weight'):
                updates['weight'] = self.weight.clone()
                # 恢复原始参数
                self.weight.copy_(old_weight)
            
            if hasattr(self, 'bias'):
                updates['bias'] = self.bias.clone()
                # 恢复原始参数  
                self.bias.copy_(old_bias)
                
            if hasattr(self, 'R'):
                updates['R'] = self.R.clone()
                # 恢复原始参数
                self.R.copy_(old_R)
            
            return updates


class GeneralizedARM(AnalyticLinear):
    """Analytic Re-weighting Module (ARM) for generalized CIL."""

    def __init__(
        self,
        in_features: int,
        gamma: float = 1e-1,
        bias: bool = False,
        device: Optional[Union[torch.device, str, int]] = None,
        dtype=torch.double,
    ) -> None:
        super().__init__(in_features, gamma, bias, device, dtype)
        factory_kwargs = {"device": device, "dtype": dtype}

        weight = torch.zeros((in_features, 0), **factory_kwargs)
        self.register_buffer("weight", weight)

        A = torch.zeros((0, in_features, in_features), **factory_kwargs)
        self.register_buffer("A", A)

        C = torch.zeros((in_features, 0), **factory_kwargs)
        self.register_buffer("C", C)

        self.cnt = torch.zeros(0, dtype=torch.int, device=device)

    @property
    def out_features(self) -> int:
        return self.C.shape[1]

    @torch.inference_mode()
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        X = X.to(self.weight)
        # Bias
        if self.bias:
            X = torch.concat((X, torch.ones(X.shape[0], 1)), dim=-1)

        # GCIL
        num_targets = int(y.max()) + 1
        if num_targets > self.out_features:
            increment_size = num_targets - self.out_features
            torch.cuda.empty_cache()
            # Increment C
            tail = torch.zeros((self.C.shape[0], increment_size)).to(self.weight)
            self.C = torch.concat((self.C, tail), dim=1)
            # Increment cnt
            tail = torch.zeros((increment_size,)).to(self.cnt)
            self.cnt = torch.concat((self.cnt, tail))
            # Increment A
            tail = torch.zeros((increment_size, self.in_features, self.in_features))
            self.A = torch.concat((self.A, tail.to(self.A)))
            torch.cuda.empty_cache()
        else:
            num_targets = self.out_features

        # ACIL
        Y = F.one_hot(y, max(num_targets, num_targets)).to(self.C)
        self.C += X.T @ Y

        # Label Balancing
        y_labels, label_cnt = torch.unique(y, sorted=True, return_counts=True)
        y_labels, label_cnt = y_labels.to(self.cnt.device), label_cnt.to(
            self.cnt.device
        )
        self.cnt[y_labels] += label_cnt

        # Accumulate
        for i in range(num_targets):
            X_i = X[y == i]
            self.A[i] += X_i.T @ X_i

    @torch.inference_mode()
    def update(self):
        cnt_inv = 1 / self.cnt.to(self.dtype)
        cnt_inv[torch.isinf(cnt_inv)] = 0  # replace inf with 0
        cnt_inv *= len(self.cnt) / cnt_inv.sum()

        weighted_A = torch.sum(cnt_inv[:, None, None].mul(self.A), dim=0)
        A = weighted_A + self.gamma * torch.eye(self.in_features).to(self.A)
        C = self.C.mul(cnt_inv[None, :])

        self.weight = torch.inverse(A) @ C
