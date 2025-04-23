import torch
import torch.nn.functional as F
from torch import nn
from .AnalyticLinear import RecursiveLinear
from .Buffer import RandomBuffer

class ACILClassifierForDETR(nn.Module):
    def __init__(self, input_dim, num_classes, buffer_size=128, gamma=1e-3, device=None, dtype=torch.double):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.device = device
        self.dtype = dtype

        # 缓存模块：用于特征扩展
        self.buffer = RandomBuffer(input_dim, buffer_size, device=device, dtype=dtype)
        # 解析分类器（RecursiveLinear）
        self.analytic_linear = RecursiveLinear(buffer_size, gamma, device=device, dtype=dtype)
        
        # eval 模式（解析学习不用 BP）
        self.eval()

    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        推理逻辑：用于替换 class_embed 的 forward()。
        输入为 Transformer decoder 输出
        X:input_dim
        buffer(X):buffer_size
        analytic_linear(X):num_classes
        """
        X = self.buffer(X)  # 特征扩展
        return self.analytic_linear(X)  

    @torch.no_grad()
    def fit(self, X, targets_classes_onehot) :
 
        X = self.buffer(X)  # 特征扩展
        
        self.analytic_linear.fit(X, targets_classes_onehot)


    
    # @torch.no_grad()
    # def update(self) -> None:

    #     """
    #     self.analytic_linear.update()


def get_src_permutation_idx_public(indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx