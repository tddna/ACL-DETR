import torch
import torch.nn.functional as F
from torch import nn
from .AnalyticLinear import RecursiveLinear
from .Buffer import RandomBuffer

class ACILClassifierForDETR(nn.Module):
    def __init__(self, input_dim, num_classes, buffer_size=8192, gamma=1e-3, device=None, dtype=torch.double):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.device = device
        self.dtype = dtype

   
        self.buffer = RandomBuffer(input_dim, buffer_size, device=device, dtype=dtype)

        self.analytic_linear = RecursiveLinear(buffer_size, gamma, device=device, dtype=dtype)

        self.eval()

    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.buffer(X)  
        return self.analytic_linear(X)  

    @torch.no_grad()
    def fit(self, X, targets_classes_onehot) :
 
        X = self.buffer(X)  
        
        self.analytic_linear.fit(X, targets_classes_onehot)


    
    @torch.no_grad()
    def update(self) -> None:
        self.analytic_linear.update()


def get_src_permutation_idx_public(indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx