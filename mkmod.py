import torch as th
from torch.nn import functional as F
from torch import nn

class MLP(nn.Module):
    def __init__(self, in_dims: int, out_dims: int, width: int | None = None, depth: int = 32):
        super().__init__()
        hid_dims = width if width is not None else 3 * in_dims
        self.enc = nn.Linear(in_features=in_dims, out_features=hid_dims)
        self.hid = nn.ModuleList([nn.Linear(in_features=hid_dims, out_features=hid_dims) for _ in range(depth)])
        self.dec = nn.Linear(in_features=hid_dims, out_features=out_dims)
        self.act = nn.Mish()

    def forward(self, X: th.Tensor):
        Z = self.enc(X)
        for linear in self.hid:
            Z = self.act(self.linear(Z)) + Z
        Y = self.dec(Z)
        return Y
    
if __name__ == '__main__':
    pass