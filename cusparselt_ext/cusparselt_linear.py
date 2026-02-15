import os
import torch
from torch import nn
from torch.utils.cpp_extension import load


_ROOT = os.path.dirname(__file__)
# Ensure extension build cache stays in project space, not home
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/fs/nexus-projects/EXACT/.torch_extensions")
_CUSPARSELT_ROOT = os.path.abspath(
    os.path.join(_ROOT, "..", "Samoyeds-Kernel", "cusparselt", "libcusparse_lt-linux-x86_64-0.5.2.1-archive")
)
_INCLUDE = os.path.join(_CUSPARSELT_ROOT, "include")
_LIB = os.path.join(_CUSPARSELT_ROOT, "lib")


def _load_ext():
    return load(
        name="cusparselt_ext",
        sources=[os.path.join(_ROOT, "cusparselt_linear.cu")],
        extra_include_paths=[_INCLUDE],
        extra_cuda_cflags=["-lineinfo"],
        extra_ldflags=[f"-L{_LIB}", "-lcusparseLt"],
        verbose=False,
    )


_ext = None


def cusparselt_spmm(A, B):
    global _ext
    if _ext is None:
        _ext = _load_ext()
    return _ext.cusparselt_spmm(A, B)


class CusparseLtLinear(nn.Module):
    """FP16 linear using cuSPARSELt with 2:4 sparse weights (A) and dense inputs (B)."""
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, in_features], weight: [out_features, in_features]
        if x.numel() == 0 or x.shape[0] == 0:
            return x.new_empty((0, self.weight.shape[0]))
        # cuSPARSELt requires dims aligned (typically multiples of 16)
        batch = x.shape[0]
        pad = (16 - (batch % 16)) % 16
        if pad:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad))
        x_t = x.transpose(0, 1).contiguous()
        y = cusparselt_spmm(self.weight, x_t)  # [out_features, batch+pad]
        y = y.transpose(0, 1).contiguous()
        return y[:batch, :]
