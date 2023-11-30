# Smooth sampler

A drop-in replacement for Pytorch's [grid_sample](https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html) supporting smoothstep activation for interpolation weights (proposed in [Instant NGP](https://nvlabs.github.io/instant-ngp/) by Müller et al.) as well as double backpropagation. Currently supports 3D inputs and trilinear interpolation mode. Based on Pytorch's native grid_sampler code. Used in [GO-Surf](https://jingwenwang95.github.io/go_surf/) by Wang et al.

# Installation

On Python3 environment with Pytorch CUDA installation run:

```bash
pip install git+https://github.com/tymoteuszb/smooth-sampler
```

# Usage

The API is consistent with Pytorch's grid_sample:

```python
import torch
from smooth_sampler import SmoothSampler

align_corners = True
padding_mode = "zeros"

input = (torch.rand([2,2,2,3,11], device="cuda")).requires_grad_(True)
grid = (torch.rand([2,2,1,5,3], device="cuda") * 2. - 1.).requires_grad_(True)

out1 = SmoothSampler.apply(input, grid, padding_mode, align_corners, False)
out2 = torch.nn.functional.grid_sample(input, grid, padding_mode=padding_mode, align_corners=align_corners)
assert torch.allclose(out1, out2)

grad1_input, grad1_grid = torch.autograd.grad(out1, [input, grid], torch.ones_like(out1), create_graph=True)
grad2_input, grad2_grid = torch.autograd.grad(out2, [input, grid], torch.ones_like(out2), create_graph=True)
assert torch.allclose(grad1_input, grad2_input)
assert torch.allclose(grad1_grid, grad2_grid)

loss1 = out1.sum() + grad1_input.sum() + grad1_grid.sum()
loss1.backward() # Works!

loss2 = out2.sum() + grad2_input.sum() + grad2_grid.sum()
loss2.backward() # RuntimeException: derivative for aten::grid_sampler_3d_backward is not implemented
```

# Citation

If you use this code in your project, please consider adding a citation:

```
@article{wang2022go-surf,
  title={GO-Surf: Neural Feature Grid Optimization for Fast, High-Fidelity RGB-D Surface Reconstruction},
  author={Wang, Jingwen and Bleja, Tymoteusz and Agapito, Lourdes},
  journal={arXiv preprint},
  year={2022}
}
```
