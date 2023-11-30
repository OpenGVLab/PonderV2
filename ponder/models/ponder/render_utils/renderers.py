import torch
from torch import nn


class RGBRenderer(nn.Module):
    """Standard volumetic rendering."""

    def __init__(self, background_color=(0.0, 0.0, 0.0)):
        super().__init__()
        self.background_color = background_color

    def forward(self, rgb, weights):
        """Composite samples along ray and render color image

        Args:
            rgb: RGB for each sample, (num_rays, num_samples, 3)
            weights: Weights for each sample, (num_rays, num_samples, 1)
        Returns:
            Outputs of rgb values.
        """
        comp_rgb = torch.sum(weights * rgb, dim=-2)  # (num_rays, 3)
        accumulated_weight = torch.sum(weights, dim=-2)
        comp_rgb = comp_rgb + comp_rgb.new_tensor(self.background_color) * (
            1.0 - accumulated_weight
        )
        if not self.training:
            torch.clamp_(comp_rgb, min=0.0, max=1.0)
        return comp_rgb


class DepthRenderer(nn.Module):
    """Calculate depth along ray."""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, ray_samples, weights):
        """Composite samples along ray and calculate depths.

        Args:
            weights: Weights for each sample.
            ray_samples: Set of ray samples.
        Returns:
            Outputs of depth values.
        """
        eps = 1e-10
        # steps = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2
        steps = ray_samples.frustums.starts
        depth = torch.sum(weights * steps, dim=-2) / (torch.sum(weights, -2) + eps)
        depth = torch.clip(depth, steps.min(), steps.max())
        return depth


class NormalRenderer(nn.Module):
    """Calculate normals along the ray."""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, normals, weights):
        """Calculate normals along the ray."""
        n = torch.sum(weights * normals, dim=-2)
        return n


class SemanticRenderer(nn.Module):
    """Calculate semantic features along the ray."""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, semantic, weights):
        """Calculate semantic features along the ray."""
        f = torch.sum(weights * semantic, dim=-2)
        return f
