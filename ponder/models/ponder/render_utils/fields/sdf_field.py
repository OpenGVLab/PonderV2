import torch
import torch.nn.functional as F
from smooth_sampler import SmoothSampler
from torch import nn

from ..builder import FIELDS
from ..decoders import RGBDecoder, SDFDecoder, SemanticDecoder


class LaplaceDensity(nn.Module):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    """Laplace density from VolSDF"""

    def __init__(self, init_val, beta_min=0.0001):
        super().__init__()
        self.register_parameter(
            "beta_min", nn.Parameter(beta_min * torch.ones(1), requires_grad=False)
        )
        self.register_parameter(
            "beta", nn.Parameter(init_val * torch.ones(1), requires_grad=True)
        )

    def forward(self, sdf, beta=None):
        """convert sdf value to density value with beta, if beta is missing, then use learable beta"""
        if beta is None:
            beta = self.get_beta()

        alpha = 1.0 / beta

        density = alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))
        return density

    def get_beta(self):
        """return current beta value"""
        beta = self.beta.abs() + self.beta_min
        return beta


class SingleVarianceNetwork(nn.Module):
    """Variance network in NeuS"""

    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter(
            "variance", nn.Parameter(init_val * torch.ones(1), requires_grad=True)
        )

    def forward(self, x):
        """Returns current variance value"""
        return torch.ones([len(x), 1], device=x.device) * torch.exp(
            self.variance * 10.0
        )

    def get_variance(self):
        """return current variance value"""
        return torch.exp(self.variance * 10.0).clip(1e-6, 1e6)


@FIELDS.register_module()
class SDFField(nn.Module):
    def __init__(
        self,
        sdf_decoder,
        beta_init,
        use_gradient=True,
        volume_type="default",
        padding_mode="zeros",
        share_volume=True,
        rgb_decoder=None,
        semantic_decoder=None,
    ):
        super().__init__()
        self.beta_init = beta_init
        self.volume_type = volume_type
        self.padding_mode = padding_mode
        self.share_volume = share_volume
        self.sdf_decoder = SDFDecoder(**sdf_decoder)
        if rgb_decoder is not None:
            self.rgb_decoder = RGBDecoder(**rgb_decoder)
        else:
            self.rgb_decoder = None
        if semantic_decoder is not None:
            self.semantic_decoder = SemanticDecoder(**semantic_decoder)
        else:
            self.semantic_decoder = None
        self.use_gradient = use_gradient
        # laplace function for transform sdf to density from VolSDF
        self.laplace_density = LaplaceDensity(init_val=self.beta_init)

        # deviation_network to compute alpha from sdf from NeuS
        self.deviation_network = SingleVarianceNetwork(init_val=self.beta_init)

        self._cos_anneal_ratio = 1.0

    def set_cos_anneal_ratio(self, anneal):
        """Set the anneal value for the proposal network."""
        self._cos_anneal_ratio = anneal

    def get_alpha(self, ray_samples, sdf, gradients):
        inv_s = self.deviation_network.get_variance()  # Single parameter

        true_cos = (ray_samples.frustums.directions * gradients).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(
            F.relu(-true_cos * 0.5 + 0.5) * (1.0 - self._cos_anneal_ratio)
            + F.relu(-true_cos) * self._cos_anneal_ratio
        )  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * ray_samples.deltas * 0.5
        estimated_prev_sdf = sdf - iter_cos * ray_samples.deltas * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)

        return alpha

    def feature_sampling(self, pts_norm, volume_feature):
        """
        Args:
            pts: (N, K, 3), [x, y, z], scaled
            feats_volume: (C, Z, Y, X)
        Returns:
            feats: (N, K, C)
        """
        pts_norm = pts_norm * 2 - 1  # [0, 1] -> [-1, 1]
        if self.volume_type == "default":
            ret_feat = []
            for fea_grid in volume_feature:
                ret_feat.append(
                    SmoothSampler.apply(
                        fea_grid.unsqueeze(0).to(pts_norm.dtype),
                        pts_norm[None, None],
                        self.padding_mode,
                        True,
                        False,
                    )
                    .squeeze(0)
                    .squeeze(1)
                    .permute(1, 2, 0)
                )  # (1, C, 1, N, K) -> (N, K, C)
            ret_feat = torch.stack(ret_feat, dim=-2)  # (N, K, L, C)
            half_ch = ret_feat.shape[-1] // 2
            ret_feat = torch.cat(
                [
                    ret_feat[..., :half_ch].flatten(-2, -1),
                    ret_feat[..., half_ch:].flatten(-2, -1),
                ],
                dim=-1,
            )  # (N, K, L*C1+L*C2)
        else:
            raise NotImplementedError
        return ret_feat

    def get_sdf(self, points, volume_feature):
        """predict the sdf value for ray samples"""
        point_features = self.feature_sampling(points, volume_feature)
        h = self.sdf_decoder(
            points,
            point_features
            if self.share_volume
            else torch.chunk(point_features, 2, dim=-1)[0],
        )
        sdf, geo_features = h[..., :1], h[..., 1:]
        return sdf, geo_features, point_features

    def get_density(self, ray_samples, volume_feature):
        """Computes and returns the densities."""
        points = ray_samples.frustums.get_start_positions()
        sdf = self.get_sdf(points, volume_feature)[0]
        density = self.laplace_density(sdf)
        return density

    def get_occupancy(self, sdf):
        """compute occupancy as in UniSurf"""
        occupancy = torch.sigmoid(-10.0 * sdf)
        return occupancy

    def forward(self, ray_samples, volume_feature, return_alphas=False):
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        outputs = {}
        rgb_inputs = []

        points = (
            ray_samples.frustums.get_start_positions()
        )  # (num_rays, num_samples, 3)

        points.requires_grad_(True)
        with torch.enable_grad():
            sdf, geo_features, point_features = self.get_sdf(points, volume_feature)

        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=points,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        if self.use_gradient:
            rgb_inputs.append(gradients)

        directions = ray_samples.frustums.directions  # (num_rays, num_samples, 3)
        rgb_inputs.extend(
            [
                point_features
                if self.share_volume
                else torch.chunk(point_features, 2, dim=-1)[1],
                geo_features,
                directions,
            ]
        )
        if self.rgb_decoder is not None:
            rgb = self.rgb_decoder(points, torch.cat(rgb_inputs, dim=-1))
            outputs["rgb"] = rgb

        if self.semantic_decoder is not None:
            semantic = self.semantic_decoder(
                points,
                torch.cat(rgb_inputs[:-1], dim=-1),
            )
            outputs["semantic"] = semantic

        density = self.laplace_density(sdf)

        outputs.update(
            {
                "density": density,
                "sdf": sdf,
                "gradients": gradients,
                "normal": F.normalize(gradients, dim=-1),  # TODO: should normalize?
            }
        )

        if return_alphas:
            # TODO use mid point sdf for NeuS
            alphas = self.get_alpha(
                ray_samples, sdf, gradients
            )  # (num_rays, num_samples, 1)
            outputs.update({"alphas": alphas})

        return outputs
