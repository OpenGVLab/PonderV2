from abc import abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

from ..builder import build_collider, build_field, build_sampler
from ..renderers import DepthRenderer, NormalRenderer, RGBRenderer, SemanticRenderer


class SurfaceModel(nn.Module):
    def __init__(self, field, collider, sampler, loss, **kwargs):
        super().__init__()
        self.field = build_field(field)
        self.collider = build_collider(collider)
        self.sampler = build_sampler(sampler)
        self.rgb_renderer = RGBRenderer()
        self.depth_renderer = DepthRenderer()
        self.normal_renderer = NormalRenderer()
        self.semantic_renderer = SemanticRenderer()
        self.loss = loss

    @abstractmethod
    def sample_and_forward_field(self, ray_bundle, volume_feature):
        """_summary_

        Args:
            ray_bundle (RayBundle): _description_
            return_samples (bool, optional): _description_. Defaults to False.
        """

    def get_outputs(self, ray_bundle, volume_feature, **kwargs):
        outputs = {}

        samples_and_field_outputs = self.sample_and_forward_field(
            ray_bundle, volume_feature
        )

        # Shotscuts
        field_outputs = samples_and_field_outputs["field_outputs"]
        ray_samples = samples_and_field_outputs["ray_samples"]
        weights = samples_and_field_outputs["weights"]

        depth = self.depth_renderer(ray_samples=ray_samples, weights=weights)
        normal = self.normal_renderer(normals=field_outputs["normal"], weights=weights)
        if "rgb" in field_outputs.keys():
            rgb = self.rgb_renderer(rgb=field_outputs["rgb"], weights=weights)
            outputs["rgb"] = rgb
        if "semantic" in field_outputs.keys():
            semantic = self.semantic_renderer(
                semantic=field_outputs["semantic"], weights=weights
            )
            outputs["semantic"] = semantic

        outputs.update(
            {
                "depth": depth,
                "normal": normal,
                "weights": weights,
                "sdf": field_outputs["sdf"],
                "gradients": field_outputs["gradients"],
                "z_vals": ray_samples.frustums.starts,
            }
        )

        """ add for visualization"""
        outputs.update({"sampled_points": samples_and_field_outputs["sampled_points"]})
        if samples_and_field_outputs.get("init_sampled_points", None) is not None:
            outputs.update(
                {
                    "init_sampled_points": samples_and_field_outputs[
                        "init_sampled_points"
                    ],
                    "init_weights": samples_and_field_outputs["init_weights"],
                    "new_sampled_points": samples_and_field_outputs[
                        "new_sampled_points"
                    ],
                }
            )

        if self.loss.weights.get("sparse_points_sdf_loss", 0.0) > 0:
            sparse_points_sdf = self.field.get_sdf(
                kwargs["points"].unsqueeze(0), volume_feature
            )[0]
            outputs["sparse_points_sdf"] = sparse_points_sdf.squeeze(0)  # (M, 1)

        return outputs

    def forward(self, ray_bundle, volume_feature, **kwargs):
        """Run forward starting with a ray bundle. This outputs different things depending on the
        configuration of the model and whether or not the batch is provided (whether or not we are
        training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
        """
        ray_bundle = self.collider(ray_bundle)  # set near and far
        return self.get_outputs(ray_bundle, volume_feature, **kwargs)

    def get_loss(self, preds_dict, targets):
        loss_dict = {}
        loss_weights = self.loss.weights

        depth_pred = preds_dict["depth"]  # (num_rays, 1)
        depth_gt = targets["depth"]
        valid_gt_mask = depth_gt > 0.0
        if loss_weights.get("depth_loss", 0.0) > 0:
            depth_loss = torch.sum(
                valid_gt_mask * torch.abs(depth_gt - depth_pred)
            ) / torch.clamp(valid_gt_mask.sum(), min=1.0)
            loss_dict["depth_loss"] = depth_loss * loss_weights.depth_loss

        if loss_weights.get("rgb_loss", 0.0) > 0:
            rgb_pred = preds_dict["rgb"]  # (num_rays, 3)
            rgb_gt = targets["rgb"]
            rgb_loss = F.l1_loss(rgb_pred, rgb_gt)
            loss_dict["rgb_loss"] = rgb_loss * loss_weights.rgb_loss
            psnr = 20.0 * torch.log10(1.0 / (rgb_pred - rgb_gt).pow(2).mean().sqrt())
            loss_dict["psnr"] = psnr

        if loss_weights.get("semantic_loss", 0.0) > 0:
            semantic_pred = preds_dict["semantic"]  # (num_rays, C)
            semantic_gt = targets["semantic"]
            semantic_pred = F.normalize(semantic_pred, dim=-1)
            # semantic_gt = F.normalize(semantic_gt, dim=-1)
            valid_semantic_mask = semantic_gt.any(dim=-1, keepdim=True)  # (num_rays, 1)

            if self.training:
                logits = (
                    torch.mm(semantic_pred, semantic_gt.transpose(1, 0))
                    / self.loss.temperature
                )
                labels = torch.arange(
                    semantic_pred.shape[0],
                    dtype=torch.long,
                    device=semantic_pred.device,
                )
                valid_semantic_loss_mask = valid_gt_mask * valid_semantic_mask
                labels[~valid_semantic_loss_mask.squeeze()] = -100
                if labels[labels != -100].shape[0] == 0:
                    semantic_loss = torch.zeros(1).to(semantic_pred.device)
                else:
                    semantic_loss = F.cross_entropy(logits, labels)
            else:  # in validation, split to ray-sized chunks
                semantic_loss = []
                chunk_size = self.loss.get("val_ray_split", 128)
                for _semantic_pred, _semantic_gt in zip(
                    semantic_pred.split(chunk_size), semantic_gt.split(chunk_size)
                ):
                    logits = (
                        torch.mm(_semantic_pred, _semantic_gt.transpose(1, 0))
                        / self.loss.temperature
                    )
                    labels = torch.arange(
                        _semantic_pred.shape[0],
                        dtype=torch.long,
                        device=semantic_pred.device,
                    )
                    valid_semantic_loss_mask = (
                        valid_gt_mask[chunk_idx : chunk_idx + chunk_size]
                        * valid_semantic_mask[chunk_idx : chunk_idx + chunk_size]
                    )
                    labels[~valid_semantic_loss_mask.squeeze()] = -100
                    if labels[labels != -100].shape[0] == 0:
                        semantic_loss.append(
                            torch.zeros(1).mean().to(semantic_pred.device)
                        )
                    else:
                        semantic_loss.append(F.cross_entropy(logits, labels))
                semantic_loss = torch.mean(torch.stack(semantic_loss))
            loss_dict["semantic_loss"] = semantic_loss * loss_weights.semantic_loss

        # free space loss and sdf loss
        pred_sdf = preds_dict["sdf"][..., 0]
        z_vals = preds_dict["z_vals"][..., 0]
        truncation = self.loss.sensor_depth_truncation

        front_mask = valid_gt_mask & (z_vals < (depth_gt - truncation))
        back_mask = valid_gt_mask & (z_vals > (depth_gt + truncation))
        sdf_mask = valid_gt_mask & (~front_mask) & (~back_mask)

        if loss_weights.get("free_space_loss", 0.0) > 0:
            free_space_loss = (
                F.relu(truncation - pred_sdf) * front_mask
            ).sum() / torch.clamp(front_mask.sum(), min=1.0)
            loss_dict["free_space_loss"] = (
                free_space_loss * loss_weights.free_space_loss
            )

        if loss_weights.get("sdf_loss", 0.0) > 0:
            sdf_loss = (
                torch.abs(z_vals + pred_sdf - depth_gt) * sdf_mask
            ).sum() / torch.clamp(sdf_mask.sum(), min=1.0)
            loss_dict["sdf_loss"] = sdf_loss * loss_weights.sdf_loss

        if loss_weights.get("eikonal_loss", 0.0) > 0:
            gradients = preds_dict["gradients"]
            eikonal_loss = ((gradients.norm(2, dim=-1) - 1) ** 2).mean()
            loss_dict["eikonal_loss"] = eikonal_loss * loss_weights.eikonal_loss

        if loss_weights.get("sparse_points_sdf_loss", 0.0) > 0:
            sparse_points_sdf_loss = torch.mean(
                torch.abs(preds_dict["sparse_points_sdf"])
            )
            loss_dict["sparse_points_sdf_loss"] = (
                sparse_points_sdf_loss * loss_weights.sparse_points_sdf_loss
            )

        return loss_dict
