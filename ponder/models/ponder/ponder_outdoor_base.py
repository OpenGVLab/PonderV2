import os
from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from torch_scatter import scatter

from ponder.models.builder import MODELS, build_model
from ponder.models.utils import offset2batch

from .render_utils import RayBundle, build_renderer


@MODELS.register_module("PonderOutdoor-v2")
class PonderOutdoor(nn.Module):
    def __init__(
        self,
        backbone,
        projection,
        renderer,
        mask=None,
        scene_bbox=((-54.0, -54.0, -5.0, 54.0, 54.0, 3.0)),
        grid_shape=((180, 180, 5)),
        grid_size=((0.6, 0.6, 1.6)),
        val_ray_split=8192,
        pool_type="mean",
        share_volume=True,
        render_semantic=False,
        conditions=None,
        template=None,
        clip_model=None,
        class_name=None,
        valid_index=None,
    ):
        super().__init__()
        self.grid_shape = grid_shape
        self.grid_size = grid_size
        self.scene_bbox = scene_bbox
        self.pool_type = pool_type
        self.val_ray_split = val_ray_split
        self.share_volume = share_volume
        self.mask = mask

        if mask is not None:
            p = nn.Parameter(torch.zeros(1, mask.channel))
            trunc_normal_(p, mean=0, std=0.02, a=-0.02, b=0.02)
            self.register_parameter(f"mtoken", p)

        self.backbone = build_model(backbone)
        self.proj_net = build_model(projection)
        self.renderer = build_renderer(renderer)

        self.render_semantic = render_semantic
        self.conditions = conditions
        self.valid_index = valid_index
        if render_semantic:
            self.load_semantic(template, clip_model, class_name)

    def load_semantic(self, template, clip_model, class_name):
        import clip

        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, _ = clip.load(
            clip_model, device=device, download_root="./.cache/clip"
        )
        clip_model.requires_grad_(False)
        if isinstance(template, str):
            class_prompt = [template.replace("[x]", name) for name in class_name]
        elif isinstance(template, Sequence):
            class_prompt = [
                temp.replace("[x]", name) for name in class_name for temp in template
            ]
        class_token = clip.tokenize(class_prompt).to(device)
        class_embedding = clip_model.encode_text(class_token)
        class_embedding = class_embedding / class_embedding.norm(dim=-1, keepdim=True)
        if (not isinstance(template, str)) and isinstance(template, Sequence):
            class_embedding = class_embedding.reshape(
                len(template), len(class_name), clip_model.text_projection.shape[1]
            )
            class_embedding = class_embedding.mean(0)
            class_embedding = class_embedding / class_embedding.norm(
                dim=-1, keepdim=True
            )
        self.register_buffer("class_embedding", class_embedding.float().cpu())

        del clip_model, class_prompt, class_token
        torch.cuda.empty_cache()

    def extract_feature(self, data_dict):
        def random_masking(B, H, W, ratio, device):
            len_keep = round(H * W * (1 - ratio))
            idx = torch.rand(B, H * W).argsort(dim=1)
            idx = idx[:, :len_keep].to(device)  # (B, len_keep)
            # (B, 1, H, W)
            mask = (
                torch.zeros(B, H * W, dtype=torch.bool, device=device)
                .scatter_(dim=1, index=idx, value=True)
                .view(B, 1, H, W)
            )
            return mask

        if self.mask is not None:
            grid_coord, feat, offset = (
                data_dict["grid_coord"],
                data_dict["feat"],
                data_dict["offset"],
            )
            batch_idx = offset2batch(offset)

            block_coord = torch.cat(
                [batch_idx[:, None], torch.div(grid_coord, self.mask.size).int()],
                dim=-1,
            )
            block_coord, inverse_indices = block_coord.unique(
                sorted=False, return_inverse=True, dim=0
            )
            block_mask = []
            for i in range(len(offset)):
                block_mask.append(
                    random_masking(
                        1,
                        (block_coord[:, 0] == i).sum().item(),
                        1,
                        self.mask.ratio,
                        block_coord.device,
                    ).squeeze()
                )
            block_mask = torch.cat(block_mask, dim=0)
            grid_mask = torch.gather(block_mask, 0, inverse_indices)
            feat[~grid_mask] = self.mtoken
            data_dict["feat"] = feat

        data_dict["sparse_backbone_feat"] = self.backbone(data_dict)
        return data_dict

    @torch.no_grad()
    def prepare_ray(self, data_dict):
        def normalize_scene(coord, scene_bbox):
            scene_bbox = torch.tensor(scene_bbox).to(coord)
            norm_coord = (coord - scene_bbox[:3]) / (scene_bbox[3:] - scene_bbox[:3])
            return norm_coord

        condition = data_dict["condition"][0]
        dataset_idx = self.conditions.index(condition)
        scene_bbox = self.scene_bbox[dataset_idx]
        # [0, 1]
        ray_start = normalize_scene(data_dict["ray_start"], scene_bbox)
        ray_end = normalize_scene(data_dict["ray_end"], scene_bbox)
        ray_d = F.normalize(ray_end - ray_start, dim=-1)
        ray_depth = torch.linalg.norm(ray_end - ray_start, dim=-1, keepdim=True)

        ray_dict = {
            "ray_offset": data_dict["ray_offset"],
            "ray_o": ray_start,
            "ray_d": ray_d,
            "depth": ray_depth,
        }
        if "ray_color" in data_dict.keys():
            ray_dict["rgb"] = data_dict["ray_color"]
        if "ray_segment" in data_dict.keys() and self.render_semantic:
            assert (
                len(set(data_dict["condition"])) == 1
            ), "assume same condition in one batch"
            condition = data_dict["condition"][0]
            segmen2semantic = self.class_embedding[
                self.valid_index[self.conditions.index(condition)], :
            ]
            ray_segment = data_dict["ray_segment"]
            ray_semantic = segmen2semantic[ray_segment.long()]
            ray_dict["semantic"] = ray_semantic
            ray_dict["segment"] = ray_segment
        return ray_dict

    def to_dense(self, data_dict):
        coord, feat, offset, condition = (
            data_dict["coord"],
            data_dict["sparse_backbone_feat"],
            data_dict["offset"],
            data_dict["condition"][0],
        )
        assert len(coord) == len(feat)
        batch_idx = offset2batch(offset)
        dataset_idx = self.conditions.index(condition)
        scene_bbox, grid_size, grid_shape = (
            torch.tensor(self.scene_bbox[dataset_idx]).to(coord),
            torch.tensor(self.grid_size[dataset_idx]).to(coord),
            torch.tensor(self.grid_shape[dataset_idx]).to(coord.device),
        )
        coord = ((coord - scene_bbox[:3]) / grid_size).long()
        batch_size = len(offset)
        dense_feat = torch.zeros(
            (batch_size * torch.prod(grid_shape), feat.shape[1])
        ).to(feat)
        index = (
            batch_idx * grid_shape[0] * grid_shape[1] * grid_shape[2]
            + coord[:, 0] * grid_shape[1] * grid_shape[2]
            + coord[:, 1] * grid_shape[2]
            + coord[:, 2]
        )
        dense_feat = scatter(feat, index, dim=0, reduce=self.pool_type, out=dense_feat)
        dense_feat = (
            dense_feat.view(batch_size, *grid_shape, -1)
            .permute(0, 4, 3, 2, 1)
            .contiguous()
        )
        return dense_feat

    def prepare_volume(self, data_dict):
        volume_feat = self.to_dense(data_dict)
        volume_feat = self.proj_net(volume_feat)
        ret_volume_feat = [volume_feat]
        return ret_volume_feat

    def render_func(self, ray_dict, volume_feature):
        ray_offset = ray_dict["ray_offset"]
        batch_idx = offset2batch(ray_offset)
        batched_render_out = []
        for i in range(len(ray_offset)):
            i_ray_o, i_ray_d = (
                ray_dict["ray_o"][i == batch_idx],
                ray_dict["ray_d"][i == batch_idx],
            )
            i_volume_feature = [v[i] for v in volume_feature]

            if self.training:
                ray_bundle = RayBundle(origins=i_ray_o, directions=i_ray_d)
                render_out = self.renderer(ray_bundle, i_volume_feature)
            else:
                render_out = defaultdict(list)
                for j_ray_o, j_ray_d in zip(
                    i_ray_o.split(self.val_ray_split, dim=0),
                    i_ray_d.split(self.val_ray_split, dim=0),
                ):
                    ray_bundle = RayBundle(origins=j_ray_o, directions=j_ray_d)
                    part_render_out = self.renderer(ray_bundle, i_volume_feature)
                    for k, v in part_render_out.items():
                        render_out[k].append(v.detach())
                    del part_render_out
                    torch.cuda.empty_cache()
                for k, v in render_out.items():
                    render_out[k] = torch.cat(v, dim=0)
            batched_render_out.append(render_out)

        render_out = {}
        for k in batched_render_out[0].keys():
            render_out[k] = torch.cat([v[k] for v in batched_render_out], dim=0)
        return render_out

    def render_loss(self, render_out, ray_dict):
        loss_dict = self.renderer.get_loss(render_out, ray_dict)
        loss = sum(_value for _key, _value in loss_dict.items() if "loss" in _key)
        return loss, loss_dict

    def forward(self, data_dict):
        data_dict = self.extract_feature(data_dict)
        ray_dict = self.prepare_ray(data_dict)
        volume_feature = self.prepare_volume(data_dict)
        render_out = self.render_func(ray_dict, volume_feature)
        loss, loss_dict = self.render_loss(render_out, ray_dict)
        out_dict = dict(loss=loss, **loss_dict)
        return out_dict
