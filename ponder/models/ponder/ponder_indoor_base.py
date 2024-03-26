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
from ponder.models.losses import build_criteria
from ponder.models.utils import offset2batch

from .render_utils import RayBundle, build_renderer


@MODELS.register_module("PonderIndoor-v2")
class PonderIndoor(nn.Module):
    def __init__(
        self,
        backbone,
        projection,
        renderer,
        mask=None,
        grid_shape=64,
        grid_size=0.02,
        val_ray_split=10240,
        ray_nsample=128,
        padding=0.1,
        backbone_out_channels=96,
        context_channels=256,
        pool_type="mean",
        render_semantic=False,  # whether to render 2D semantic maps.
        conditions=None,
        template=None,
        clip_model=None,
        class_name=None,
        valid_index=None,
        ppt_loss_weight=1.0,  # whether and how much to use PPT's loss
        ppt_criteria=None,
    ):
        super().__init__()
        self.grid_shape = (
            tuple(grid_shape) if isinstance(grid_shape, Sequence) else (grid_shape,) * 3
        )
        self.grid_size = grid_size
        self.pool_type = pool_type
        self.val_ray_split = val_ray_split
        self.ray_nsample = ray_nsample
        self.mask = mask

        self.bounds = [
            [-0.5 - padding / 2, -0.5 - padding / 2, -0.5 - padding / 2],
            [0.5 + padding / 2, 0.5 + padding / 2, 0.5 + padding / 2],
        ]

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
        self.embedding_table = nn.Embedding(len(conditions), context_channels)
        self.backbone_out_channels = backbone_out_channels
        if render_semantic:
            self.ppt_loss_weight = ppt_loss_weight
            self.load_semantic(template, clip_model, class_name)
        else:
            self.ppt_loss_weight = (
                0.0  # ppt loss is not available when render_semantic is `False`
            )

        if self.ppt_loss_weight > 0:
            assert ppt_criteria is not None, "Please provide PPT's loss function."
            self.ppt_criteria = build_criteria(ppt_criteria)

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
        self.logit_scale = clip_model.logit_scale
        if self.ppt_loss_weight > 0:
            self.proj_head = nn.Linear(
                self.backbone_out_channels, clip_model.text_projection.shape[1]
            )

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

        if "condition" in data_dict:
            condition = data_dict["condition"][0]
            assert condition in self.conditions
            context = self.embedding_table(
                torch.tensor(
                    [self.conditions.index(condition)], device=data_dict["coord"].device
                )
            )
            data_dict["context"] = context

        data_dict["sparse_backbone_feat"] = self.backbone(data_dict)
        return data_dict

    def to_dense(self, data_dict):
        coords = data_dict["coord"]
        sparse_backbone_feat = data_dict["sparse_backbone_feat"]
        offset = data_dict["offset"]
        batch = offset2batch(offset)

        batch_size = batch[-1].tolist() + 1
        c_dim = sparse_backbone_feat.shape[1]

        fea_grid = torch.zeros(
            (batch_size, np.prod(self.grid_shape), c_dim),
            device=sparse_backbone_feat.device,
            dtype=sparse_backbone_feat.dtype,
        )
        # average pooling each tensor to out_resolution
        for i in range(len(offset)):
            coord, feat = (
                coords[offset[i - 1] * int(i != 0) : offset[i]],
                sparse_backbone_feat[offset[i - 1] * int(i != 0) : offset[i]],
            )
            coord = (coord // self.grid_size).int()
            current_resolution = int(data_dict["resolution"][i] + 1)
            if current_resolution >= min(self.grid_shape):  # downsample, pooling
                # define the index of the grid of the current tensor
                grid_index = (
                    coord
                    // (
                        current_resolution
                        / torch.FloatTensor(self.grid_shape).to(coord.device)
                    )
                ).long()
                grid_index = (
                    grid_index[:, 0:1] * self.grid_shape[1] * self.grid_shape[2]
                    + grid_index[:, 1:2] * self.grid_shape[2]
                    + grid_index[:, 2:3]
                )
                # average the features to the grid according to the grid index
                fea_grid[i] = scatter(
                    feat, grid_index, dim=0, reduce=self.pool_type, out=fea_grid[i]
                )
            elif current_resolution <= min(self.grid_shape):  # upsample, resize
                grid_index = (
                    coord[:, 0:1] * current_resolution**2
                    + coord[:, 1:2] * current_resolution
                    + coord[:, 2:3]
                ).long()
                dense_tensor = torch.zeros(
                    (current_resolution**3, c_dim),
                    device=sparse_backbone_feat.device,
                )
                dense_tensor = (
                    scatter(
                        feat, grid_index, dim=0, reduce=self.pool_type, out=dense_tensor
                    )
                    .view(
                        1,
                        current_resolution,
                        current_resolution,
                        current_resolution,
                        c_dim,
                    )
                    .permute(0, 4, 3, 2, 1)
                )
                fea_grid[i] = (
                    F.interpolate(
                        dense_tensor, size=self.grid_shape[::-1], mode="trilinear"
                    )
                    .permute(0, 4, 3, 2, 1)
                    .contiguous()
                    .view(np.prod(self.grid_shape), c_dim)
                )
            else:  # first pooling, then resize
                _out_resolution = (
                    min(current_resolution, self.grid_shape[0]),
                    min(current_resolution, self.grid_shape[1]),
                    min(current_resolution, self.grid_shape[2]),
                )
                grid_index = (
                    coord
                    // (
                        current_resolution
                        / torch.FloatTensor(self.grid_shape).to(coord.device)
                    )
                ).long()
                grid_index = (
                    grid_index[:, 0:1] * _out_resolution[1] * _out_resolution[2]
                    + grid_index[:, 1:2] * _out_resolution[2]
                    + grid_index[:, 2:3]
                )

                _fea_grid = torch.zeros(
                    (np.prod(_out_resolution), c_dim),
                    device=sparse_backbone_feat.device,
                    dtype=sparse_backbone_feat.dtype,
                )
                _fea_grid = scatter(
                    feat, grid_index, dim=0, reduce=self.pool_type, out=_fea_grid
                )

                coord = (
                    torch.FloatTensor(
                        list(
                            zip(
                                *torch.where(
                                    torch.any(
                                        _fea_grid.reshape((*_out_resolution, c_dim)),
                                        dim=-1,
                                    )
                                )
                            )
                        )
                    )
                    .reshape(-1, 3)
                    .to(coord.device)
                )
                feat = _fea_grid[torch.where(torch.any(_fea_grid, dim=-1))].reshape(
                    -1, c_dim
                )
                current_resolution = _out_resolution

                grid_index = (
                    coord[:, 0:1] * current_resolution[1] * current_resolution[2]
                    + coord[:, 1:2] * current_resolution[2]
                    + coord[:, 2:3]
                ).long()
                dense_tensor = torch.zeros(
                    (np.prod(current_resolution), c_dim),
                    device=sparse_backbone_feat.device,
                    dtype=sparse_backbone_feat.dtype,
                )
                dense_tensor = (
                    scatter(
                        feat, grid_index, dim=0, reduce=self.pool_type, out=dense_tensor
                    )
                    .view(
                        1,
                        current_resolution[0],
                        current_resolution[1],
                        current_resolution[2],
                        c_dim,
                    )
                    .permute(0, 4, 3, 2, 1)
                )
                fea_grid[i] = (
                    F.interpolate(
                        dense_tensor,
                        size=self.grid_shape[::-1][::-1],
                        mode="trilinear",
                        align_corners=True,
                    )
                    .permute(0, 4, 3, 2, 1)
                    .contiguous()
                    .view(np.prod(self.grid_shape[::-1]), c_dim)
                )

        return (
            fea_grid.view(
                batch_size,
                self.grid_shape[0],
                self.grid_shape[1],
                self.grid_shape[2],
                c_dim,
            )
            .permute(0, 4, 3, 2, 1)
            .contiguous()
        )

    @torch.no_grad()
    def to_unit_cube(self, data_dict, z_level=-0.5):
        batched_coords = data_dict["coord"].clone()
        batch_offsets = data_dict["offset"]
        batch_size = len(batch_offsets)
        data_dict["pc_scale"] = torch.ones_like(data_dict["depth_scale"])
        data_dict["bbox"] = torch.ones((batch_size, 2, 3), device=batched_coords.device)

        for batch_idx in range(batch_size):
            coords = batched_coords[
                batch_offsets[batch_idx - 1]
                * int(batch_idx != 0) : batch_offsets[batch_idx]
            ]
            bbox = torch.FloatTensor(
                [
                    [
                        coords[:, 0].min() - 1e-5,
                        coords[:, 1].min() - 1e-5,
                        coords[:, 2].min() - 1e-5,
                    ],
                    [
                        coords[:, 0].max() + 1e-5,
                        coords[:, 1].max() + 1e-5,
                        coords[:, 2].max() + 1e-5,
                    ],
                ]
            ).to(coords.device)

            # get center and scale
            loc = (bbox[0] + bbox[1]) / 2  # center
            scale = 1.0 / (bbox[1] - bbox[0]).max()
            tmp_coords = (coords - loc.reshape(-1, 3)) * scale
            z_min = tmp_coords[:, 2].min()
            # create first transalte matrix
            S_loc = torch.eye(4, device=coords.device)
            S_loc[:-1, -1] = -loc
            # create scale mat
            S_scale = torch.eye(4, device=coords.device) * scale
            S_scale[-1, -1] = 1
            # create last translate matrix
            S_loc2 = torch.eye(4, device=coords.device)
            S_loc2[2, -1] = -z_min + z_level
            S = S_loc2 @ S_scale @ S_loc
            # transform points
            stack = torch.column_stack(
                (coords, torch.ones(coords.shape[0], device=coords.device))
            )
            coords = torch.mm(S, stack.T).T[:, :3]
            coords = torch.clip(coords, min=-0.5 + 1e-5, max=0.5 - 1e-5).float()

            # transform cameras parameters, K, R, T
            pose = torch.zeros(
                (data_dict["extrinsic"][batch_idx].shape[0], 4, 4),
                device=coords.device,
            )  # V, 4, 4
            pose[:, :, :] = data_dict["extrinsic"][batch_idx]
            pose[:, 3, 3] = 1
            for view_idx in range(
                (data_dict["extrinsic"][batch_idx].shape[0])
            ):  # for each camera
                cam_pose = torch.mm(
                    pose[view_idx, :, :], torch.linalg.inv(S.float())
                )  # 4, 4
                data_dict["extrinsic"][batch_idx][view_idx, :, :] = cam_pose

            # records scale
            data_dict["depth_scale"][batch_idx] = (
                scale * data_dict["depth_scale"][batch_idx]
            )
            # record PC size
            data_dict["pc_scale"][batch_idx] = (
                bbox[1] - bbox[0]
            ).max()  # data_dict["pc_scale"][batch_idx] *
            # re-compute bbox
            data_dict["bbox"][batch_idx] = torch.FloatTensor(
                [
                    [
                        coords[:, 0].min() - 1e-5,
                        coords[:, 1].min() - 1e-5,
                        coords[:, 2].min() - 1e-5,
                    ],
                    [
                        coords[:, 0].max() + 1e-5,
                        coords[:, 1].max() + 1e-5,
                        coords[:, 2].max() + 1e-5,
                    ],
                ]
            ).to(coords.device)

            # to grid resolution
            coords = (coords + 0.5) * data_dict["pc_scale"][batch_idx]
            data_dict["bbox"][batch_idx] = (
                data_dict["bbox"][batch_idx] + 0.5
            ) * data_dict["pc_scale"][batch_idx]

            data_dict["coord"][
                batch_offsets[batch_idx - 1]
                * int(batch_idx != 0) : batch_offsets[batch_idx]
            ] = coords

        return data_dict

    @torch.no_grad()
    def get_rays(self, H, W, K, R, T):
        K = K.float()
        R = R.float()
        T = T.float()[:, None]
        # pose = torch.cat([R, T], dim=1)
        pose = torch.eye(4, device=R.device).float()
        pose[:3, :4] = torch.cat([R, T], dim=1)
        # pose = torch.cat([pose, torch.tensor([[0, 0, 0, 1]]).to(pose.device)], dim=0)
        pose = torch.linalg.inv(pose)

        l = 1
        tx = torch.linspace(0, W - 1, W // l, device=pose.device)
        ty = torch.linspace(0, H - 1, H // l, device=pose.device)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack(
            [pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1
        )  # W, H, 3
        p = torch.matmul(
            torch.linalg.inv(K)[None, None, :, :], p[:, :, :, None]
        ).squeeze()  # W, H, 3
        rays_v_norm = torch.linalg.norm(
            p, ord=2, dim=-1, keepdim=True
        )  # TODO: used in sdfstudio
        rays_v = p / rays_v_norm  # W, H, 3
        rays_v = torch.matmul(
            pose[None, None, :3, :3], rays_v[:, :, :, None]
        ).squeeze()  # W, H, 3
        rays_o = pose[None, None, :3, 3].expand(rays_v.shape)  # W, H, 3

        # normalize ray directions
        rays_v = F.normalize(rays_v, dim=-1)
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    @torch.no_grad()
    def get_mask_at_box(self, ray_o, ray_d):
        norm_d = np.linalg.norm(ray_d, axis=-1, keepdims=True)
        viewdir = ray_d / norm_d
        viewdir[(viewdir < 1e-5) & (viewdir > -1e-10)] = 1e-5
        viewdir[(viewdir > -1e-5) & (viewdir < 1e-10)] = -1e-5
        inv_dir = 1.0 / viewdir
        tmin = (self.bounds[:1] - ray_o[:1]) * inv_dir
        tmax = (self.bounds[1:2] - ray_o[:1]) * inv_dir

        t1 = np.minimum(tmin, tmax)
        t2 = np.maximum(tmin, tmax)
        near = np.max(t1, axis=-1)
        far = np.min(t2, axis=-1)
        near = np.maximum(near, 0.1)
        mask_at_box = near < far

        return mask_at_box

    @torch.no_grad()
    def ray_sample(self, data_dict):
        batch_colors = data_dict["rgb"].float()  # b, n, h, w, 3
        batch_depths = data_dict["depth"].float()
        batch_intrinsic = data_dict["intrinsic"].float()
        batch_extrinsics = data_dict["extrinsic"].float()
        batch_depth_scale = data_dict["depth_scale"].float()
        masks = (batch_depths > 0).float()
        if self.render_semantic:
            batch_semantics = data_dict["semantic"]

        if "condition" in data_dict:
            condition = data_dict["condition"][0]
            assert condition in self.conditions

        if self.render_semantic and "condition" in data_dict:
            data_dict["index2semantic"] = self.class_embedding[
                self.valid_index[self.conditions.index(condition)], :
            ]

        if self.render_semantic:
            if "index2semantic" in data_dict:
                index2semantic = data_dict["index2semantic"]
            else:
                index2semantic = self.class_embedding

        batch_size = batch_colors.shape[0]
        batch_ret = []
        for b_idx in range(batch_size):
            view_data = []
            for v_idx in range(len(batch_colors[b_idx])):
                img = batch_colors[b_idx, v_idx]
                depth = batch_depths[b_idx, v_idx]
                mask = masks[b_idx, v_idx]
                depth = depth * mask
                if self.render_semantic:
                    semantic = batch_semantics[b_idx, v_idx]
                K = (
                    batch_intrinsic[b_idx][:3, :3]
                    if len(batch_intrinsic[b_idx].shape) == 2
                    else batch_intrinsic[b_idx, v_idx, :3, :3]
                )
                RT = batch_extrinsics[b_idx][v_idx]
                R = RT[:3, :3]
                T = RT[:3, 3]

                # all rays
                ray_o, ray_d = self.get_rays(depth.shape[0], depth.shape[1], K, R, T)

                pixels_y, pixels_x = torch.where(mask > 0)
                idxs = torch.randperm(len(pixels_x))[: self.ray_nsample]
                pixels_x = pixels_x[idxs]
                pixels_y = pixels_y[idxs]

                color = img[pixels_y, pixels_x, :]
                depth = depth[pixels_y, pixels_x] * batch_depth_scale[b_idx]
                ray_o = ray_o[pixels_y, pixels_x, :]
                ray_d = ray_d[pixels_y, pixels_x, :]

                # convert plane-to-plane depth to point-to-point depth
                cam2lidar = torch.linalg.inv(RT)
                plane_dir = (
                    cam2lidar @ torch.FloatTensor([0, 0, 1, 1]).to(RT.device)[:, None]
                )[:3, 0] - ray_o[0]
                plane_dir = plane_dir / torch.linalg.norm(plane_dir)
                depth = depth / torch.linalg.multi_dot((ray_d, plane_dir))

                mask_at_box = torch.from_numpy(
                    self.get_mask_at_box(ray_o.cpu().numpy(), ray_d.cpu().numpy())
                ).to(ray_o.device)
                color[~mask_at_box, :] = 0.0
                depth[~mask_at_box] = -0.001

                data = dict(
                    ray_o=ray_o.float(),
                    ray_d=ray_d.float(),
                    rgb=color.float(),
                    depth=depth.float(),
                    # intrinsic=K.float(),
                    # extrinsic_rotation=R.float(),
                    # extrinsic_translation=T.float(),
                )
                if self.render_semantic:
                    semantic = semantic[pixels_y, pixels_x]
                    semantic[~mask_at_box] = -1
                    assert semantic.max() < index2semantic.shape[0], (
                        semantic.max(),
                        index2semantic.shape,
                        data_dict["condition"][0],
                    )
                    semantic_map = torch.zeros(
                        *semantic.shape,
                        index2semantic.shape[-1],
                        device=semantic.device,
                    )
                    semantic_map[semantic > 0] = index2semantic[
                        semantic[[semantic > 0]].long()
                    ]
                    data.update(dict(semantic=semantic_map.float()))

                view_data.append(data)

            batch_data = dict()
            for k in view_data[0].keys():
                batch_data[k] = torch.stack([v[k] for v in view_data], dim=0)
            batch_ret.append(batch_data)

        ray_dict = dict()
        for k in batch_ret[0].keys():
            ray_dict[k] = torch.stack([v[k] for v in batch_ret], dim=0)

        B, V, N, C = ray_dict["ray_o"].shape
        ray_dict["rgb"] = ray_dict["rgb"].view(-1, ray_dict["rgb"].shape[-1])
        ray_dict["depth"] = ray_dict["depth"].view(-1, 1)
        if self.render_semantic:
            ray_dict["semantic"] = ray_dict["semantic"].view(
                -1, ray_dict["semantic"].shape[-1]
            )
        ray_dict["ray_o"] = ray_dict["ray_o"].view(B, V * N, C)
        ray_dict["ray_d"] = ray_dict["ray_d"].view(B, V * N, C)

        return ray_dict

    def grid_sample(self, data_dict):
        data_dict["bbox"] = (data_dict["bbox"] // self.grid_size).int()
        data_dict["resolution"] = (
            data_dict["bbox"][:, 1] - data_dict["bbox"][:, 0]
        ).max(dim=1)[0].int() + 1
        return data_dict

    @torch.no_grad()
    def prepare_ray(self, data_dict):
        data_dict = self.to_unit_cube(data_dict)
        ray_dict = self.ray_sample(data_dict)
        return ray_dict, data_dict

    def prepare_volume(self, data_dict):
        data_dict = self.grid_sample(data_dict)
        volume_feat = self.to_dense(data_dict)
        volume_feat = self.proj_net(volume_feat)
        ret_volume_feat = [volume_feat]
        return ret_volume_feat

    def render_func(self, ray_dict, volume_feature):
        batch_size = ray_dict["ray_o"].shape[0]
        batched_render_out = []
        for i in range(batch_size):
            i_ray_o, i_ray_d = (
                ray_dict["ray_o"][i],
                ray_dict["ray_d"][i],
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

    def ppt_loss(self, data_dict):
        feat = self.proj_head(data_dict["sparse_backbone_feat"])
        feat = feat / feat.norm(dim=-1, keepdim=True)
        sim = (
            feat
            @ self.class_embedding[
                self.valid_index[self.conditions.index(data_dict["condition"][0])], :
            ].t()
        )
        logit_scale = self.logit_scale.exp()
        seg_logits = logit_scale * sim
        return self.ppt_criteria(seg_logits, data_dict["segment"])

    def forward(self, data_dict):
        data_dict = self.extract_feature(data_dict)
        ray_dict, data_dict = self.prepare_ray(data_dict)
        volume_feature = self.prepare_volume(data_dict)
        render_out = self.render_func(ray_dict, volume_feature)
        loss, loss_dict = self.render_loss(render_out, ray_dict)
        out_dict = dict(loss=loss, **loss_dict)

        if self.ppt_loss_weight > 0:
            ppt_loss = self.ppt_loss(data_dict)
            out_dict["ppt_loss"] = ppt_loss

        return out_dict
