# Copyright (c) SSDAL-Net Authors. All rights reserved.
"""
Deformable Bi-level Routing Attention FPN (DBRA-FPN)

Paper: Adaptive Sparse-Deformable Synergistic Mechanism for Object Detection
       in Complex Underwater Scenes
Venue: The Visual Computer

Key Innovation:
    - Level-1 Routing (Region-level Semantic Routing):
      Splits feature map into GxG regions, computes semantic similarity
      matrix R to guide region-level attention.
    - Level-2 Sampling (Offset-driven Deformable Convolution):
      Uses R to predict sampling offsets for K deformable sample points,
      achieving precise geometric alignment for non-rigid underwater objects.
    - Alleviates the "offset drift" problem of standard deformable convolutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule

from ..builder import NECKS


class DBRA(BaseModule):
    """
    Deformable Bi-level Routing Attention Module.

    Architecture:
        Level 1 (Region Routing): Partition feature map into GxG regions,
        compute region semantic vectors, then region similarity matrix R.
        Level 2 (Deformable Sampling): Use R to predict K per-point offsets,
        sample with deformable convolution for geometric alignment.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_groups (int): Number of region groups G for Level-1 routing (G x G regions). Default: 4.
        num_sampling_points (int): Number K of deformable sampling points. Default: 9.
        mlp_ratio (float): Expansion ratio of FFN. Default: 4.0.
        temperature (float): Temperature for region routing softmax. Default: 0.07.
        init_cfg: Initialization config.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_groups=4,
        num_sampling_points=9,
        mlp_ratio=4.0,
        temperature=0.07,
        init_cfg=None,
    ):
        super(DBRA, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_groups = num_groups
        self.num_sampling_points = num_sampling_points
        self.temperature = temperature

        # Input projection
        self.input_conv = ConvModule(
            in_channels, out_channels, 1,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'))

        # ---- Level 1: Region-level Semantic Routing ----
        self.q_conv = nn.Conv2d(out_channels, out_channels, 1)
        self.k_conv = nn.Conv2d(out_channels, out_channels, 1)
        self.v_conv = nn.Conv2d(out_channels, out_channels, 1)

        # FFN after region attention
        self.region_ffn = nn.Sequential(
            nn.Conv2d(out_channels, int(out_channels * mlp_ratio), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(out_channels * mlp_ratio), out_channels, 1),
        )
        self.region_norm = nn.GroupNorm(num_groups=32, num_channels=out_channels)

        # ---- Level 2: Offset Prediction (guided by region routing) ----
        self.offset_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * 2, num_sampling_points * 2, 1),
        )

        # Deformable sampling weight
        self.sampling_weight_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.Sigmoid(),
        )

        # Output fusion
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C_in, H, W = x.size()
        x = self.input_conv(x)

        # Level 1: Region-level Semantic Routing
        Q = self.q_conv(x)
        K = self.k_conv(x)
        V = self.v_conv(x)

        # Pool to GxG region representations
        Q_region = F.adaptive_avg_pool2d(Q, (self.num_groups, self.num_groups))
        K_region = F.adaptive_avg_pool2d(K, (self.num_groups, self.num_groups))
        V_region = F.adaptive_avg_pool2d(V, (self.num_groups, self.num_groups))

        # Region similarity matrix R
        Q_r = Q_region.view(B, self.in_channels, -1).permute(0, 2, 1)  # (B, G^2, C)
        K_r = K_region.view(B, self.in_channels, -1)  # (B, C, G^2)
        region_sim = torch.bmm(Q_r, K_r)  # (B, G^2, G^2)
        region_sim = region_sim / (self.in_channels ** 0.5 * self.temperature)
        region_attn = self.softmax(region_sim)  # (B, G^2, G^2)

        V_r = V_region.view(B, self.in_channels, -1).permute(0, 2, 1)  # (B, G^2, C)
        region_out = torch.bmm(region_attn, V_r)  # (B, G^2, C)
        region_out = region_out.permute(0, 2, 1).view(B, self.in_channels,
                                                       self.num_groups, self.num_groups)

        # Broadcast region attention back to spatial positions
        region_attn_spatial = F.interpolate(
            region_out, size=(H, W), mode='bilinear', align_corners=True)

        # FFN + residual
        x_region = x + self.gamma * region_attn_spatial
        x_region = self.region_norm(x_region)
        x_region = x_region + self.region_ffn(x_region)

        # Level 2: Offset-driven Deformable Sampling
        offsets = self.offset_conv(x_region)  # (B, K*2, H, W)
        offsets = offsets.view(B, self.num_sampling_points, 2, H, W)
        offsets = offsets.permute(0, 1, 3, 4, 2).contiguous()  # (B, K, H, W, 2)

        # Sample V with deformable offsets
        V_sampled = self._deformable_sample(V, offsets)  # (B, C, H, W, K)
        V_sampled = V_sampled.mean(dim=-1)  # (B, C, H, W)

        sw = self.sampling_weight_conv(x_region)
        x_deform = sw * V_sampled + (1 - sw) * V  # (B, C, H, W)

        out = x + self.alpha * x_deform
        return out

    def _deformable_sample(self, features, offsets):
        B, C, H, W = features.size()
        K = offsets.size(1)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=features.device),
            torch.linspace(-1, 1, W, device=features.device),
            indexing='ij')
        base_grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
        base_grid = base_grid.unsqueeze(0).unsqueeze(1)  # (1, 1, H, W, 2)
        sample_grid = base_grid + offsets  # (B, K, H, W, 2)
        sample_grid = sample_grid.view(B, K * H * W, 2)  # (B, K*H*W, 2)
        sampled = F.grid_sample(
            features, sample_grid,
            mode='bilinear', padding_mode='zeros', align_corners=True)
        sampled = sampled.view(B, C, K, H, W)
        sampled = sampled.permute(0, 1, 3, 4, 2).contiguous()  # (B, C, H, W, K)
        return sampled


@NECKS.register_module()
class DBRA_FPN(BaseModule):
    """
    DBRA-FPN: Deformable Bi-level Routing Attention Feature Pyramid Network.

    Replaces the standard FPN with DBRA modules for each pyramid level,
    enabling spatially-adaptive feature extraction for non-rigid underwater
    objects.

    Architecture:
        - Lateral connections to align channel dimensions
        - Top-down feature fusion
        - DBRA module at each FPN level (P3-P7)
        - Output convolutions for final feature refinement

    Args:
        in_channels (list[int]): Input channels for each backbone stage.
        out_channels (int): Output channels for all FPN levels.
        num_outs (int): Number of output feature maps. Default: 5 (P3-P7).
        start_level (int): Start level of backbone features. Default: 0.
        end_level (int): End level of backbone features. Default: -1.
        dbra_groups (int): Number of region groups G in DBRA Level-1 routing. Default: 4.
        dbra_sampling_points (int): Number of deformable sampling points K. Default: 9.
        init_cfg: Initialization config.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_outs=5,
        start_level=0,
        end_level=-1,
        dbra_groups=4,
        dbra_sampling_points=9,
        init_cfg=None,
    ):
        super(DBRA_FPN, self).__init__(init_cfg)

        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.start_level = start_level
        if end_level == -1:
            end_level = self.num_ins
        self.end_level = end_level

        # Lateral convolutions
        self.lateral_convs = nn.ModuleList()
        # DBRA modules
        self.dbra_convs = nn.ModuleList()
        # Output convs
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.end_level):
            self.lateral_convs.append(
                ConvModule(self.in_channels[i], out_channels, 1,
                           norm_cfg=dict(type='GN', num_groups=32)))

        for i in range(self.start_level, self.end_level):
            self.dbra_convs.append(
                DBRA(in_channels=out_channels, out_channels=out_channels,
                     num_groups=dbra_groups,
                     num_sampling_points=dbra_sampling_points))
            self.fpn_convs.append(
                ConvModule(out_channels, out_channels, 3,
                           padding=1, norm_cfg=dict(type='GN', num_groups=32)))

        # Extra levels P6, P7
        self.add_extra_convs = num_outs > (self.end_level - self.start_level)
        if self.add_extra_convs:
            extra_levels = num_outs - (self.end_level - self.start_level)
            for i in range(extra_levels):
                ch = self.in_channels[-1] if i == 0 else out_channels
                self.__setattr__(
                    f'extra_conv_{i}',
                    ConvModule(ch, out_channels, 3, stride=2, padding=1,
                               norm_cfg=dict(type='GN', num_groups=32)))

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # Build laterals
        laterals = [
            self.lateral_convs[i](inputs[i + self.start_level])
            for i in range(len(self.lateral_convs))
        ]

        # Apply DBRA at each level
        dbra_outs = [dbra_conv(lat) for dbra_conv, lat in zip(self.dbra_convs, laterals)]

        # Top-down fusion
        for i in range(len(dbra_outs) - 1, 0, -1):
            prev_size = dbra_outs[i - 1].shape[2:]
            curr_size = dbra_outs[i].shape[2:]
            if prev_size[0] != curr_size[0] or prev_size[1] != curr_size[1]:
                upsampled = F.interpolate(
                    dbra_outs[i], size=prev_size,
                    mode='bilinear', align_corners=True)
            else:
                upsampled = dbra_outs[i]
            dbra_outs[i - 1] = dbra_outs[i - 1] + upsampled

        # Output convs
        outs = [self.fpn_convs[i](dbo) for i, dbo in enumerate(dbra_outs)]

        # Extra levels
        if self.add_extra_convs:
            inp = inputs[-1]
            for i in range(self.num_outs - (self.end_level - self.start_level)):
                conv = getattr(self, f'extra_conv_{i}')
                outs.append(conv(inp if i == 0 else outs[-1]))

        return tuple(outs)
