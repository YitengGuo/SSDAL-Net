# Copyright (c) SSDAL-Net Authors. All rights reserved.
"""
Adaptive Sparse Self-Attention (ASSA) Module

Paper: Adaptive Sparse-Deformable Synergistic Mechanism for Object Detection
       in Complex Underwater Scenes
Venue: The Visual Computer

Key Innovation:
    - Squared-ReLU sparse mask: M = ReLU(QK^T)^2, dynamically suppresses
      background noise responses
    - Hybrid fusion: W = alpha * S_dense + (1-alpha) * S_sparse
    - Learnable temperature and fusion weight for adaptive sparsity control
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule


class ASSA(BaseModule):
    """
    Adaptive Sparse Self-Attention Module.

    Integrates dense self-attention with a squared-ReLU sparse mask to
    simultaneously model long-range dependencies and suppress redundant
    background scattering in underwater scenes.

    Args:
        in_channels (int): Number of input channels.
        key_channels (int): Number of key/query channels (default: in_channels // 8).
        value_channels (int): Number of value channels (default: in_channels).
        num_heads (int): Number of attention heads (default: 1).
        temperature (float): Initial temperature for softmax (default: 1.0).
        init_cfg (dict | None): Initialization config.
    """

    def __init__(
        self,
        in_channels,
        key_channels=None,
        value_channels=None,
        num_heads=1,
        temperature=1.0,
        init_cfg=None,
    ):
        super(ASSA, self).__init__(init_cfg)
        self.in_channels = in_channels
        key_channels = key_channels or in_channels // 8
        value_channels = value_channels or in_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.num_heads = num_heads
        self.temperature = temperature

        # QKV projections
        self.query_conv = nn.Conv2d(
            in_channels, key_channels * num_heads, kernel_size=1
        )
        self.key_conv = nn.Conv2d(
            in_channels, key_channels * num_heads, kernel_size=1
        )
        self.value_conv = nn.Conv2d(
            in_channels, value_channels * num_heads, kernel_size=1
        )

        # Learnable fusion weight: alpha controls dense vs sparse ratio
        # alpha=1 -> pure dense attention; alpha=0 -> pure sparse
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # Learnable residual weight
        self.gamma = nn.Parameter(torch.tensor(1.0))

        # Output projection
        self.out_conv = nn.Conv2d(
            value_channels * num_heads, in_channels, kernel_size=1
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input feature map, shape (B, C, H, W).

        Returns:
            Tensor: Enhanced feature map, shape (B, C, H, W).
        """
        B, C, H, W = x.size()

        # Project Q, K, V
        # Q: (B, num_heads * key_channels, H, W)
        Q = self.query_conv(x)
        K = self.key_conv(x)
        V = self.value_conv(x)

        # Reshape to (B * num_heads, key_channels, H*W) for matrix multiply
        Q = Q.view(B * self.num_heads, self.key_channels, H * W)
        K = K.view(B * self.num_heads, self.key_channels, H * W)
        V = V.view(B * self.num_heads, self.value_channels, H * W)

        # ---- Dense Attention (standard softmax attention) ----
        energy_dense = torch.bmm(Q.transpose(1, 2), K)  # (B*heads, HW, HW)
        energy_dense = energy_dense / (self.key_channels ** 0.5)
        attention_dense = self.softmax(energy_dense)  # (B*heads, HW, HW)

        # ---- Squared-ReLU Sparse Mask ----
        # M = ReLU(QK^T)^2: clamp to avoid overflow, then square
        energy_raw = torch.bmm(Q.transpose(1, 2), K)  # (B*heads, HW, HW)
        sparse_mask = F.relu(energy_raw).pow(2)  # (B*heads, HW, HW)
        # Normalize sparse mask to [0, 1] per row (across key dimension)
        sparse_max = sparse_mask.max(dim=-1, keepdim=True)[0].clamp(min=1e-6)
        sparse_mask_norm = sparse_mask / sparse_max
        # Apply temperature scaling before softmax fusion
        attention_sparse = self.softmax(sparse_mask_norm / self.temperature)

        # ---- Hybrid Fusion ----
        # W = sigmoid(alpha) * S_dense + (1 - sigmoid(alpha)) * S_sparse
        alpha_weight = torch.sigmoid(self.alpha)
        attention_hybrid = (
            alpha_weight * attention_dense
            + (1 - alpha_weight) * attention_sparse
        )

        # ---- Compute output ----
        out = torch.bmm(V, attention_hybrid.transpose(1, 2))  # (B*heads, val_C, HW)
        out = out.view(B, self.num_heads * self.value_channels, H, W)

        # ---- Output projection + residual ----
        out = self.out_conv(out)
        out = self.gamma * out + x

        return out


class ASSALayer(BaseModule):
    """
    A single layer containing ASSA followed by FFN (Feed-Forward Network).
    Can be inserted after any ResNet stage for feature enhancement.
    """

    def __init__(
        self,
        in_channels,
        expansion=4,
        key_channels=None,
        value_channels=None,
        num_heads=1,
        temperature=1.0,
        init_cfg=None,
    ):
        super(ASSALayer, self).__init__(init_cfg)
        self.assa = ASSA(
            in_channels=in_channels * expansion,
            key_channels=key_channels,
            value_channels=value_channels,
            num_heads=num_heads,
            temperature=temperature,
        )
        # FFN: two conv layers with GELU activation
        self.ffn = nn.Sequential(
            nn.Conv2d(in_channels * expansion, in_channels * expansion * 4, 1),
            nn.GELU(),
            nn.Conv2d(in_channels * expansion * 4, in_channels * expansion, 1),
        )
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels * expansion)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=in_channels * expansion)

    def forward(self, x):
        # ASSA with residual
        identity = x
        x = self.norm1(x)
        x = self.assa(x)
        x = x + identity

        # FFN with residual
        identity2 = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + identity2

        return x
