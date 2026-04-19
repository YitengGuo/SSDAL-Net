# Copyright (c) SSDAL-Net Authors. All rights reserved.
"""
ResNet-SSDAL: ResNet backbone with Adaptive Sparse Self-Attention (ASSA)

Paper: Adaptive Sparse-Deformable Synergistic Mechanism for Object Detection
       in Complex Underwater Scenes
Venue: The Visual Computer

Modification over standard ResNet:
    - Stage 3 (layer3) output: ASSA is applied after the residual blocks
    - Stage 4 (layer4) output: ASSA is applied after the residual blocks
    - ASLA layers enhance long-range semantic dependencies and suppress
      redundant background scattering responses in underwater scenes.
"""

import warnings

import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from ..builder import BACKBONES
from ..utils import ResLayer
from .resnet import BasicBlock, Bottleneck


class ASSAForResNet(nn.Module):
    """
    ASSA (Adaptive Sparse Self-Attention) integration wrapper for ResNet.

    This wrapper applies ASSA to the output of a ResNet stage, enabling
    global semantic denoising without changing the spatial resolution.
    """

    def __init__(self, in_channels, assa_key_channels=None, assa_value_channels=None,
                 assa_num_heads=1, assa_temperature=1.0):
        super().__init__()
        from ..utils.assa import ASSA
        self.assa = ASSA(
            in_channels=in_channels,
            key_channels=assa_key_channels,
            value_channels=assa_value_channels,
            num_heads=assa_num_heads,
            temperature=assa_temperature,
        )
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        # Pure residual: out = x + gamma * ASSA(x)
        # gamma=0 during early training, gradually learned
        out = x + self.gamma * self.assa(x)
        return out


@BACKBONES.register_module()
class ResNet_SSDAL(BaseModule):
    """
    ResNet backbone with ASSA enhancement on stage 3 and stage 4.

    This backbone extends standard ResNet with ASSA modules that:
        1. Suppress redundant background scattering in underwater images
        2. Enhance global semantic dependencies for salient targets
        3. Improve detection of small-scale and irregularly-deformed objects

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        stage3_assa (bool): Whether to apply ASSA after stage 3 (layer3). Default: True.
        stage4_assa (bool): Whether to apply ASSA after stage 4 (layer4). Default: True.
        assa_key_channels (int | None): Key channels for ASSA. Default: None (use in_channels // 8).
        assa_num_heads (int): Number of attention heads. Default: 1.
        assa_temperature (float): Temperature for sparse attention. Default: 1.0.
        **kwargs: All other arguments passed to ResNet.
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(
        self,
        depth,
        in_channels=3,
        stem_channels=None,
        base_channels=64,
        num_stages=4,
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        out_indices=(0, 1, 2, 3),
        style='pytorch',
        deep_stem=False,
        avg_down=False,
        frozen_stages=-1,
        conv_cfg=None,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        dcn=None,
        stage_with_dcn=(False, False, False, False),
        plugins=None,
        with_cp=False,
        zero_init_residual=True,
        pretrained=None,
        init_cfg=None,
        # ASSA-specific args
        stage3_assa=True,
        stage4_assa=True,
        assa_key_channels=None,
        assa_num_heads=1,
        assa_temperature=1.0,
    ):
        # Build base ResNet
        super(ResNet_SSDAL, self).__init__(init_cfg)

        block, stage_blocks = self.arch_settings[depth]
        self.block = block
        self.stage_blocks = stage_blocks[:num_stages]

        self.depth = depth
        self.num_stages = num_stages
        self.strides = strides
        self.dilations = dilations
        self.out_indices = out_indices
        self.style = style
        self.base_channels = base_channels
        self.stem_channels = stem_channels or base_channels
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.stage_with_dcn = stage_with_dcn
        self.zero_init_residual = zero_init_residual

        # ASSA settings
        self.stage3_assa = stage3_assa
        self.stage4_assa = stage4_assa
        self.assa_key_channels = assa_key_channels
        self.assa_num_heads = assa_num_heads
        self.assa_temperature = assa_temperature

        # Init cfg
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth}')
        block_init_cfg = None
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('pretrained is deprecated, use init_cfg instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(type='Constant', val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                ]
                if self.zero_init_residual:
                    if block is BasicBlock:
                        block_init_cfg = dict(
                            type='Constant', val=0,
                            override=dict(name='norm2'))
                    elif block is Bottleneck:
                        block_init_cfg = dict(
                            type='Constant', val=0,
                            override=dict(name='norm3'))

        # Build stem
        self._make_stem_layer(in_channels, self.stem_channels)

        # Build stages
        self.res_layers = []
        self.assa_modules = {}  # stage_idx -> ASSAForResNet

        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            stage_dcn = self.stage_with_dcn[i]
            planes = base_channels * 2 ** i
            inplanes = self.stem_channels if i == 0 else base_channels * 2 ** (i - 1) * block.expansion

            res_layer = self._make_res_layer(
                block=block,
                inplanes=inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn if stage_dcn else None,
                plugins=plugins,
                init_cfg=block_init_cfg,
            )
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

            # Insert ASSA after stage 3 (index 2) and stage 4 (index 3)
            stage_idx = i + 1
            if (stage_idx == 3 and stage3_assa) or (stage_idx == 4 and stage4_assa):
                out_channels = planes * block.expansion
                assa = ASSAForResNet(
                    in_channels=out_channels,
                    assa_key_channels=assa_key_channels,
                    assa_num_heads=assa_num_heads,
                    assa_temperature=assa_temperature,
                )
                assa_name = f'assa_stage{stage_idx}'
                self.add_module(assa_name, assa)
                self.assa_modules[stage_idx] = assa_name

        self.feat_dim = block.expansion * base_channels * 2 ** (len(self.stage_blocks) - 1)
        self._freeze_stages()

    def _make_stem_layer(self, in_channels, stem_channels):
        self.conv1 = build_conv_layer(
            None, in_channels, stem_channels, kernel_size=7,
            stride=2, padding=3, bias=False)
        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, stem_channels, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _make_res_layer(self, **kwargs):
        return ResLayer(**kwargs)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.norm1.eval()
            for m in [self.conv1, getattr(self, self.norm1_name)]:
                for param in m.parameters():
                    param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            stage_idx = i + 1
            # Apply ASSA after stage 3 and stage 4
            if stage_idx in self.assa_modules:
                assa_name = self.assa_modules[stage_idx]
                x = getattr(self, assa_name)(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        super(ResNet_SSDAL, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
