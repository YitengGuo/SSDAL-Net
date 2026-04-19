# Copyright (c) SSDAL-Net Authors. All rights reserved.
"""
SSDALNet: Synergistic Sparse-Deformable Attention Learning Network

Paper: Adaptive Sparse-Deformable Synergistic Mechanism for Object Detection
       in Complex Underwater Scenes
Venue: The Visual Computer

Architecture:
    - Backbone: ResNet_SSDAL (ASSA-enhanced ResNet for global semantic denoising)
    - Neck: DBRA_FPN (Deformable Bi-level Routing Attention FPN for local geometric alignment)
    - Head: TOODHead (Task-aligned Object Detection Head)

The synergy of ASSA (backbone) and DBRA (neck) provides:
    1. Global: Long-range semantic dependency modeling with adaptive sparsity
    2. Local: Spatially-adaptive feature extraction for non-rigid geometric alignment
    3. Combined: Robust underwater object detection in complex optical environments
"""

from ..builder import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class SSDALNet(SingleStageDetector):
    """
    SSDALNet: Synergistic Sparse-Deformable Attention Learning Network.

    Integrates:
        - ResNet_SSDAL backbone with Adaptive Sparse Self-Attention (ASSA)
        - DBRA_FPN neck with Deformable Bi-level Routing Attention
        - TOODHead for task-aligned detection

    Args:
        backbone (dict): Backbone config. Use ResNet_SSDAL type.
        neck (dict): Neck config. Use DBRA_FPN type.
        bbox_head (dict): BBox head config. Use TOODHead type.
        train_cfg (dict | None): Training config.
        test_cfg (dict | None): Testing config.
        pretrained (str | None): Pretrained checkpoint path.
        init_cfg (dict | list | None): Initialization config.
    """

    def __init__(
        self,
        backbone,
        neck,
        bbox_head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        init_cfg=None,
    ):
        super(SSDALNet, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg,
        )

    def set_epoch(self, epoch):
        """Set epoch number for adaptive training strategies."""
        if hasattr(self.bbox_head, 'epoch'):
            self.bbox_head.epoch = epoch
