# Copyright (c) OpenMMLab. All rights reserved.

import torch
from torch import nn

from .image import ImageClassifier


class ImageTokenClassifier(ImageClassifier):
    def __init__(
        self,
        center_size=1,
        in_channels=512,
        embed_dims=512,
        num_img_token=1,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.center_size = center_size
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.num_img_token = num_img_token
        self.img_fcs = self.build_img_fcs()

    def build_img_fcs(self):
        return nn.ModuleList(
            [nn.Linear(self.in_channels, self.embed_dims) for _ in range(2)]
        )

    def extract_center_round_feat(self, x):
        x = x.view(x.shape[0], self.num_img_token, -1, *x.shape[2:])
        pool_size = x.shape[-1]
        center_feat = x[
            ...,
            (pool_size - self.center_size) // 2 : (pool_size + self.center_size) // 2,
            (pool_size - self.center_size) // 2 : (pool_size + self.center_size) // 2,
        ]
        center_feat = torch.mean(center_feat, dim=[-2, -1])
        round_feat = (
            torch.sum(x, dim=[-2, -1]) - center_feat * self.center_size**2
        ) / (pool_size**2 - self.center_size**2)

        return center_feat, round_feat

    def extract_img_token(self, x):
        img_token = self.extract_center_round_feat(x)
        return [
            torch.cat(
                [
                    self.img_fcs[i](token.squeeze(1))
                    for i, token in enumerate(img_token)
                ],
                dim=1,
            )
        ]

    @staticmethod
    def extract_all_cls_tokens(tokens):
        return tokens

    @staticmethod
    def extract_cls_token(token):
        return token

    def token_forward(self, x, label=None):
        return self.extract_img_token(x)

    def forward_train(self, img, gt_label, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            gt_label (Tensor): It should be of shape (N, 1) encoding the
                ground-truth label of input images for single label task. It
                shoulf be of shape (N, C) encoding the ground-truth label
                of input images for multi-labels task.
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if self.augments is not None:
            img, gt_label = self.augments(img, gt_label)

        x = self.extract_feat(img)
        if isinstance(x, tuple):
            x = x[-1]

        tokens = self.token_forward(x)
        cls_token = self.extract_cls_token(tokens[-1])

        return self.head.forward_train(cls_token, gt_label, **kwargs)

    def simple_test(self, img, img_metas=None, **kwargs):
        """Test without augmentation."""
        x = self.extract_feat(img)
        if isinstance(x, tuple):
            x = x[-1]
        tokens = self.token_forward(x)
        cls_token = self.extract_cls_token(tokens[-1])

        return self.head.simple_test(cls_token, **kwargs)
