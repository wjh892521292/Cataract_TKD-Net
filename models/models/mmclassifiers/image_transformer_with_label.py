# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn

from .image_transformer import ImageTransformerClassifier


class ImageTransformerWithLabelClassifier(ImageTransformerClassifier):
    def __init__(self,
                 num_labels = 1024,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_labels = num_labels
        self.label_fcs = nn.ModuleList([nn.Linear(1, self.embed_dims) for _ in range(num_labels)])

    def extract_label_token(self, label):
        label_token = []
        for i in range(self.num_labels):
            label_token.append(self.label_fcs[i](label[:, i, None]))
        label_token = torch.stack(label_token, dim = 1)

        if self.num_transformer == 2:
            label_token = [label_token[:, :self.num_labels // 2], label_token[:, self.num_labels // 2:]]
        elif self.num_transformer == 1:
            label_token = [label_token]
        else:
            raise NotImplementedError

        return label_token

    def token_forward(self, x, label):
        img_tokens = self.extract_img_token(x)
        label_tokens = self.extract_label_token(label)
        tokens = [torch.cat([label_token, img_token], dim = 1) for img_token, label_token in zip(img_tokens, label_tokens)]
        return self.fusion_transformer(self.fusion_transformer.embed_forward(tokens))

    def forward_train(self, img, label, gt_label, **kwargs):
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

        tokens = self.token_forward(x, label.to(x.dtype))
        cls_token = self.extract_cls_token(tokens[-1])

        return self.head.forward_train(cls_token, gt_label, **kwargs)

    def simple_test(self, img, label, img_metas = None, **kwargs):
        """Test without augmentation."""
        x = self.extract_feat(img)
        if isinstance(x, tuple):
            x = x[-1]
        tokens = self.token_forward(x, label.to(x.dtype))
        cls_token = self.extract_cls_token(tokens[-1])

        return self.head.simple_test(cls_token, **kwargs)
