# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcls.models import build_loss
from torch import nn

from .image_transformer import ImageTransformerClassifier
from .image_transformer_with_label import ImageTransformerWithLabelClassifier


class ImageTransformerDistillationLabelTeacherClassifier(
    ImageTransformerWithLabelClassifier
):
    def distillation_forward(self, img, label):
        x = self.extract_feat(img)
        if isinstance(x, tuple):
            x = x[-1]

        tokens = self.token_forward(x, label)
        cls_token = self.extract_all_cls_tokens(tokens)
        return {
            "cls_token": cls_token,
            "label": self.head.pre_logits(cls_token[-1]),
        }


class ImageTransformerDistillationLabelClassifier(ImageTransformerClassifier):
    def __init__(
        self,
        teacher_classifier: nn.Module = None,
        distillation_loss=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.teacher_classifier = teacher_classifier
        self.train_teacher = self.teacher_classifier.init_cfg is None
        if not self.train_teacher:
            for p in self.teacher_classifier.parameters():
                p.requires_grad = False

        if distillation_loss is None:
            distillation_loss = {}
        self.distillation_loss = nn.ModuleDict()
        for k, v in distillation_loss.items():
            if isinstance(v, dict):
                self.distillation_loss[k] = build_loss(v)
            elif isinstance(v, nn.Module):
                self.distillation_loss[k] = v

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
        cls_token = self.extract_all_cls_tokens(tokens)
        distillation_target = self.teacher_classifier.distillation_forward(
            img, label.to(x.dtype)
        )

        distillation_predict = {
            "cls_token": cls_token,
            "label": self.head.pre_logits(cls_token[-1]),
        }

        losses = dict()
        losses["loss_cls"] = self.head.forward_train(cls_token[-1], gt_label, **kwargs)[
            "loss"
        ]
        for name, loss in self.distillation_loss.items():
            target, predict = distillation_target[name], distillation_predict[name]
            if not isinstance(target, list):
                target, predict = [target], [predict]
            losses[f"loss_{name}_distillation"] = torch.mean(
                torch.stack([loss(t, p) for t, p in zip(target, predict)])
            )
        if self.train_teacher:
            losses["loss_teacher"] = self.teacher_classifier.forward_train(
                img, label, gt_label, **kwargs
            )["loss"]
        return losses
