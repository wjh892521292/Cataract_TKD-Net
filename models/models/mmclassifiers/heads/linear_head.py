# Copyright (c) OpenMMLab. All rights reserved.

from mmcls.models.builder import HEADS
from mmcls.models.heads import LinearClsHead as _LinearClsHead


@HEADS.register_module(force=True)
class LinearClsHead(_LinearClsHead):
    def simple_test(self, x, softmax=True, post_process=False, **kwargs):
        return super().simple_test(x, softmax=softmax, post_process=post_process)

    def forward_train(self, x, gt_label, **kwargs):
        x = self.pre_logits(x)
        cls_score = self.fc(x)
        if hasattr(self.compute_loss, "use_sigmoid") and self.compute_loss.use_sigmoid:
            gt_label = kwargs["gt_smooth_label"]
        losses = self.loss(cls_score, gt_label)
        return losses
