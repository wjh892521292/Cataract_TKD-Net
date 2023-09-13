# Copyright (c) OpenMMLab. All rights reserved.

from mmcls.models.builder import HEADS
from mmcls.models.heads import ClsHead as _ClsHead


@HEADS.register_module(force=True)
class ClsHead(_ClsHead):
    def simple_test(self, x, softmax=True, post_process=False, **kwargs):
        return super().simple_test(x, softmax=softmax, post_process=post_process)

    def forward_train(self, cls_score, gt_label, **kwargs):
        if isinstance(cls_score, tuple):
            cls_score = cls_score[-1]
        if hasattr(self.compute_loss, "use_sigmoid") and self.compute_loss.use_sigmoid:
            gt_label = kwargs["gt_smooth_label"]
        losses = self.loss(cls_score, gt_label)
        return losses
