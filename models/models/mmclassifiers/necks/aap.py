# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmcls.models.builder import NECKS


@NECKS.register_module()
class AdaptiveAveragePooling(nn.Module):
    """Adaptive Average Pooling neck.

    Note that we use `view` to remove extra channel after pooling. We do not
    use `squeeze` as it will also remove the batch dimension when the tensor
    has a batch dimension of size 1, which can lead to unexpected errors.

    Args:
        dim (int): Dimensions of each sample channel, can be one of {1, 2, 3}.
            Default: 2
    """

    def __init__(self, pool_size = (1, 1), dim = 2):
        super().__init__()
        assert dim in [1, 2, 3], 'AdaptiveAveragePooling dim only support ' \
                                 f'{1, 2, 3}, get {dim} instead.'
        if dim == 1:
            self.aap = nn.AdaptiveAvgPool1d(pool_size)
        elif dim == 2:
            self.aap = nn.AdaptiveAvgPool2d(pool_size)
        else:
            self.aap = nn.AdaptiveAvgPool3d(pool_size)

    def init_weights(self):
        pass

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            outs = tuple([self.aap(x) for x in inputs])
        elif isinstance(inputs, torch.Tensor):
            outs = self.aap(inputs)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs
