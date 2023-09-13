# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcls.models.builder import NECKS


@NECKS.register_module()
class CataractEyeNet(nn.Module):
    """Adaptive Average Pooling neck.

    Note that we use `view` to remove extra channel after pooling. We do not
    use `squeeze` as it will also remove the batch dimension when the tensor
    has a batch dimension of size 1, which can lead to unexpected errors.

    Args:
        dim (int): Dimensions of each sample channel, can be one of {1, 2, 3}.
            Default: 2
    """

    def __init__(self, in_channels=512, num_classes=1000):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=3, padding=3),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 64, kernel_size=3, padding=3),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=3),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(128, 128, kernel_size=3, padding=3),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),
            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=3),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(256, 256, kernel_size=3, padding=3),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(256, 256, kernel_size=3, padding=3),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),
            nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, padding=3),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(512, 512, kernel_size=3, padding=3),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(512, 512, kernel_size=3, padding=3),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),
            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=3),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(512, 512, kernel_size=3, padding=3),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(512, 512, kernel_size=3, padding=3),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),
        )

        self.fc = nn.Linear(512, num_classes)
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        pass

    def forward(self, inputs):
        outs = self.layers(inputs[0])
        outs = self.fc(outs.flatten(start_dim=1))
        outs = self.sigmoid(outs)
        return outs
