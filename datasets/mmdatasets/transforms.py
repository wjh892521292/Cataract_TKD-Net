# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcls.datasets.builder import PIPELINES as MMCLS_PIPELINES
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import DefaultFormatBundle as _DefaultFormatBundle
from mmdet.datasets.pipelines.formatting import to_tensor


@PIPELINES.register_module()
class GenSegmentationFromBBox:
    """Generate segmentation from bbox.

    Added key is "gt_segments_from_bboxes".

    Args:
        num_classes (int): number of classes. Generate instance segmentation,
            if is None, else semantic segmentation. Default : None.
    """

    def __init__(self, num_classes = None):
        self.num_classes = num_classes

    def __call__(self, results):
        """Call function to generate segmentation.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Segmentation generated results, 'gt_segments_from_bboxes' key
                is added into result dict.
        """
        img_shape = results['img_shape'][:2]
        if self.num_classes is None:
            kernels = [self.__gen_gaussian_kernel(bbox, img_shape) for bbox in results['gt_bboxes']]
        else:
            kernels = np.zeros((self.num_classes, *img_shape), dtype = np.float32)
            for bbox, label in zip(results['gt_bboxes'], results['gt_labels']):
                kernels[label - 1, :, :] += self.__gen_gaussian_kernel(bbox, img_shape)

        results['gt_segments_from_bboxes'] = kernels
        results['seg_fields'].append('gt_segments_from_bboxes')
        return results

    @staticmethod
    def __gen_gaussian_kernel(bbox, img_shape):
        mu = [(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2]
        sigma = [(bbox[3] - bbox[1]) / 3, (bbox[2] - bbox[0]) / 3]
        h, w = [np.exp(-np.power((np.arange(1., sh + 1) - m) / s, 2) / 2) for sh, m, s in zip(img_shape, mu, sigma)]
        kernel = h[:, np.newaxis] * w[np.newaxis, :]
        kernel = 9 * sigma[0] * sigma[1] * kernel / np.sum(kernel)
        return kernel.astype(np.float32)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(num_classes={self.num_classes})'
        return repr_str


@MMCLS_PIPELINES.register_module()
@PIPELINES.register_module()
class GenSmoothLabel:
    """Generate smooth label.

    Added key is "gt_smooth_label".

    Args:
        num_classes (int): number of classes. Generate instance segmentation,
            if is None, else semantic segmentation. Default : None.
    """

    def __init__(self, num_classes = None, pow = 1):
        self.num_classes = num_classes
        self.pow = pow

    def __call__(self, results):
        """Call function to generate segmentation.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Segmentation generated results, 'gt_segments_from_bboxes' key
                is added into result dict.
        """
        gt_smooth_label = np.exp(-np.power(np.abs(np.arange(self.num_classes) - results['gt_label']), self.pow))
        gt_smooth_label /= np.sum(gt_smooth_label)
        results['gt_smooth_label'] = gt_smooth_label
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(num_classes={self.num_classes})'
        return repr_str


@MMCLS_PIPELINES.register_module()
@PIPELINES.register_module()
class GenDimlyLabel:
    """Generate Dimly label.

    Added key is "gt_dimly_label".

    Args:
        num_classes (int): number of classes. Generate instance segmentation,
            if is None, else semantic segmentation. Default : None.
    """

    def __init__(self, num_classes = None):
        self.num_classes = num_classes

    def __call__(self, results):
        """Call function to generate segmentation.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Segmentation generated results, 'gt_segments_from_bboxes' key
                is added into result dict.
        """
        gt_dimly_label = np.zeros(self.num_classes, dtype = np.int64)
        gt_dimly_label[results['gt_min_label']:results['gt_max_label'] + 1] = 1
        results['gt_dimly_label'] = gt_dimly_label
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(num_classes={self.num_classes})'
        return repr_str


@MMCLS_PIPELINES.register_module()
@PIPELINES.register_module()
class GenRotateImage:
    """Generate rotated image.

    Added key is "img_rotate".
    """

    def __call__(self, results):
        """Call function to generate segmentation.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Segmentation generated results, 'gt_segments_from_bboxes' key
                is added into result dict.
        """
        img_rotate = [results['img'], np.rot90(results['img']), np.rot90(results['img'], 2), np.rot90(results['img'], -1)]
        results['img_rotate'] = np.stack(img_rotate).transpose(0, 3, 1, 2)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'()'
        return repr_str


@PIPELINES.register_module(force = True)
class DefaultFormatBundle(_DefaultFormatBundle):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor, \
                       (3)to DataContainer (stack=True)
    - gt_segments_from_bboxes (1)to tensor, (2)to DataContainer (stack=True)
    """

    def __call__(self, results):
        results = super().__call__(results)
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(to_tensor(results['gt_semantic_seg']), stack = True)
        return results
