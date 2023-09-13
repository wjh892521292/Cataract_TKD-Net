import csv

import numpy as np
from mmcls.datasets import BaseDataset
from mmdet.datasets import DATASETS


@DATASETS.register_module()
class CataractDataSet(BaseDataset):
    CLASSES = ['0'] + [str(i) for i in range(2, 7)]

    def __init__(self,
                 label_indices = None,
                 *args, **kwargs):
        self.label_indices = label_indices
        super().__init__(*args, **kwargs)

    def load_annotations(self):
        data_infos = []
        for line in csv.reader(open(self.ann_file)):
            if line[1] not in self.CLASSES:
                continue
            info = {
                'img_prefix': self.data_prefix,
                'img_info': {'filename': line[0]},
                'gt_label': np.array(line[1], dtype = np.int64),
                'gt_min_label': np.array(line[2], dtype = np.int64),
                'gt_max_label': np.array(line[3], dtype = np.int64)
            }
            if len(line) > 4:
                if self.label_indices is None:
                    info['label'] = np.array(line[4:], dtype = np.float32)
                else:
                    label = [line[4 + i] for i in self.label_indices]
                    info['label'] = np.array(label, dtype = np.int64)
            data_infos.append(info)
        return data_infos
