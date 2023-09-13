import csv
import os
from abc import ABC

import mmcv
import torch
from torch import nn
from torchmetrics import (
    Accuracy,
    CohenKappa,
    F1Score,
    MatthewsCorrCoef,
    Precision,
    Recall,
    Specificity,
)

from ..mmdetectors import MMDetModelAdapter


class MMClsModelAdapter(MMDetModelAdapter, ABC):
    def get_default_metrics(self):
        metrics = nn.ModuleList(
            [
                Accuracy("multiclass", num_classes=7),
                Precision("multiclass", average="macro", num_classes=7),
                Recall("multiclass", average="macro", num_classes=7),
                F1Score("multiclass", average="macro", num_classes=7),
                Specificity("multiclass", average="macro", num_classes=7),
                CohenKappa("multiclass", num_classes=7),
                MatthewsCorrCoef("multiclass", num_classes=7),
            ]
        )
        metrics_log_info = [{"prog_bar": ["acc"]}]
        return metrics, metrics_log_info

    def convert_raw_predictions(self, batch, preds):
        """Convert raw predictions from the model to library standard."""
        return preds, batch["gt_label"]

    def on_predict_start(self) -> None:
        super().on_predict_start()

        output_paths = ["correct", "wrong"]
        for name in output_paths:
            path = os.path.join(self.result_output_path, name)
            self.rm_and_create(path)
            self.__setattr__(name + "_output_path", path)

        self.correct_preds = []
        self.wrong_preds = []

    def result_visualization(self, batch, *args, **kwargs):
        preds = self.model.simple_test(**batch)
        preds, target = self.convert_raw_predictions(batch, preds)
        pred_inds = torch.argmax(preds, dim=1)
        result = pred_inds == target
        imgs = batch["img"].permute(0, 2, 3, 1).cpu().numpy()
        for i, res in enumerate(result):
            mmcv.imwrite(
                self.denormalize_img(imgs[i], batch["img_metas"][i]["img_norm_cfg"]),
                os.path.join(
                    self.correct_output_path if res else self.wrong_output_path,
                    f"gt={target[i]}_pred={pred_inds[i]}_"
                    + batch["img_metas"][i]["ori_filename"],
                ),
            )
            pred_list = self.correct_preds if res else self.wrong_preds
            pred_list.append(
                [
                    res.item(),
                    target[i].item(),
                    pred_inds[i].item(),
                    *preds[i].cpu().numpy().tolist(),
                    batch["img_metas"][i]["ori_filename"],
                ]
            )

    def on_predict_end(self) -> None:
        preds = self.wrong_preds + self.correct_preds
        if len(preds):
            file_header = (
                ["correct", "gt", "pred"]
                + [f"score_{i}" for i in range(len(preds[0]) - 4)]
                + ["filename"]
            )
            with open(os.path.join(self.result_output_path, "preds.csv"), "w") as f:
                writer = csv.writer(f)
                writer.writerow(file_header)
                writer.writerows(preds)
