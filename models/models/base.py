from abc import ABC
from typing import Any, Mapping, Union

import torch
from mmcv.runner import BaseModule
from pytorch_lightning import LightningModule as _LightningModule


class LightningModule(_LightningModule, BaseModule, ABC):
    def __init__(self,
                 normalize_config = None,
                 loss_config: Mapping[str, Union[torch.nn.Module, Mapping[str, Union[torch.nn.Module, int, float]]]] = None,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.automatic_lr_schedule = True
        self.normalize_config = normalize_config
        self.lr = None
        self.batch_size = None

        if loss_config is not None:
            self._parse_loss_config(loss_config)

    def configure_optimizer_parameters(self):
        return None

    @staticmethod
    def add_prefix(log_dict, prefix = 'train/'):
        return {f'{prefix}{k}': v for k, v in log_dict.items()}

    def log(self, *args, batch_size = None, **kwargs):
        if batch_size is None and hasattr(self, 'batch_size') and self.batch_size is not None:
            batch_size = self.batch_size
        super().log(*args, batch_size = batch_size, **kwargs)

    def _parse_loss_config(self, loss_config):
        for key, value in loss_config.items():
            if not isinstance(value, Mapping):
                loss_config[key] = {'module': value, 'weight': 1}
            setattr(self, 'loss_' + key, loss_config[key]['module'])
        self.loss_weight = {k: v.get('weight', 1) for k, v in loss_config.items()}

    def _loss_step(self, batch, res, prefix = 'train'):
        raise NotImplementedError

    def loss_step(self, batch, res, prefix = 'train', use_loss_weight = True, loss_use_loss_weight = True, detach = None):
        loss = self._loss_step(batch, res, prefix)
        # multi loss weights
        if use_loss_weight:
            loss = {k: v * (1 if k not in self.loss_weight else self.loss_weight[k]) for k, v in loss.items()}
        # calculate loss
        if not use_loss_weight and loss_use_loss_weight:
            total_loss = [v * (1 if k not in self.loss_weight else self.loss_weight[k]) for k, v in loss.items()]
        else:
            total_loss = [v for k, v in loss.items()]
        loss['loss'] = torch.sum(torch.stack(total_loss))
        # add prefix
        if detach is None:
            detach = prefix != 'train'
        loss = {(f'{prefix}/' if prefix is not None else '') + ('loss_' if 'loss' not in k else '') + k: (v.detach() if detach else v) for
                k, v in loss.items()}
        return loss

    def on_fit_start(self):
        self.init_weights()

    def _dump_init_info(self, logger_name):
        pass

    def training_step(self, batch, *args, **kwargs):
        res = self(batch)
        loss = self.loss_step(batch, res, 'train')
        self.log_dict(loss)
        return loss['train/loss']

    def validation_step(self, batch, *args, **kwargs):
        res = self(batch)
        loss = self.loss_step(batch, res, 'val')
        self.log_dict(loss, sync_dist = True)
        return loss

    def test_step(self, batch, *args, **kwargs):
        res = self(batch)
        loss = self.loss_step(batch, res, 'test')
        self.log_dict(loss, sync_dist = True)
        return loss
