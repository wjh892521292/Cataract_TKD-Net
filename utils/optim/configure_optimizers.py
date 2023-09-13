import copy
from typing import List, Sequence, Tuple, Union

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.cli import instantiate_class
from pytorch_lightning.utilities.types import LRSchedulerType
from torch.optim import Optimizer


def parser_optimizer_config(model, optimizer_config):
    optimizer_config = copy.deepcopy(optimizer_config)
    # parse the optimizer config
    if 'optimizer' not in optimizer_config:
        optimizer_config = {'optimizer': optimizer_config}
    if not isinstance(optimizer_config['optimizer'], Sequence):
        parameters = model.configure_optimizer_parameters()
        if parameters is not None:
            optimizer_config['optimizer'] = [copy.deepcopy(optimizer_config['optimizer']) for _ in parameters]
        else:
            optimizer_config['optimizer'] = [optimizer_config['optimizer']]

    if 'lr_scheduler' in optimizer_config:
        if not isinstance(optimizer_config['lr_scheduler'], Sequence):
            optimizer_config['lr_scheduler'] = [copy.deepcopy(optimizer_config['lr_scheduler']) for _ in optimizer_config['optimizer']]
    return optimizer_config


def construct_optimizer(model, optimizer, params = None, set_lr = False):
    """
    Constructs the optimizer.

    Args:
        model: a LightningModule.
        params: a list of parameters to optimize, if None, all parameters of model will be optimized.
        optimizer: dictionary containing optimizer configuration.
        set_lr: whether to set the learning rate by the model.lr
    """
    if model.lr is not None and set_lr:
        if 'init_args' not in optimizer:
            optimizer['init_args'] = {}
        optimizer['init_args']['lr'] = model.lr
    optimizer = instantiate_class(model.parameters() if params is None else params, optimizer)
    if set_lr and model.lr is None:
        model.lr = optimizer.param_groups[0]['lr']
    return optimizer


def construct_optimizers(model, optimizers):
    """
    Constructs all optimizers.

    Args:
        model: a LightningModule.
        optimizers: list of dictionary containing optimizer configuration.
    """
    optimizer_parameters = model.configure_optimizer_parameters()
    if optimizer_parameters is None:
        optimizer_parameters = [None] * len(optimizers)
    assert isinstance(optimizer_parameters, list), 'optimizer_parameters should be None or list'
    if len(optimizer_parameters) < len(optimizers):
        optimizer_parameters += [None] * (len(optimizers) - len(optimizer_parameters))

    for i in range(len(optimizers)):
        optimizers[i] = construct_optimizer(model, optimizers[i], optimizer_parameters[i], set_lr = i == 0)

    return optimizers


def construct_lr_schedulers(lr_schedulers, optimizers):
    """
    Constructs all lr_schedulers.

    Args:
        lr_schedulers: list of dictionary containing lr_scheduler configuration.
        optimizers: list of optimizers constructed.
    """
    constructed_lr_schedulers = []
    warmup_lr_schedulers = []
    opt_idx = -1
    # construct lr_scheduler
    for lr_scheduler in lr_schedulers:
        # select optimizer
        if len(optimizers) == 1:
            opt_idx = 0
        else:
            if 'opt_idx' in lr_scheduler:
                opt_idx = lr_scheduler['opt_idx']
            else:
                opt_idx += 1
        optimizer = optimizers[opt_idx]

        # construct lr_scheduler
        if 'scheduler' not in lr_scheduler:
            lr_scheduler = {'scheduler': lr_scheduler}
        lr_scheduler['scheduler'] = instantiate_class(optimizer, lr_scheduler['scheduler'])
        lr_scheduler['opt_idx'] = opt_idx
        constructed_lr_schedulers.append(lr_scheduler)

        # construct warmup_lr_scheduler
        if 'warmup_config' in lr_scheduler:
            warmup_config = lr_scheduler.pop('warmup_config')
            if 'scheduler' not in warmup_config:
                warmup_config = {'scheduler': {'class_path': 'utils.optim.WarmupScheduler',
                                               'init_args': warmup_config}}
            warmup_config['scheduler'] = instantiate_class(optimizer, warmup_config['scheduler'])
            warmup_config.update({'interval': 'step', 'opt_idx': opt_idx})
            warmup_lr_schedulers.append(warmup_config)
    return constructed_lr_schedulers + warmup_lr_schedulers


def get_configure_optimizers_method(optimizer_config):
    def configure_optimizers(self: LightningModule) -> Union[Optimizer, Tuple[List[Optimizer], List[LRSchedulerType]]]:
        optimizer_cfg = parser_optimizer_config(self, optimizer_config)
        # construct optimizer
        optimizer_cfg['optimizer'] = construct_optimizers(self, optimizer_cfg['optimizer'])
        # construct lr_scheduler
        if 'lr_scheduler' in optimizer_cfg:
            optimizer_cfg['lr_scheduler'] = construct_lr_schedulers(optimizer_cfg['lr_scheduler'],
                                                                    optimizer_cfg['optimizer'])
            return optimizer_cfg['optimizer'], optimizer_cfg['lr_scheduler']
        return optimizer_cfg['optimizer']

    return configure_optimizers
