import os
import re
from typing import Optional, Union

from pytorch_lightning import LightningModule, Trainer, Callback
from pytorch_lightning.loggers.wandb import WandbLogger


class WandbLoggerLogAllCodeCallback(Callback):
    """Let wandb logger watch model when training starts."""

    def __init__(self,
                 root: str = '.',
                 name: str = None,
                 include_pattern: Union[str, list] = None,
                 exclude_pattern: Union[str, list] = None):
        self.root = root
        self.name = name

        if include_pattern is None:
            include_pattern = ['py', 'yaml', 'yml', 'json', 'sh', 'md', 'txt']
        elif isinstance(include_pattern, str):
            include_pattern = [include_pattern]
        include_pattern = [re.compile(rf'.*\.{p}$') for p in include_pattern]

        if exclude_pattern is None:
            exclude_pattern = ['wandb', 'venv', 'temp', 'tmp', 'work_dirs', 'requirements', r'\.', r'.*\.egg-info', r'setup.py']
        elif isinstance(exclude_pattern, str):
            exclude_pattern = [exclude_pattern]
        ex_p = []
        for p in exclude_pattern:
            ex_p.extend([re.compile(rf'.*/{p}.*'), re.compile(rf'^[/]{p}.*')])

        self.include_fn = lambda path: any([p.fullmatch(path) for p in include_pattern])
        self.exclude_fn = lambda path: any([p.fullmatch(path) for p in ex_p])

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None) -> None:
        if trainer.logger is not None and isinstance(trainer.logger, WandbLogger):
            if self.name is None:
                self.name = trainer.logger.experiment.project_name()
            trainer.logger.experiment.log_code(root = self.root, name = self.name, include_fn = self.include_fn,
                                               exclude_fn = self.exclude_fn)
