from typing import Optional

from pytorch_lightning import LightningModule, Trainer, Callback
from pytorch_lightning.loggers.wandb import WandbLogger


class WandbLoggerWatchModelCallback(Callback):
    """Let wandb logger watch model when training starts."""

    def __init__(self,
                 log: Optional[str] = 'gradients',
                 log_freq: Optional[int] = 1000,
                 log_graph: Optional[bool] = False):
        self.log_attr = log
        self.log_freq = log_freq
        self.log_graph = log_graph

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None) -> None:
        if trainer.logger is not None and isinstance(trainer.logger, WandbLogger):
            trainer.logger.watch(model = pl_module, log = self.log_attr, log_freq = self.log_freq, log_graph = self.log_graph)
