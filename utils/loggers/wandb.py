import os
from typing import Dict, Optional

from pytorch_lightning.loggers.logger import rank_zero_experiment
from pytorch_lightning.loggers.wandb import Run, WandbLogger, wandb
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn
from pytorch_lightning.utilities.logger import _add_prefix


class WandbNamedLogger(WandbLogger):
    @property
    @rank_zero_experiment
    def experiment(self) -> Run:
        r"""

        Actual wandb object. To use wandb features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.

        Example::

        .. code-block:: python

            self.logger.experiment.some_wandb_function()

        """
        if self._experiment is None:
            if self._offline:
                os.environ["WANDB_MODE"] = "dryrun"

            attach_id = getattr(self, "_attach_id", None)
            if wandb.run is not None:
                # wandb process already created in this instance
                rank_zero_warn(
                    "There is a wandb run already in progress and newly created instances of `WandbLogger` will reuse"
                    " this run. If this is not desired, call `wandb.finish()` before instantiating `WandbLogger`."
                )
                self._experiment = wandb.run
            elif attach_id is not None and hasattr(wandb, "_attach"):
                # attach to wandb process referenced
                self._experiment = wandb._attach(attach_id)
            else:
                # create new wandb process
                self._experiment = wandb.init(**self._wandb_init)

                # define default x-axis
                if getattr(self._experiment, "define_metric", None):
                    self._experiment.define_metric("global_step")
                    self._experiment.define_metric("*", step_metric = "global_step", step_sync = True)

        return self._experiment

    @property
    def name(self) -> Optional[str]:
        """Gets the name of the experiment.

        Returns:
            The name of the experiment if the experiment exists else the name given to the constructor.
        """
        # don't create an experiment if we don't have one
        return self._experiment.name if self._experiment else self._name

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"

        metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)
        if step is not None:
            self.experiment.log({**metrics, "global_step": step})
        else:
            self.experiment.log(metrics)
