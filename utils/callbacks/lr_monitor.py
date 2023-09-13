from collections import defaultdict
from typing import Any, DefaultDict, List, Optional, Tuple, Type

from pytorch_lightning.utilities.types import LRSchedulerConfig
from torch.optim.optimizer import Optimizer

from pytorch_lightning.callbacks import LearningRateMonitor as _LearningRateMonitor


class LearningRateMonitor(_LearningRateMonitor):

    def __init__(self, logging_interval: Optional[str] = None, log_momentum: bool = False, name_prefix: Optional[str] = 'lr/'):
        super().__init__(logging_interval = logging_interval, log_momentum = log_momentum)
        self.name_prefix = name_prefix

    def _find_names_from_schedulers(
        self,
        lr_scheduler_configs: List[LRSchedulerConfig],
    ) -> Tuple[List[List[str]], List[Optimizer], DefaultDict[Type[Optimizer], int]]:
        # Create unique names in the case we have multiple of the same learning
        # rate scheduler + multiple parameter groups
        names = []
        seen_optimizers: List[Optimizer] = []
        seen_optimizer_types: DefaultDict[Type[Optimizer], int] = defaultdict(int)
        for config in lr_scheduler_configs:
            sch = config.scheduler
            if sch.optimizer in seen_optimizers:
                updated_names = names[seen_optimizers.index(sch.optimizer)]
            else:
                name = self.name_prefix if self.name_prefix is not None else ''
                if config.name is not None:
                    name += config.name
                else:
                    name += "lr-" + sch.optimizer.__class__.__name__

                updated_names = self._check_duplicates_and_update_name(
                    sch.optimizer, name, seen_optimizers, seen_optimizer_types, config
                )
            names.append(updated_names)

        return names, seen_optimizers, seen_optimizer_types

    def _find_names_from_optimizers(
        self,
        optimizers: List[Any],
        seen_optimizers: List[Optimizer],
        seen_optimizer_types: DefaultDict[Type[Optimizer], int],
    ) -> Tuple[List[List[str]], List[Optimizer]]:
        names = []
        optimizers_without_scheduler = []

        for optimizer in optimizers:
            # Deepspeed optimizer wraps the native optimizer
            optimizer = optimizer.optimizer if hasattr(optimizer, "optimizer") else optimizer
            if optimizer in seen_optimizers:
                continue

            name = (self.name_prefix if self.name_prefix is not None else '') + "lr-" + optimizer.__class__.__name__
            updated_names = self._check_duplicates_and_update_name(
                optimizer, name, seen_optimizers, seen_optimizer_types, None
            )
            names.append(updated_names)
            optimizers_without_scheduler.append(optimizer)

        return names, optimizers_without_scheduler
