import os
from typing import Dict, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.types import _METRIC


class ModelCheckpointWithLinkBest(ModelCheckpoint):
    CHECKPOINT_NAME_BEST = "best"

    def __init__(
            self,
            save_best: Optional[bool] = None,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.save_best = save_best

    def _update_best_and_save(
            self, current: torch.Tensor, trainer: "pl.Trainer", monitor_candidates: Dict[str, _METRIC]
    ) -> None:
        super()._update_best_and_save(current, trainer, monitor_candidates)
        self._save_best_checkpoint(trainer, monitor_candidates)

    def _save_best_checkpoint(self, trainer: "pl.Trainer", monitor_candidates: Dict[str, _METRIC]) -> None:
        if not self.save_best:
            return

        filepath = self.format_checkpoint_name(monitor_candidates, self.CHECKPOINT_NAME_BEST)

        if trainer.is_global_zero:
            if self._fs.lexists(filepath):
                self._fs.rm_file(filepath)
            if self._fs.protocol == 'file':
                os.symlink(os.path.basename(self.best_model_path), filepath)
            else:
                self._fs.cp_file(self.best_model_path, filepath)
