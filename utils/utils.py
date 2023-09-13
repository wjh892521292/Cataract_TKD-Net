import os
import sys


def get_log_dir(trainer):
    if trainer.checkpoint_callback is not None and trainer.checkpoint_callback.dirpath is not None:
        log_dir = os.path.dirname(trainer.checkpoint_callback.dirpath)
    elif len(trainer.loggers) > 0:
        if trainer.loggers[0].save_dir is not None:
            save_dir = trainer.loggers[0].save_dir
        else:
            save_dir = trainer.default_root_dir
        name = trainer.loggers[0].name
        version = trainer.loggers[0].version
        version = version if isinstance(version, str) else f"version_{version}"
        log_dir = os.path.join(save_dir, str(name), version)
    else:
        log_dir = trainer.default_root_dir
    return log_dir


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
