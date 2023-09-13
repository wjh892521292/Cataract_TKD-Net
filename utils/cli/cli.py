import os.path
import time
from types import MethodType
from typing import Any, Callable, Dict, Optional, Type, Union

from jsonargparse import ActionConfigFile
import pytorch_lightning as pl
from pytorch_lightning.cli import LightningArgumentParser, LightningCLI, SaveConfigCallback

from utils.callbacks.save_and_log_config_callback import SaveAndLogConfigCallback
from utils.optim import get_configure_optimizers_method
from .trainer import Trainer, _Trainer
from .yaml_with_merge import ArgumentParser


class CLI(LightningCLI):
    def __init__(
            self,
            save_config_callback: Optional[Type[SaveConfigCallback]] = SaveAndLogConfigCallback,
            trainer_class: Union[Type[_Trainer], Callable[..., _Trainer]] = Trainer,
            *args, **kwargs
    ) -> None:
        super().__init__(save_config_callback = save_config_callback, trainer_class = trainer_class, *args, **kwargs)

    def init_parser(self, **kwargs: Any) -> ArgumentParser:
        """Method that instantiates the argument parser."""
        kwargs.setdefault("dump_header", [f"pytorch_lightning=={pl.__version__}"])
        parser = ArgumentParser(**kwargs)
        parser.add_argument(
            "-c", "--config", action = ActionConfigFile, help = "Path to a configuration file in json or yaml format."
        )
        return parser

    def add_default_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_default_arguments_to_parser(parser)
        parser.add_argument(
            "--optimizer_config",
            type = Optional[Dict],
            default = None,
            help = "Configuration for the optimizers and lr schedulers."
        )

    def add_core_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Adds arguments from the core classes to the parser."""
        parser.add_lightning_class_args(self.trainer_class, "trainer")
        trainer_defaults = {"trainer." + k: v for k, v in self.trainer_defaults.items() if k != "callbacks"}
        parser.set_defaults(trainer_defaults)

        parser.add_lightning_class_args(self._model_class, "model", subclass_mode = self.subclass_mode_model)

        # this should not be required because the user might want to use the `LightningModule` dataloaders
        parser.add_lightning_class_args(
            self._datamodule_class, "data", subclass_mode = self.subclass_mode_data, required = False
        )

    def before_instantiate_classes(self) -> None:
        """Implement to run some code before instantiating the classes."""
        super().before_instantiate_classes()
        config = self.config['config'] if 'subcommand' not in self.config else self.config[self.config['subcommand']]
        exp_name = os.path.splitext(os.path.split(config['config'][0].abs_path)[1])[0]
        exp_id = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
        value_dict = {'name': exp_name, 'version': exp_id}
        if config.get('trainer') is not None and config['trainer'].get('logger') is not None:
            if config['trainer']['logger'].get('init_args') is None:
                config['trainer']['logger']['init_args'] = value_dict
            else:
                for k, v in value_dict.items():
                    if config['trainer']['logger']['init_args'].get(k) is None:
                        config['trainer']['logger']['init_args'][k] = v

    def _add_configure_optimizers_method_to_model(self, subcommand: Optional[str]) -> None:
        super()._add_configure_optimizers_method_to_model(subcommand)
        optimizer_config = self._get(self.config_init, 'optimizer_config')
        if optimizer_config:
            self.model.configure_optimizers = MethodType(get_configure_optimizers_method(optimizer_config), self.model)
