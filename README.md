## Installation

- install [pytorch](https://pytorch.org/get-started/locally/), torchvision etc.
  and [mmcv-full](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) from their official site.
- install required packages with `pip install -r requirements.txt`
- install this project with `pip install -e .`

For the versions of packages, you can check `freeze.yml`.

## Usage

This project base on the CLI of pytorch-lightning, you can get more information
on [here](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_cli.html).

### Config and config files ###

#### Predefined config files ####

The predefined config files are located at `configs/`.

Every config file under `configs/runs` is a complete config file, so you can run an experiment with just a config file
from `configs/runs`. But every other config file is just a part of complete config file, you should use them by
combining them with each other or write a complete config file under `configs/runs` using `__base__` to inherit from
them.

#### Syntax ####

The syntax of config file is [yaml](https://yaml.readthedocs.io/en/latest/) and some additional keyword described as
follow.

| keyword       | value                                                                                                                                                      | effect                                                                                                 |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `__base__`    | `str` or `list[str]`,(each `str` should be a relative path from current cofig file)                                                                        | Merge every config one by one, current last.                                                           |
| `__delete__`  | `True` or `str,int` or `list[str,int]`,`True` for delete all keys from other config, `str,int` only delete the specific key (for dict) or index (for list) | Delete some part of config from other.                                                                 |
| `__import__`  | Any                                                                                                                                                        | Just delete this, for convenience of reference in yaml                                                 |
| `change_item` | `list[[index, item]]`,used only when merge list                                                                                                            | Add ability of merg list, change the `list[index]` from other to `item`                                |
| `insert_item` | `list[[index, item, (extend)]]`,used only when merge list                                                                                                  | Add ability of merg list, insert iterm to the `list` at `index`, extend=True if insert a list of items |
| `pre_item`    | `Any`or `list[Any]`,used only when merge list                                                                                                              | Add ability of merg list, add the value in the start of the list from other to item                    |
| `post_item`   | `Any`or `list[Any]`,used only when merge list                                                                                                              | Add ability of merg list, add the value in the end of the list from other to item                      |

### CLI ###

CLI script is located at `tools/cli.py`, so you can run it with `python tools/cli.py`.

For commands and options, you can get all available options and commands from `python tools/cli.py --help`.

#### Train ####

You can start a experiment with command as follow, in which, `gpu_ids` is comma-separated id list or just one int.

```bash
CUDA_VISIBLE_DEVICES=<gpu_ids> python tools/cli.py fit --config configs/runs/path/to/config
```

#### Validation test and predict etc. ####

```bash
CUDA_VISIBLE_DEVICES=<gpu_ids> python tools/cli.py {validation, test, predict, tune} --config configs/runs/path/to/config
```

## Experiments

See `configs/runs/image_classifier/`. You can run them with `python tools/cli.py fit --config configs/runs/image_classifier/path/to/config`.

Run teacher model:

```bash
CUDA_VISIBLE_DEVICES=<gpu_ids> python tools/cli.py fit --config configs/runs/image_classifier/resnet50_transformer_with_label_cataract
```

Run student model:

First, modify the checkpoint in `configs/runs/image_classifier/resnet50_transformer_distillation_label_cataract.yaml` to the ckpt path of teacher model.

Then, run:
```bash
CUDA_VISIBLE_DEVICES=<gpu_ids> python tools/cli.py fit --config configs/runs/image_classifier/resnet50_transformer_distillation_label_cataract.yaml
```

