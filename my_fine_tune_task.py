import logging
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from classy_vision.dataset import build_dataset
from classy_vision.generic.registry_utils import import_all_packages_from_directory
from classy_vision.generic.util import load_checkpoint, load_json
from classy_vision.hooks import (
    CheckpointHook,
    LossLrMeterLoggingHook,
    ProfilerHook,
    ProgressBarHook,
    TensorboardPlotHook,
)
from classy_vision.losses import build_loss
from classy_vision.meters import build_meters
from classy_vision.models import ClassyModel
from classy_vision.optim import build_optimizer
from classy_vision.tasks import FineTuningTask
from classy_vision.trainer import LocalTrainer

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class My_FineTuningTask(FineTuningTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def eval_step(self):
        super().eval_step()
        print("test_epochs:" + str(self.eval_phase_idx))

    def train_step(self):
        super().train_step()
        print("train_epochs:" + str(self.train_phase_idx))

    def on_end(self):
        super().on_end()
        print("loss:" + str(self.losses[-1]) + ",acc:" + str(self.meters[0].value))


file_root = Path(__file__).parent
import_all_packages_from_directory(str(file_root))

config_file = './configs/my_config.json'
config = load_json(config_file)

task = (My_FineTuningTask()
        .set_loss(build_loss(config["loss"]))
        # .set_model(build_model(config["model"]))
        .set_optimizer(build_optimizer(config["optimizer"]))
        .set_meters(build_meters(config.get("meters", {})))
        .set_reset_heads(config["reset_heads"])
        .set_freeze_trunk(config["freeze_trunk"]))
# dataset
for split in ["train", "test"]:
    dataset = build_dataset(config["dataset"][split])
    dataset.set_num_workers(config["dataset"]["num_workers"])
    task.set_dataset(dataset, split)
# model
num_classes = 2
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
classy_model = ClassyModel().from_model(model)
task.set_model(classy_model)


# model = models.resnet50(pretrained=False)
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, num_classes)
# classy_model = ClassyModel().from_model(model)  # wrapper
# task.set_model(classy_model)
# pretrained_checkpoint_path = config["pretrained_checkpoint"]
# pretrained_checkpoint = load_checkpoint(pretrained_checkpoint_path)
# if pretrained_checkpoint is not None:
#     assert isinstance(
#         task, FineTuningTask
#     ), "Can only use a pretrained checkpoint for fine tuning tasks"
#     task.set_pretrained_checkpoint(pretrained_checkpoint_path)


# hooks
def configure_hooks(log_freq=10, checkpoint_folder="", skip_tensorboard=True,
                    checkpoint_period=10, profiler=False, show_progress=True):
    # hooks = [LossLrMeterLoggingHook(log_freq), ModelComplexityHook()]
    hooks = [LossLrMeterLoggingHook(log_freq)]
    # Make a folder to store checkpoints and tensorboard logging outputs
    suffix = datetime.now().isoformat()
    index = suffix.find(':')
    base_folder = f"{Path(__file__).parent}/output_{suffix[0:index]}"
    if checkpoint_folder == "":
        checkpoint_folder = base_folder + "/checkpoints"
        os.makedirs(checkpoint_folder, exist_ok=True)

    logging.info(f"Logging outputs to {base_folder}")
    logging.info(f"Logging checkpoints to {checkpoint_folder}")

    if not skip_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_writer = SummaryWriter(log_dir=Path(base_folder) / "tensorboard")
            hooks.append(TensorboardPlotHook(tb_writer))
        except ImportError:
            logging.warning("tensorboard not installed, skipping tensorboard hooks")

    args_dict = None
    hooks.append(
        CheckpointHook(
            checkpoint_folder, args_dict, checkpoint_period=checkpoint_period
        )
    )
    if profiler:
        hooks.append(ProfilerHook())
    if show_progress:
        hooks.append(ProgressBarHook())
    return hooks


checkpoint_dir = "./checkpoints"
task.set_hooks(configure_hooks(checkpoint_folder=checkpoint_dir))
task.set_num_epochs(config["num_epochs"])
task.set_use_gpu(torch.cuda.is_available())

if __name__ == "__main__":
    trainer = LocalTrainer()
    trainer.train(task)
