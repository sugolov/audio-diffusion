import os
import torch

from diffusers.optimization import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from datasets import load_dataset
from torchvision import transforms


from train.config import VAESpectrogramUNetTrainingConfig

"""
Will be useful in the future!
"""


class RunHandler:
    """
    Directory and config handling for methods that show up in training loops

    TODO: handle run naming better
    """

    def __init__(self, output_dir, data_dir, checkpoint_step, scheduler, config, run_name=None):
        self.output_dir = output_dir
        self.data_dir = data_dir
        self.checkpoint_step = checkpoint_step
        self.scheduler = scheduler
        self.config = config

        if run_name is not None:
            self.config.run_name = run_name

    def prepare(self):
        pass

    def get_dataloader(self, column="image"):
        dataset = load_dataset("imagefolder", data_dir=self.data_dir, split="train")

        # TODO: REMOVE OUTSIDE OF TEST MODE
        dataset = dataset.select(range(16))
        preprocess = transforms.Compose(self.config.transforms)

        def transform(examples):
            images = [preprocess(image.convert("RGB")) for image in examples["image"]]
            return {"images": images}

        dataset.set_transform(transform)

        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        return train_dataloader

    def get_global_step(self):
        return self.checkpoint_step if self.checkpoint_step else 0

    def get_output_path(self):
        """
        :param step: step to format for
        :return: current checkpoint, if no step, otherwise format with step
        """
        return os.path.join(self.output_dir, self.config.run_name)

    def get_checkpoint_path(self, step=None):
        """
        :param step: step to format for
        :return: current checkpoint, if no step, otherwise format with step
        """
        step = step if step is not None else self.checkpoint_step
        if step is None:
            raise ValueError("step not passed")
        return os.path.join(self.get_output_path(), f"step_{step}")


    def get_scheduler(self, optimizer, global_train_steps=None):
        if self.scheduler == "constant":
            lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=self.config.lr_warmup_steps
            )
        elif self.scheduler == "cosine":
            if global_train_steps is not None:
                lr_scheduler = get_cosine_schedule_with_warmup(
                    optimizer=optimizer,
                    num_warmup_steps=self.config.lr_warmup_steps,
                    num_training_steps=global_train_steps,
                )
            else:
                raise Exception("Global steps must be specified for cosine scheduler.")

        return lr_scheduler

    def _get_global_steps(self, n_data):
        return int(n_data / self.config.batch_size * self.config.num_global_epochs)


class VAERunHandler(RunHandler):
    def __init__(self, output_dir, data_dir, checkpoint_step, scheduler, run_name=None, *args, **kwargs):
        config = VAESpectrogramUNetTrainingConfig()

        super().__init__(output_dir, data_dir, checkpoint_step, scheduler, config, run_name)

    def prepare(self):
        from model.presets import vae_spectrogram

        train_dataloader = self.get_dataloader(self.config.transforms)
        model = vae_spectrogram(self.config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.lr)

        global_train_steps = self._get_global_steps(len(train_dataloader))
        lr_scheduler = self.get_scheduler(optimizer, global_train_steps)

        return self.config, train_dataloader, model, optimizer, lr_scheduler
