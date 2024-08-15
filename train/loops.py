from diffusers import DDPMPipeline
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch
import wandb

#from utils.utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
import os

import torch.nn.functional as F


def train_loop_vae(handler, config, train_dataloader, model, optimizer, lr_scheduler):
    # Initialize accelerator and logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="wandb",
        project_dir=handler.output_dir
    )

    if accelerator.is_main_process:
        if handler.output_dir is not None:
            os.makedirs(handler.output_dir, exist_ok=True)
        if config.push_to_hub:
            repo_id = create_repo(repo_id=config.hub_model_id or Path(handler.output_dir).name, exist_ok=True).repo_id
        accelerator.init_trackers(
            project_name="audio-diffusion",
            init_kwargs={"wandb": {"name": config.run_name}},
            config=config
        )

    # Prepare everything
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    accelerator.register_for_checkpointing(optimizer, lr_scheduler, model)

    # if checkpoint step is passed, load it
    global_step = handler.get_global_step()
    if handler.checkpoint_step is not None:
        accelerator.load_state(handler.get_checkpoint_path())

    # set parameters

    # training loop
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            images = batch["images"]

            with accelerator.accumulate(model):
                # kl div
                q = model.encode(images).latent_dist
                loss_kl = q.kl().mean()

                # recopnstruction loss
                images_pred = model(images).sample
                loss_recon = F.mse_loss(images_pred, images)

                loss = loss_recon + loss_kl

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
                "epoch": epoch
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        if accelerator.is_main_process:

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:

                out_path = os.path.join(handler.output_dir, config.run_name)
                model.save_pretrained(out_path)

                if config.push_to_hub:
                    upload_folder(
                        repo_id=config.hub_model_id,
                        folder_path=out_path,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )

            if (epoch + 1) % config.save_checkpoint_epochs:
                checkpoint_path = handler.get_checkpoint_path(global_step)
                accelerator.save_state(checkpoint_path)
                print(f"Created checkpoint at step {global_step} in " + checkpoint_path)

    accelerator.end_training()


# TODO: adapt this to the correct training pipeline
# TODO: clean up train loops without overengineering using classes (?)

def train_loop_ddpm(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, checkpoint_step=None):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="wandb",
        # project_dir=os.path.join(config.output_dir, "logs"),
        project_dir=config.output_dir
    )

    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        if config.push_to_hub:
            repo_id = create_repo(
                repo_id=config.hub_model_id or Path(config.output_dir).name, exist_ok=True
            ).repo_id
        accelerator.init_trackers(
            project_name="diffusion",
            init_kwargs={"wandb": {"name": config.run_name}},
            config=config
        )

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    accelerator.register_for_checkpointing(optimizer, lr_scheduler, model)

    # if checkpoint step is passed, load it
    if checkpoint_step is None:
        global_step = 0
    else:
        accelerator.load_state(get_checkpoint_path(config, checkpoint_step))
        global_step = checkpoint_step

    # training loop
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"]
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]

            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                dtype=torch.int64
            )

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
                "epoch": epoch
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        if accelerator.is_main_process:

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:

                # tag = f"-epoch-{epoch + 1}" if (epoch + 1) < config.num_epochs and config.save_by_epoch else ""
                pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

                out_path = os.path.join(
                    config.output_dir,
                    config.run_name
                )

                pipeline.save_pretrained(out_path)

                if config.push_to_hub:
                    upload_folder(
                        repo_id=config.hub_model_id,
                        folder_path=out_path,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )

            if (epoch + 1) % config.save_checkpoint_epochs:
                # turn off for this epoch
                # save as end of previous global step
                checkpoint_path = get_checkpoint_path(config, global_step)
                accelerator.save_state(checkpoint_path)
                print(f"Created checkpoint at step {global_step} in " + checkpoint_path)
    accelerator.end_training()
