import torch
from torchvision import datasets, transforms
from diffusers import DDPMScheduler

from datasets import load_dataset
from diffusers.optimization import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup

from model.config import *
from train.config import *

"""
Prepare objects for training loop
"""
def prepare(task, n_subset=None, output_dir="out", scheduler="linear"):
    # match correct run names
    if task == "butterfly":
        config = SmithsonianButterflyTrainingConfig()
        model = unet_ddpm_smithsonian_butterfly(config)
        data_key = "image"
    elif task == "cifar10":
        config = CIFAR10TrainingConfig()
        model = unet_ddpm_cifar10(config)
        data_key = "img"
    else:
        raise ValueError(f"No config for `{task}`")

    # subset training data if needed
    config.n_subset = n_subset
    config.output_dir = output_dir

    # transform functions
    preprocess = transforms.Compose(config.transforms)

    def transform(examples):
        images = [preprocess(image.convert("RGB")) for image in examples[data_key]]
        return {"images": images}

    # load data
    print(config.dataset_name)
    dataset = load_dataset(config.dataset_name, split="train")

    # if subset size given, subset the dataset
    if config.n_subset is not None:
        dataset = dataset.shuffle(seed=config.seed).select(range(config.n_subset))
        config.run_name += f"-{config.n_subset}"
        config.hub_model_id += f"-{config.n_subset}"


    dataset.set_transform(transform)
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=True
    )
    # create ddpm noise schedule
    noise_scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps)

    # optimization
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    if scheduler == "constant":
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config.lr_warmup_steps
        )
    elif scheduler == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * config.num_epochs),
        )

    return config, train_dataloader, model, noise_scheduler, optimizer, lr_scheduler
