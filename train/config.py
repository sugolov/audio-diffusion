
import dataclasses
from dataclasses import dataclass
from torchvision import transforms

@dataclass
class SmithsonianButterflyTrainingConfig:
    image_size = 128  # the generated image resolution
    train_batch_size = 32
    #eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "out/ddpm-butterfly-unet"  # the model name locally and on the HF Hub
    run_name = "ddpm-butterfly"
    dataset_name = "uoft-cs/cifar10"

    num_train_timesteps = 1000

    n_subset = None
    transforms = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_model_id = "sugolov/ddpm-smithsonian-butterfly-256-unet" if n_subset is None else f"sugolov/ddpm-smithsonian-butterfly-256-unet-{n_subset}"
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0

@dataclass
class CIFAR10TrainingConfig:
    image_size = 32  # the generated image resolution
    train_batch_size = 64
    #eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 200
    gradient_accumulation_steps = 1
    learning_rate = 2e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 100
    save_by_epoch = True

    save_checkpoint_epochs = 10

    mixed_precision = "fp16"
    run_name = "ddpm-cifar10"
    output_dir = f"out/{run_name}"
    dataset_name = "uoft-cs/cifar10"

    num_train_timesteps = 1000

    n_subset = None
    transforms = [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]

    push_to_hub = False
    hub_model_id = f"sugolov/{run_name}"
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0