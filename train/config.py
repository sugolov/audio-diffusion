
import dataclasses
from dataclasses import dataclass
from torchvision import transforms

@dataclass
class VAESpectrogramUNetTrainingConfig:
    image_size = 32  # the generated image resolution
    batch_size = 64

    num_epochs = 200
    num_global_epochs = 2000 # total passes over dataset
    gradient_accumulation_steps = 1

    lr = 2e-4
    lr_warmup_steps = 500

    save_model_epochs = 100
    save_by_epoch = True
    save_checkpoint_epochs = 10

    mixed_precision = "fp16"
    run_name = "vae-spectrogram"

    transforms = [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]

    push_to_hub = False
    #hub_model_id = f"sugolov/{run_name}"
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0