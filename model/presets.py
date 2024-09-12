

from diffusers import UNet2DModel, DDPMScheduler, AutoencoderKL


def vae_spectrogram(config):
    return AutoencoderKL(
        in_channels=1,
        out_channels=1,
        latent_channels=4,
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(32, 32, 32, 32,),  # the number of output channels for each UNet block
        down_block_types=(
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
        ),
        up_block_types=(
            "UpDecoderBlock2D",  # a regular ResNet upsampling block
            "UpDecoderBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
        ),
    )

"""
OLD
"""

def unet_ddpm_cifar10(config):
    return UNet2DModel(
        sample_size=config.image_size,  # the target image resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(128, 256, 256, 256,),  # the number of output channels for each UNet block
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
