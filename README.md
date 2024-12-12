# audio-diffusion

`train.py` runs training with arguments
```
train.py [-h] [--vae] [--ddpm] [--unet] [--output_dir OUTPUT_DIR] [--data_dir DATA_DIR] [--run_name RUN_NAME] [--checkpoint_step CHECKPOINT_STEP] [--scheduler SCHEDULER]
```

## Train the VAE

```commandline
python train.py --vae --run_name --data_dir <DATA_DIR> --output_dir <OUTPUT_DIR> --scheduler constant
```

latent audio diffusion with a transformer backbone
