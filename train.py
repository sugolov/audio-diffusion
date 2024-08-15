import argparse

from train.prepare import prepare_ddpm
from train.loops import train_loop_ddpm, train_loop_vae

parser = argparse.ArgumentParser(description="slurm training arg parser")

# specify training of either vae or ddpm
parser.add_argument("--vae", action="store_true", default="False")
parser.add_argument("--ddpm", action="store_true", default="False")

parser.add_argument("--unet", action="store_true", default="False")

parser.add_argument("--output_dir", action="store", type=str, default="out")
parser.add_argument("--data_dir", action="store", type=str, default="dataset")

parser.add_argument("--run_name", action="store", type=str, default=None)

# what checkpoint step to load
parser.add_argument("--checkpoint_step", action="store", type=int)

# cosine or linear scheduler
## TODO: test global scheduler aligns with training iters
parser.add_argument("--scheduler",  action="store", type=str, default="constant")

args = parser.parse_args()

from train.handler import VAERunHandler

if args.vae:
    print(vars(args))
    handler = VAERunHandler(**vars(args))
    train_loop_vae(handler, *handler.prepare())

# TODO: cleanup
elif args.ddpm:
    # prepare and run train loop
    config, train_dataloader, model, noise_scheduler, optimizer, lr_scheduler = prepare_ddpm(
        task=args.run_task,
        n_subset=args.n_subset,
        output_dir=args.out,
        scheduler=args.scheduler
    )

    train_loop_ddpm(
        config=config,
        train_dataloader=train_dataloader,
        model=model,
        noise_scheduler=noise_scheduler,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        checkpoint_step=args.checkpoint
    )
