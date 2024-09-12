import argparse

from train.loops import train_loop_ddpm, train_loop_vae

parser = argparse.ArgumentParser(description="Training arg parser")

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
vargs = vars(args)

from train.handler import VAERunHandler

def start_message():
    print("Starting training with")
    [print(f"{arg}: {vargs[arg]}") for arg in vargs]
    print("\n")

if args.vae:
    start_message()
    handler = VAERunHandler(**vargs)
    train_loop_vae(handler, *handler.prepare())

# TODO: cleanup
elif args.ddpm:
    pass