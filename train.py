import argparse

from train.prepare import prepare
from train.ddpm import train_loop_ddpm

parser = argparse.ArgumentParser(description="slurm training arg parser")
parser.add_argument("--run_task", action="store", type=str)
parser.add_argument("--out", action="store", type=str, default="out")
parser.add_argument("--chec kpoint", action="store", type=int)
parser.add_argument("--n_subset", default=None, action="store", type=int)
parser.add_argument("--scheduler",  action="store", type=str, default="constant")

args = parser.parse_args()

# prepare and run train loop
config, train_dataloader, model, noise_scheduler, optimizer, lr_scheduler = prepare(
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
