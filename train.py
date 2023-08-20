import torch
import torch.nn as nn
from torch import optim

from model import VAE, VAEParams
from dataset import PokeDataset
from visualization import AnimationHandler

import os

# training params

epochs: int = 250
batch_size: int = 128
lr: float = 3e-4
beta1: float = 0.5
beta2: float = 0.999


device: torch.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

data_root: str = "./data"
csv_path: str = "./pokemon.csv"

checkpoint_dir: str = "pretrained_models"
from_checkpoint: bool = False

to_checkpoint: bool = True
cp_out_path: str =  ""

save_animation: bool = True
anim_per_epoch: int = 20


# useful functions

def elbo_loss(recon_loss, mu, logvar, beta:float=1.):
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss

# load checkpointed model

vae = VAE(VAEParams).to(device)

if from_checkpoint:
    pass

# setup

loss = nn.MSELoss(reduction='sum')
optimizer = optim.Adam(vae.parameters(), lr=lr, betas=(beta1, beta2))

if (save_animation):
    batch: torch.dtype = None
    animator = AnimationHandler(batch=batch)

# training loop

for epoch in range(epochs):
    pass

    # if (save_animation):
    #     animator.push_frame(vae(animator.batch))