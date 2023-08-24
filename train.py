import torch
import torch.nn as nn
from torch import optim, Tensor
from torchvision import transforms

from model import VAE, VAEParams
from dataset import PokemonDataset
from visualization import AnimationHandler

import os

# training params

epochs: int = 250
batch_size: int = 128
lr: float = 3e-4
beta1: float = 0.5
beta2: float = 0.999


device: torch.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
workers: int = 2

data_root: str = "data"
csv_path: str = "pokemon.csv"

checkpoint_dir: str = "pretrained_models"
from_checkpoint: bool = False
cp_in_path: str = ""

to_checkpoint: bool = True
cp_out_path: str =  ""

save_animation: bool = True
anim_out_path: str = "tanh_250_epochs.mp4"
epochs_per_frame: int = 10
anim_batch: int = 64


# useful functions

def kl_divergence(mu: Tensor, logvar: Tensor):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def elbo_loss(recon_loss, kl_div, beta:float=1.):
    return recon_loss + beta * kl_div


def main():
    # load checkpointed model
    vae = VAE(VAEParams).to(device)

    if from_checkpoint:
        pass

    # load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5, fill=1),
        transforms.ColorJitter(hue=0.05),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = PokemonDataset(csv_file=csv_path, root_dir=data_root, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

    # setup

    loss = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(vae.parameters(), lr=lr, betas=(beta1, beta2))

    if (save_animation):
        batch: Tensor = next(iter(loader))[:anim_batch]
        animator = AnimationHandler(batch=batch.to(device))

    # training loop
    print("Starting Training Loop")
    for epoch in range(epochs):
        loss_queue = []

        for i, data in enumerate(loader, 0):
            # forward step
            images = data.to(device)
            output_images, mus, logvars = vae(images)
            
            # calculate loss
            kl_div = kl_divergence(mus, logvars)
            recon_loss = loss(output_images, images)
            errVAE = elbo_loss(recon_loss, kl_div)

            # backprop
            vae.zero_grad()
            errVAE.backward()
            optimizer.step()

            # some bookkeeping and loss printing TODO: replace with tqdm
            loss_queue.append(errVAE.item())
            if i % 10 == 0:
                print('[%d/%d][%d/%d]\tLoss_recon: %.4f\tkl_div: %.4f'
                    % (epoch, epochs, i, len(loader),
                        recon_loss.item(), kl_div.item()))


        # save a frame of current progress every few epochs
        if save_animation and epoch%epochs_per_frame == 0 :
            with torch.no_grad(): animator.push_frame(vae(animator.batch)[0])

    # cleanup - save animation and checkpoint model
    if save_animation:
        animator.save_animation(path=anim_out_path)

    if to_checkpoint:
        pass


# required for multithreaded dataloading
if __name__ == "__main__":
    main()