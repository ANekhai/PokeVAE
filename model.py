import torch
import torch.nn as nn
from torch import TensorType


class VAEParams:
    im_size: int = 96
    im_c: int = 3
    nef: int = 64       
    ndf: int = 64
    z_dim: int = 8
    palette_dim: int = 16


class Encoder(nn.Module):
    ''' Compress an image into a latent representation  '''
    def __init__(self):
        super(Encoder, self).__init__()

    def forward(self, im):
        pass


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self, z):
        pass


class ColorBlock(nn.Module):
    def __init__(self):
        super(ColorBlock, self).__init__()

    def forward(self, z):
        pass


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

    def forward(self, im):
        pass