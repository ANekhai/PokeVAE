import torch
import torch.nn as nn
from torch import TensorType


class VAEParams:
    im_size: int = 96
    im_c: int = 3
    nef: int = 64       # number of features for encoder       
    ndf: int = 64       # number of features for decoder
    z_dim: int = 8
    palette_dim: int = 16
    

class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int, s: int, p: int, slope=0.2):
        super(ConvBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, p),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(slope, inplace=True)
        )
    
    def forward(self, input):
        return self.model(input)


class TransConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int, s: int, p: int, slope=0.2):
        super(TransConvBlock, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, k, s, p),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(slope, inplace=True)
        )

    def forward(self, input):
        return self.model(input)


class Encoder(nn.Module):
    ''' Compress an image into a latent representation  '''
    def __init__(self, config):
        super(Encoder, self).__init__()

        assert config.im_c is not None
        assert config.nef is not None
        nef = config.nef

        self.model = nn.Sequential(
            # input: (b, 3, 96, 96)
            ConvBlock(config.im_c, nef, 4, 2, 1),
            # (b, nef, 48, 48)
            ConvBlock(nef, nef*2, 4, 2, 1),
            # (b, nef*2, 24, 24)
            ConvBlock(nef*2, nef*4, 4, 2, 1),
            # (b, nef*4, 12, 12)
            ConvBlock(nef*4, nef*8, 4, 2, 1),
            # (b, nef*8, 6, 6)
            ConvBlock(nef*8, nef*16, 6, 1, 0)
        )


    def forward(self, im):
        return self.model(im)


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

    def generate(self):
        pass

    def interpolate(self):
        pass


if __name__ == "__main__":
    pass


if __name__ == "__main__":
    test_block = Encoder(VAEParams)

    test_tensor = torch.randn(4, 3, 96, 96)
    out_tensor = test_block(test_tensor)
    print(out_tensor.shape)