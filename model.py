import torch
import torch.nn as nn
from torch import Tensor


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
            # (b, nef*16, 1, 1)
        )


    def forward(self, im):
        return self.model(im)


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        assert config.im_c is not None
        assert config.ndf is not None
        ndf = config.ndf

        self.model = nn.Sequential(
            # input: (b, ndf*16, 1, 1)
            TransConvBlock(ndf*16, ndf*8, 6, 1, 0),
            # (b, ndf*8, 6, 6)
            TransConvBlock(ndf*8, ndf*4, 4, 2, 1),
            # (b, ndf*4, 12, 12)
            TransConvBlock(ndf*4, ndf*2, 4, 2, 1),
            # (b, ndf*2, 24, 24)
            TransConvBlock(ndf*2, ndf, 4, 2, 1),
            # (b, ndf, 48, 48)
            #TODO: Change output block to use softmax + colorspace
            nn.ConvTranspose2d(ndf, config.im_c, 4, 2, 1),
            nn.Tanh()
            # (b, 3, 96, 96)
        )


    def forward(self, z):
        return self.model(z)


class ColorBlock(nn.Module):
    def __init__(self):
        super(ColorBlock, self).__init__()

    def forward(self, z):
        pass


class VAE(nn.Module):
    def __init__(self, config: VAEParams):
        super(VAE, self).__init__()

        assert config.z_dim is not None
        self.z_dim = config.z_dim

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.to_mu = nn.Conv2d(config.nef*16, config.z_dim, 1, 1, 0)
        self.to_logvar = nn.Conv2d(config.nef*16, config.z_dim, 1, 1, 0)
        self.from_latent = nn.Conv2d(config.z_dim, config.ndf*16, 1, 1, 0)

    def forward(self, im):
        features = self.encoder(im)
        mu, logvar = self.to_mu(features), self.to_logvar(features)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


    def reparameterize(self, mu, log_sigma):
        sigma = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(sigma)
        return mu + (eps * sigma)
        
    def decode(self, z):
        return self.decoder(self.from_latent(z))

    @torch.no_grad()
    def generate(self, mu: Tensor, logvar: Tensor):
        z = self.reparameterize(mu, logvar)
        return self.decode(z)
        


if __name__ == "__main__":
    test_block = VAE(VAEParams)

    test_tensor = torch.randn(4, 3, 96, 96)
    out_tensor, _, _ = test_block(test_tensor)
    print(out_tensor.shape)