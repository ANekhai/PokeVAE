import os
import pandas as pd

import torch
from torch.utils.data import Dataset
# from torchvision.io import read_image
from PIL import Image

class PokemonDataset(Dataset):

    def __init__(self, csv_file:str, root_dir:str, transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.df.iloc[idx, 0])
        # image = read_image(img_path).type(torch.float32)
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image
