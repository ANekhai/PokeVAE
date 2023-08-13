import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset


class PokeDataset(Dataset):

    def __init__(self, csv, root_dir="./data", transform=None):
        self.df = pd.read_csv(csv).fillna("None")
        self.data_root = root_dir


    def __length__(self):
        return len(self.df)
    

    def __getitem__(self, idx):
        pass