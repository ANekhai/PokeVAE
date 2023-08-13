import torch
import torch.nn as nn
from .model import VAE

import os


class TrainParams:
    epochs: int = 200
    batch_size: int = 128
    lr: float = 0.0003

    data_root: str = "./data"
    csv_path: str = "./pokemon.csv"

    from_checkpoint: bool = False
    cp_in_path: str = ""

    to_checkpoint: bool = True
    cp_out_path: str =  ""
    


def ELBOLoss():
    pass