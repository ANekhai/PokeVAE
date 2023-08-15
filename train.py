import torch
import torch.nn as nn
from torch import TensorType
from .model import VAE

import os


class TrainParams:
    epochs: int = 200
    batch_size: int = 128
    lr: float = 0.0003
    device: str = ...

    data_root: str = "./data"
    csv_path: str = "./pokemon.csv"

    from_checkpoint: bool = False
    cp_in_path: str = ""

    to_checkpoint: bool = True
    cp_out_path: str =  ""
    
    save_animation: bool = True
    anim_per_epoch: int = 20


class AnimationHandler():
    def __init__(self, ):
        self.animation_frames: list[TensorType] = []
        self.random_batch

    def push_frame(self):
        pass

    def save_animation(self, path):
        pass



def ELBOLoss():
    pass