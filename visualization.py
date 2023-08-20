import torch
import torchvision.utils as vutils


class AnimationHandler():
    def __init__(self, batch:torch.dtype=None):
        self.animation_frames: list[torch.dtype] = []
        self.batch: torch.dtype = batch

    def push_frame(self, new_frame:torch.dtype):
        pass

    def save_animation(self, path:str, show=True):
        pass
