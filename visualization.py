import torch
from torch import Tensor
import torchvision.utils as vutils

from numpy import arccos, sin

class AnimationHandler():
    def __init__(self, batch:Tensor=None):
        self.animation_frames = []
        self.batch = batch

    def push_frame(self, new_frame: Tensor):
        pass

    def save_animation(self, path:str, show=True):
        pass

def slerp(v1: Tensor, v2: Tensor, steps:int):
    ''' Spherical linear interpolation between two vectors '''
    omega = arccos( (v1 @ v2).item() )
    interpolation = lambda v, w, t: sin((1-t)*omega)/sin(omega) * v + sin(t*omega)/sin(omega) * w
    steps = [i * (1/steps) for i in range(steps+1)]
    return [interpolation(v1, v2, step) for step in steps] 


if __name__ == "__main__":
    v1 = torch.tensor([1., 0., 0.])
    v2 = torch.tensor([0., 0.5, 1.])

    print(slerp(v1, v2, 3))