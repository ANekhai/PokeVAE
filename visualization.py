import torch
from torch import Tensor
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from numpy import arccos, sin

class AnimationHandler():
    ''' A class meant to remove the pain of animating CNN training '''
    def __init__(self, batch:Tensor=None, nrow=8, padding=2):
        self.animation_frames = []
        self.batch = batch
        self.nrow = nrow
        self.padding = padding

    def push_frame(self, new_frame: Tensor):
        self.animation_frames.append(new_frame.cpu())

    def animate(self, show=True, fig_size=(8,8)) -> list[Tensor]:
        # to animation
        fig = plt.figure(figsize=fig_size)
        plt.axis("off")
        
        frames = [vutils.make_grid(frame, padding=self.padding, normalize=True, nrow=self.nrow) 
                     for frame in self.animation_frames]
        
        frames = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in frames]
        anim = animation.ArtistAnimation(fig, frames, interval=1000, repeat_delay=1000, blit=True)
        if show: fig.show()
        
        return anim
    
    def save_animation(self, path:str, show=True, fps=5):
        anim = self.animate(show=show)
        anim.save(path, fps=fps)


def slerp(v1: Tensor, v2: Tensor, steps:int):
    ''' Spherical linear interpolation between two vectors '''
    omega = arccos( (v1 @ v2).item() )
    interpolation = lambda v, w, t: sin((1-t)*omega)/sin(omega) * v + sin(t*omega)/sin(omega) * w
    steps = [i * (1/steps) for i in range(steps+1)]
    return [interpolation(v1, v2, step) for step in steps] 


if __name__ == "__main__":
    # v1 = torch.tensor([1., 0., 0.]) 
    # v2 = torch.tensor([0., 0.5, 1.])

    # print(slerp(v1, v2, 3))
    test_animator = AnimationHandler(padding=0)
    for i in range(30):
        test_animator.push_frame(torch.randn(1, 3, 96, 96))

    test_animator.save_animation("test_anim.mp4", show=False)