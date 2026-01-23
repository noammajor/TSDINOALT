import numpy as np
import torch
import random
import torch.nn.functional as F
from torchvision import transforms
import torch.nn as nn

class polar_transformation:
    def __init__(self):
        self.warp_factor = random.uniform(0.5, 1.5)

    def __call__(self, x):
        num_patches, n_vars, size_patch = x.shape
        device = x.device
        t = torch.linspace(0, 1, steps=size_patch).to(device)
        t = t.view(1, 1, -1).expand_as(x)
        r = torch.sqrt(t**2 + x**2)
        theta = torch.atan2(x, t)
        theta = theta * self.warp_factor
        x_new = r * torch.sin(theta) 
        return x_new
class galilien_transformation:
    def __init__(self, a_range=(0.5, 1.5)):
        self.a_range = a_range

    def __call__(self, x):
        device = x.device
        a = random.uniform(*self.a_range)
        x_new = a*x
        return x_new
class rotation_transformation:
    def __init__(self, angle_range=(0, np.pi/2)):
        self.angle_range = angle_range

    def __call__(self, x):
        device = x.device
        length = x.shape[-1]
        angle = random.uniform(*self.angle_range)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        t = torch.linspace(0, 1, steps=length).to(device)
        t = t.view(1, 1, -1).expand_as(x)
        x_new = (t * sin_a) + (x * cos_a)
        return x_new
class boost_transformation:
    def __init__(self, b_range=(0.5, 1.5)):
        self.b_range = b_range

    def __call__(self, x):
        device = x.device
        length = x.shape[-1]
        b = random.uniform(*self.b_range)
        t = torch.linspace(0, 1, steps=length).to(device)
        t = t.view(1, 1, -1).expand_as(x)
        x_new = x + (b * t)
        return x_new
class lorentz_transformation:
    def __init__(self, v_range=(0.1, 0.9)):
        self.v_range = v_range

    def __call__(self, x):
        device = x.device
        length = x.shape[-1]
        v = random.uniform(*self.v_range)
        gamma = 1 / torch.sqrt(torch.tensor(1 - v**2))
        t = torch.linspace(0, 1, steps=length).to(device)
        t = t.view(1, 1, -1).expand_as(x)
        x_new = gamma * (x - v * t)    
        return x_new
class hyperbolic_amplitude_warp:
    def __init__(self, warp_range=(0.5, 1.5)):
        self.warp_range = warp_range

    def __call__(self, x):
        device = x.device
        warp_factor = random.uniform(*self.warp_range)
        x_new = torch.tanh(x * warp_factor)
        return x_new

class HyperBolicGeometry(nn.Module):
    def __init__(self, shift_magnitude=0.3, eps=1e-8):
        super().__init__()
        self.shift_magnitude = shift_magnitude
        self.eps = eps

    def to_poincare(self, x):
        length = x.shape[-1]
        device = x.device
        t = torch.linspace(-0.9, 0.9, steps=length).to(device)
        t = t.view(1, 1, -1).expand_as(x)
        y_min = x.min(dim=-1, keepdim=True)[0]
        y_max = x.max(dim=-1, keepdim=True)[0]
        y = 1.8 * (x - y_min) / (y_max - y_min + self.eps) - 0.9 
        return t, y

    def mobius_add(self, u, v, u0, v0):
        norm_z0_sq = u0**2 + v0**2
        norm_z_sq = u**2 + v**2
        inner_prod = u0*u + v0*v       
        denom = 1 + 2*inner_prod + norm_z0_sq * norm_z_sq
        num_v = (1 + 2*inner_prod + norm_z_sq) * v0 + (1 - norm_z0_sq) * v
        return num_v / (denom + self.eps)

    def __call__(self, x):
        t, y = self.to_poincare(x)
        z0 = torch.randn(2, 1, device=x.device) 
        z0 = self.shift_magnitude * z0 / (z0.norm() + self.eps)
        u0, v0 = z0[0], z0[1]
        y_new = self.mobius_add(t, y, u0, v0)  
        return y_new