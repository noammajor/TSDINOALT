import torch.nn.functional as F
import torch.nn as nn
import torch
import os
import sys
import time
import math
import random
import datetime
import subprocess
from collections import defaultdict, deque

import numpy as np
import torch.distributed as dist

def get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "glu":
        return F.glu
    elif activation == "tanh":
        return torch.tanh
    elif activation == "sigmoid":
        return torch.sigmoid
    elif activation == "leakyrelu":
        return F.leaky_relu
    else:
        raise RuntimeError("activation should be relu/gelu/glu/tanh/sigmoid/leakyrelu, not {}".format(activation))


def init_distributed_mode(args):
    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    # launched naively with `python main_dino.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.barrier()
    setup_for_distributed(args.rank == 0)

def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class TSMultiCropWrapper(nn.Module):
    def __init__(self, backbone, head):
        super(TSMultiCropWrapper, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        """
        x: List of tensors. 
           Each tensor is [bs x num_patch x n_vars x patch_len]
        """
        if not isinstance(x, list):
            x = [x]

        # 1. Group inputs by their temporal shape (num_patch and patch_len)
        # We look at the shape of the patches to decide what can be batched together
        unique_shapes = []
        for inp in x:
            # We use num_patch (dim 1) and patch_len (dim 3) as the 'resolution' key
            shape_key = (inp.shape[1], inp.shape[3])
            if shape_key not in unique_shapes:
                unique_shapes.append(shape_key)

        # 2. Logic to find indices where the shapes change
        # This mirrors the logic in the DINO code but for TS dimensions
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[1] for inp in x]), # Based on num_patches
            return_counts=True,
        )[1], 0)

        start_idx, output = 0, torch.empty(0).to(x[0].device)

        # 3. Forward pass through backbone for each group
        for end_idx in idx_crops:
            # Concatenate all views of the same size along the batch dimension
            # Result: [bs * num_views_in_group, num_patch, n_vars, patch_len]
            clubbed_inputs = torch.cat(x[start_idx: end_idx], dim=0)
            
            _out = self.backbone(clubbed_inputs)
            
            # PatchTST usually returns the [CLS] token or a flattened representation
            # If it's a tuple (like some Transformers), take the first element
            if isinstance(_out, tuple):
                _out = _out[0]
                
            output = torch.cat((output, _out))
            start_idx = end_idx

        # 4. Run the projection head on ALL features from ALL views
        return self.head(output)

def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False

def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]
def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule
    
def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)