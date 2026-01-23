import numpy as np
import pandas as pd
import os
import torch
from torch import nn
import random
import torch.nn.functional as F
import torchvision.transforms as transforms
from models.patchTST import PatchTST
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import time
import datetime
import math
import sys
from pathlib import Path
import json
import data_agumentation as aug
from models.layers.Dino_Head import DINOHead
import utils.util as utils
import dataPuller as dpuller
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Transformer:
    parser.add_argument('--target_points', type=int, default=96, help='forecast horizon')
    parser.add_argument('--patch_len', type=int, default=12, help='patch length')
    parser.add_argument('--n_layers', type=int, default=3, help='number of Transformer layers')
    parser.add_argument('--n_heads', type=int, default=16, help='number of Transformer heads')
    parser.add_argument('--embed_dim', type=int, default=128, help='Transformer d_model')
    parser.add_argument('--d_ff', type=int, default=512, help='Tranformer MLP dimension')
    parser.add_argument('--dropout', type=float, default=0.2, help='Transformer dropout')
    parser.add_argument('--head_dropout', type=float, default=0.2, help='head dropout')
    parser.add_argument('--c_in', type=int, default=7, help='number of input variables')
    #dino
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")
    parser.add_argument('--out_dim', default=35536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')
    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=False, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--num_patches', default=32, type=int,help='Batch size')
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")
    #data
    parser.add_argument('--stride', type=int, default=12, help='stride between patch')
    # Misc
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=10, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--local_crops_number', type=float, default=0.5, help='size of local crops.')
    parser.add_argument('--transformation_group_size', type=int, default=2, help='Number of transformations per sample.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")


    args = parser.parse_args()
    print('args:', args)
    num_patch = args.num_patches
    print('number of patches:', num_patch)
    return parser

def train_TS_DINO(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    #print("git:\n  {}\n".format(utils.get_sha()))
    #print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    #cudnn.benchmark = True

    #-------------DATA-----------------
    dataAugmentationDino=DataAugmentationDino(
        local_crops_number=args.local_crops_number,
        transformation_group_size=args.transformation_group_size)
    dataset= dpuller.DataPuller(
        data_dir=args.data_path,
        split='train',
        transform=dataAugmentationDino,
        batch_size=args.num_patches,
        patch_size=args.patch_len
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True),
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )


    #------------- Student - Teacher network ---------------

    student = PatchTST(
        c_in= args.c_in,
        target_dim=args.target_points,
        patch_len=args.patch_len,
        stride=args.stride,
        num_patch=args.num_patches,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.embed_dim,
        shared_embedding=True,
        d_ff=args.d_ff,                        
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        act='relu',
        head_type='Dino',
        res_attention=False,
        drop_path_rate=args.drop_path_rate
        )
    teacher = PatchTST(
        c_in= args.c_in,
        target_dim=args.target_points,
        patch_len=args.patch_len,
        stride=args.stride,
        num_patch=args.num_patches,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.embed_dim,
        shared_embedding=True,
        d_ff=args.d_ff,                        
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        act='relu',
        head_type='Dino',
        res_attention=False,
        drop_path_rate=0.0
        )
    embed_dim = student.backbone.d_model
    student = utils.TSMultiCropWrapper(student, DINOHead(
        embed_dim*args.c_in,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    teacher = utils.TSMultiCropWrapper(
        teacher,
        DINOHead(embed_dim*args.c_in, args.out_dim, args.use_bn_in_head, norm_last_layer=args.norm_last_layer),
    )
    # move networks to gpu
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    student, teacher = student.to(device), teacher.to(device)
    # synchronize batch norms (if any)
    device_ids = [args.gpu] if torch.cuda.is_available() else None
    if utils.has_batchnorms(student):
        # Around line 190 in train_TS_DINO
        if torch.cuda.is_available():
            student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
            teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
        else:
            print("Skipping SyncBatchNorm on Mac/CPU—using standard BatchNorm instead.")
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=device_ids)
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=device_ids)
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both TS networks.")

#-----------------Loss function --------------------
    dino_loss = DINOLoss(
        args.out_dim,
        args.transformation_group_size+3,
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).to(device)
# ----------------Optimizer --------------------
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
       # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()
#-----------------Scheduler --------------------
    lr_schedule = utils.cosine_scheduler(
        args.lr* (args.batch_size_per_gpu * utils.get_world_size()) / 256.,
        args.min_lr,
        args.epochs,
        len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs,
        len(data_loader),
    )
    momentum_schedule = utils.cosine_scheduler(
        args.momentum_teacher,
        1,
        args.epochs,
        len(data_loader),
    )
#----------------Train Loop --------------------
    #to_restore = {"epoch": 0}
   #utils.restart_from_checkpoint(
    #    os.path.join(args.output_dir, "checkpoint.pth"),
    #    run_variables=to_restore,
    #    student=student,
     #   teacher=teacher,
      #  optimizer=optimizer,
       # fp16_scaler=fp16_scaler,
      #  dino_loss=dino_loss,
    #)
    #start_epoch = to_restore["epoch"]
    start_epoch = 0

    start_time = time.time()
    print("Starting TS - DINO training !")

    for epoch in range(start_epoch, args.epochs):
        print(f'Starting epoch {epoch}/{args.epochs}')
        data_loader.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            student,
            teacher,
            teacher_without_ddp,
            dino_loss,
            data_loader,
            optimizer,
            epoch,
            fp16_scaler,
            lr_schedule,
            wd_schedule,
            momentum_schedule,
            args
        )
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader, optimizer, epoch, fp16_scaler, lr_schedule, wd_schedule, momentum_schedule, args):
    student.train()
    teacher.train()  # teacher is in eval mode but we need to keep track of BN stats
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('weight_decay', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    for it, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        samples = batch
        samples = [s.to(device, non_blocking=True) for s in samples]
        # update learning rate and weight decay according to their schedule
        it_global = it + epoch * len(data_loader)
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr_schedule[it_global]
            if i == 0:  # only the first group is regularized
                param_group['weight_decay'] = wd_schedule[it_global]
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])
        metric_logger.update(weight_decay=optimizer.param_groups[0]['weight_decay'])
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(samples[:3])  # only the 3 global views pass through the teacher
            student_output = student(samples)
            loss = dino_loss(student_output, teacher_output, epoch)
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
         # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        if device.type == 'cuda':
            torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(3)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        print('DINO loss:', total_loss.item())
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
#------Data Augmentation for Time Series DINO -------
class DataAugmentationDino:
    def __init__(self, local_crops_number, transformation_group_size):
        self.sigma = random.uniform(0.01, 0.1)
        self.local_crops_number = local_crops_number
        self.transformation_group_size = transformation_group_size
        self.global_transform1 = transforms.Compose([
            transforms.Lambda(lambda x: x + torch.randn_like(x) * self.sigma),
            aug.polar_transformation()
            ])
        self.global_transform2 = transforms.Compose([
            transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1*self.sigma),
            ])
        self.global_transform3 = transforms.Compose([
            transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1*self.sigma),
            aug.HyperBolicGeometry()
            ])
        self.local_transform1 = transforms.Compose([
            lambda x: self._random_crop(x,self.local_crops_number),
            transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1*self.sigma),
            aug.galilien_transformation()
            ])
        self.local_transform2 = transforms.Compose([
            lambda x: self._random_crop(x,self.local_crops_number),
            transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1*self.sigma),
            aug.rotation_transformation()
            ])
        self.local_transform3 = transforms.Compose([
            lambda x: self._random_crop(x,self.local_crops_number),
            transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1*self.sigma),
            aug.boost_transformation()
            ])
        self.local_transform4 = transforms.Compose([
            lambda x: self._random_crop(x,self.local_crops_number),
            transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1*self.sigma),
            aug.lorentz_transformation()
            ])
        self.local_transform5 = transforms.Compose([
            lambda x: self._random_crop(x,self.local_crops_number),
            transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1*self.sigma),
            aug.polar_transformation()
            ])
        self.local_transform6 = transforms.Compose([
            lambda x: self._random_crop(x,self.local_crops_number),
            transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1*self.sigma),
            aug.HyperBolicGeometry()
            ])


    def _random_crop(self, x, length):
        total_len = x.shape[0]
        start = torch.randint(0, total_len - int(length*total_len) + 1, (1,)).item()
        return x[start : start + int(length*total_len), :, :]

    def __call__(self, x):
        crops = []
        crops.append(self.global_transform1(x))
        crops.append(self.global_transform2(x))
        crops.append(self.global_transform3(x))
        local_transforms = [self.local_transform1, self.local_transform2, self.local_transform3,
                            self.local_transform4, self.local_transform5, self.local_transform6]
        for _ in range(self.transformation_group_size):
            transform = random.choice(local_transforms)
            crops.append(transform(x))
        return crops
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_TS_DINO(args)
