import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import openood.utils.comm as comm
from openood.utils import Config
import torchvision.transforms as transforms

from .lr_scheduler import cosine_annealing
import copy
import random
from attack import *


class RotPredTrainer:
    def __init__(self, net: nn.Module, train_loader: DataLoader,
                 config: Config) -> None:

        self.net = net
        self.train_loader = train_loader
        self.config = config

        self.optimizer = torch.optim.SGD(
            net.parameters(),
            config.optimizer.lr,
            momentum=config.optimizer.momentum,
            weight_decay=config.optimizer.weight_decay,
            nesterov=True,
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                config.optimizer.num_epochs * len(train_loader),
                1,
                1e-6 / config.optimizer.lr,
            ),
        )

    def train_epoch(self, epoch_idx):
        self.net.train()

        loss_avg = 0.0
        train_dataiter = iter(self.train_loader)

        for train_step in tqdm(range(1,
                                     len(train_dataiter) + 1),
                               desc='Epoch {:03d}: '.format(epoch_idx),
                               position=0,
                               leave=True,
                               disable=not comm.is_main_process()):
            batch = next(train_dataiter)
            data = batch['data'].cuda()
            target = batch['label'].cuda()

            batch_size = len(data)
            x_90 = torch.rot90(data, 1, [2, 3])
            x_180 = torch.rot90(data, 2, [2, 3])
            x_270 = torch.rot90(data, 3, [2, 3])

            x_rot = torch.cat([data, x_90, x_180, x_270])
            y_rot = torch.cat([
                torch.zeros(batch_size),
                torch.ones(batch_size),
                2 * torch.ones(batch_size),
                3 * torch.ones(batch_size),
            ]).long().cuda()

            # forward
            logits, logits_rot = self.net(x_rot, return_rot_logits=True)
            loss_cls = F.cross_entropy(logits[:batch_size], target)
            loss_rot = F.cross_entropy(logits_rot, y_rot)
            loss = loss_cls + loss_rot

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        # comm.synchronize()

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = self.save_metrics(loss_avg)

        return self.net, metrics

    def save_metrics(self, loss_avg):
        all_loss = comm.gather(loss_avg)
        total_losses_reduced = np.mean([x for x in all_loss])

        return total_losses_reduced


# FedOV+RotPred

def cut(x):
    x_gen = copy.deepcopy(x.cpu().numpy())
    half = int(x_gen.shape[2] / 2)
    rnd = random.randint(0,5)
    pl = random.randint(0,half-1)
    pl2 = random.randint(0,half-1)
    while (abs(pl-pl2)<half/2):
        pl2 = random.randint(0,half-1)
    if rnd <= 1:
        x_gen[:,:,pl:pl+half] = x_gen[:,:,pl2:pl2+half]
    elif rnd == 2:
        x_gen[:,:,half:] = x_gen[:,:,:half]
        x_gen[:,:,:half] = copy.deepcopy(x.cpu().numpy())[:,:,half:]
    elif rnd <= 4:
        x_gen[:,pl:pl+half,:] = x_gen[:,pl2:pl2+half,:]
    else:
        x_gen[:,half:,:] = x_gen[:,:half,:]
        x_gen[:,:half,:] = copy.deepcopy(x.cpu().numpy())[:,half:,:]
    x_gen = torch.Tensor(x_gen)

    return x_gen

def rot(x):
    #rnd = random.randint(0,20)
    #if rnd < 21:
    x_gen = copy.deepcopy(x.cpu().numpy())
    half = int(x_gen.shape[2] / 2)
    pl = random.randint(0,half-1)
    rnd = random.randint(1,3)

    x_gen[:,pl:pl+half,half:] = np.rot90(x_gen[:,pl:pl+half,half:],k=rnd,axes=(1,2))
    x_gen[:,pl:pl+half,:half] = np.rot90(x_gen[:,pl:pl+half,:half],k=rnd,axes=(1,2))
    x_gen = torch.Tensor(x_gen)
    #else:
    #    x_gen = op(copy.deepcopy(x))
    #    if rnd < 20:
    #        x_gen = torch.max(x_gen, x)
    #    else:
    #        x_gen = torch.min(x_gen, x)

    return x_gen


def blur(x):
    rnd = random.randint(0,1)
    sz = random.randint(1,4)*2+1
    sz2 = random.randint(0,2)*2+1
    if rnd == 0:
        func = transforms.GaussianBlur(kernel_size=(sz, sz2), sigma=(10, 100))
    else:
        func = transforms.GaussianBlur(kernel_size=(sz2, sz), sigma=(10, 100))
    
    return func(x)

def adaptive_cross_entropy_loss(output, target):
    """
    Custom Cross Entropy Loss with adaptive weighting for the 'unknown' class.
    Args:
    - output: Model predictions (logits) of shape (batch_size, num_classes)
    - target: Ground truth labels of shape (batch_size)
    
    The last class (#num_classes) is considered 'unknown', and its weight is set 
    based on the number of unique classes in the current batch.
    """
    batch_size, num_classes = output.size()

    # Get the unique classes in the batch (target)
    unique_classes_in_batch = torch.unique(target)

    # Count the number of unique classes in the batch
    num_classes_in_batch = len(unique_classes_in_batch)

    # Initialize weights as ones
    weights = torch.ones(num_classes)

    # Set the weight of the 'unknown' class (last class) as 1/num_classes_in_batch
    weights[-1] = 1.0 / num_classes_in_batch

    # Move weights to the same device as output
    weights = weights.to(output.device)

    # Define the cross-entropy loss with the adaptive weights
    criterion = nn.CrossEntropyLoss(weight=weights).to(output.device)

    # Compute the loss
    loss = criterion(output, target)
    
    return loss

class FedOVRotPredTrainer:
    def __init__(self, net: nn.Module, train_loader: DataLoader,
                 config: Config) -> None:

        self.net = net
        self.train_loader = train_loader
        self.config = config

        self.optimizer = torch.optim.SGD(
            net.parameters(),
            config.optimizer.lr,
            momentum=config.optimizer.momentum,
            weight_decay=config.optimizer.weight_decay,
            nesterov=True,
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                config.optimizer.num_epochs * len(train_loader),
                1,
                1e-6 / config.optimizer.lr,
            ),
        )

        self.check_epoch = 50

        self.attack = FastGradientSignUntargeted(self.net, 
                                        epsilon=0.5, 
                                        alpha=0.002, 
                                        min_val=0, 
                                        max_val=1, 
                                        max_iters=5,
                                        device="cuda:0")

        #self.use_rotpred = True

    def train_epoch(self, epoch_idx):
        self.net.train()

        num_class = 10
        if (self.config.dataset.name=="cifar100"):
            num_class = 100

        loss_avg = 0.0
        loss_known_avg = 0.0
        loss_rot_avg = 0.0
        train_dataiter = iter(self.train_loader)
        for sample in self.train_loader:
            sz = sample['data'].shape[-1]
            break
        aug_crop =  transforms.RandomChoice( [
            transforms.RandomResizedCrop(sz, scale=(0.1, 0.33)), # good
            transforms.Lambda(lambda img: blur(img)), # good
            #transforms.Lambda(lambda img: shuffle(img)), # bad
            transforms.RandomErasing(p=1, scale=(0.33, 0.5)), # good
            transforms.Lambda(lambda img: cut(img)), # fine
            transforms.Lambda(lambda img: rot(img)),
            transforms.Lambda(lambda img: cut(img)),
            transforms.Lambda(lambda img: rot(img)),
            #transforms.Lambda(lambda img: paint(img))
        ])

        aug_final = transforms.RandomChoice( [
            transforms.Lambda(lambda img: aug_crop(img)),
            #transforms.Lambda(lambda img: cp(img)) # delete this comment if you want to add cutpaste augmentation
        ])

        loss_rot_total = []

        for train_step in tqdm(range(1,
                                     len(train_dataiter) + 1),
                               desc='Epoch {:03d}: '.format(epoch_idx),
                               position=0,
                               leave=True,
                               disable=not comm.is_main_process()):
            batch = next(train_dataiter)
            data = batch['data'].cuda()
            target = batch['label'].cuda()

            batch_size = len(data)
            y_gen = np.ones(data.shape[0]) * num_class
            y_gen = torch.LongTensor(y_gen).cuda()
            
            # y_gen = torch.cat([
            #     torch.ones(batch_size),
            #     torch.zeros(batch_size)
            # ]).long().cuda()
         
            x_gen = copy.deepcopy(data.cpu().numpy())
            for i in range(x_gen.shape[0]):
                x_gen[i] = aug_final(torch.Tensor(x_gen[i]))
            x_gen = torch.Tensor(x_gen).cuda()

            # x_con = torch.cat([data,x_gen],dim=0)
            # y_con = torch.cat([target,y_gen],dim=0)

            # x_90 = copy.deepcopy(data)
            # x_180 = copy.deepcopy(data)
            # x_270 = copy.deepcopy(data)
            # half = int(data.shape[2] / 2)
            # pl = random.randint(0,half-1)
            # pl2 = random.randint(0,half-1)

            x_90 = torch.rot90(data, 1, [2, 3])
            x_180 = torch.rot90(data, 2, [2, 3])
            x_270 = torch.rot90(data, 3, [2, 3])

            x_rot = torch.cat([data, x_90, x_180, x_270])
            y_rot = torch.cat([
                torch.zeros(batch_size),
                torch.ones(batch_size),
                2 * torch.ones(batch_size),
                3 * torch.ones(batch_size),
            ]).long().cuda()

            x_con = torch.cat([data, x_gen])
            y_con = torch.cat([target,y_gen])

            if self.config.dataset.name == "mnist":
                adv_data = self.attack.perturb(x_gen, y_gen)
                x_con = torch.cat([x_con, adv_data])
                y_con = torch.cat([y_con, y_gen])
                

            # forward
            logits, logits_rot = self.net(x_rot, return_rot_logits=True)
            loss_cls = adaptive_cross_entropy_loss(self.net(x_con), y_con)
            if self.net.use_rotpred:
                loss_rot = F.cross_entropy(logits_rot, y_rot)
            else:
                loss_rot = 0

            loss = loss_cls
            

            # if epoch_idx == self.check_epoch:
            #     loss_rot_item = F.cross_entropy(logits_rot[:batch_size*4], y_rot, reduce=False).detach().cpu()
            #     loss_rot_item_whole = (loss_rot_item[:batch_size]+loss_rot_item[batch_size:batch_size*2]+loss_rot_item[batch_size*2:batch_size*3]+loss_rot_item[batch_size*3:])/4
            #     loss_rot_total.append(loss_rot_item_whole)
                
            if self.net.use_rotpred:
                loss += loss_rot

            # backward
            if epoch_idx != self.check_epoch:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2
                loss_rot_avg = loss_rot_avg * 0.8 + float(loss_rot) * 0.2

        # comm.synchronize()

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = self.save_metrics(loss_avg)
        metrics['rot'] = loss_rot_avg
        if epoch_idx == self.check_epoch:
            # loss_rot_total = torch.cat(loss_rot_total)
            # k = int(len(loss_rot_total)*0.95)
            # threshold_value, _ = torch.kthvalue(loss_rot_total, k)
            # print(threshold_value)
            if loss_rot_avg > 0.5:
                self.net.use_rotpred = False

        return self.net, metrics

    def save_metrics(self, loss_avg):
        all_loss = comm.gather(loss_avg)
        total_losses_reduced = np.mean([x for x in all_loss])

        return total_losses_reduced

