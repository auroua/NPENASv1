import math
import torch
from ..layers.loss_gausian import Criterion
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
import torch.distributed as dist


def gen_batch_idx(idx_list, batch_size):
    ds_len = len(idx_list)
    idx_batch_list = []

    for i in range(0, math.ceil(ds_len/batch_size)):
        if (i+1)*batch_size > ds_len:
            idx_batch_list.append(idx_list[i*batch_size:])
        else:
            idx_batch_list.append(idx_list[i*batch_size: (i+1)*batch_size])
    return idx_batch_list


def make_agent_optimizer(model, base_lr, weight_deacy=1e-4, bias_multiply=True):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = base_lr
        weight_decay = weight_deacy
        if "bias" in key:
            if bias_multiply:
                lr = base_lr*2.0
            else:
                lr = base_lr
            weight_decay = 0.0
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    optimizer = torch.optim.Adam(params, base_lr, (0.0, 0.9))
    return optimizer


def make_agent_optimizer_std(model, base_lr, fileds='', weight_deacy=1e-4, bias_multiply=True):
    params = []
    for key, value in model.named_parameters():
        if fileds not in key:
            continue
        if not value.requires_grad:
            continue
        lr = base_lr
        weight_decay = weight_deacy
        if "bias" in key:
            if bias_multiply:
                lr = base_lr*2.0
            else:
                lr = base_lr
            weight_decay = 0.0
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    optimizer = torch.optim.Adam(params, base_lr, (0.0, 0.9))
    return optimizer


def make_agent_optimizer_mean(model, base_lr, fileds='', weight_deacy=1e-4, bias_multiply=True):
    params = []
    for key, value in model.named_parameters():
        if fileds in key:
            continue
        if not value.requires_grad:
            continue
        lr = base_lr
        weight_decay = weight_deacy
        if "bias" in key:
            if bias_multiply:
                lr = base_lr*2.0
            else:
                lr = base_lr
            weight_decay = 0.0
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    optimizer = torch.optim.Adam(params, base_lr, (0.0, 0.9))
    return optimizer


def lr_step(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_loss_criteria(loss_type):
    if loss_type == 'mse':
        criterion = torch.nn.MSELoss()
    elif loss_type == 'mae':
        criterion = torch.nn.L1Loss()
    elif loss_type == 'gaussian':
        criterion = Criterion()
    else:
        raise ValueError('This loss type does not support!')
    return criterion


class CosineLR(_LRScheduler):
    def __init__(self, optimizer, epochs, train_images, batch_size):
        self.epochs = epochs
        self.train_image_num = train_images
        self.batch_size = batch_size
        self.total_steps = int(self.epochs*self.train_image_num / self.batch_size)
        super(CosineLR, self).__init__(optimizer, -1)

    def get_lr(self):
        progress_fraction = float(self._step_count+1) / self.total_steps
        lr_lists = [(0.5 * base_lr * (1 + math.cos(np.pi * progress_fraction)))
                    for base_lr in self.base_lrs]
        return lr_lists

    def set_train_images(self, new_count):
        self.train_image_num = new_count


def compute_best_test_losses(data, k, total_queries):
    """
    Given full data from a completed nas algorithm,
    output the test error of the arch with the best val error
    after every multiple of k
    """
    results = []
    results_keys = []
    total_data = []
    for i in range(total_queries):
        total_data.append(data[i])
    for query in range(k, total_queries + k, k):
        best_arch = sorted(total_data[:query], key=lambda i: i[0])[0]
        test_error = best_arch[1]
        results.append((query, test_error))
        results_keys.append(best_arch[2])
    return results, results_keys


def compute_bananas_test_losses(data, k, total_queries):
    """
    Given full data from a completed nas algorithm,
    output the test error of the arch with the best val error
    after every multiple of k
    """
    results = []
    results_keys = []
    model_archs, model_keys = data
    losses = [(model_archs[k][1], model_archs[k][2], k) for k in model_keys]
    for query in range(k, total_queries + k, k):
        best_arch = sorted(losses[:query], key=lambda i: i[0])[0]
        test_error = best_arch[1]
        results.append((query, test_error))
        results_keys.append(best_arch[2])
    return results, results_keys


def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()