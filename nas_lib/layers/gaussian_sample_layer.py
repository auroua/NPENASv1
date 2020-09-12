# Copyright (c) XiDian University and Xi'an University of Posts&Telecommunication. All Rights Reserved

import torch
from torch.autograd import Function
from torch.autograd import gradcheck


class GaussianFunction(Function):
    @staticmethod
    def forward(ctx, mean, std, vec):
        ctx.save_for_backward(mean, std, vec)
        # output = vec.mul_(std).add_(mean)
        output = vec.mul(std).add(mean)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        mean, std, vec = ctx.saved_tensors
        grad_mean = grad_std = grad_vec = None
        if ctx.needs_input_grad[0]:
            grad_mean = torch.ones_like(mean).mul(grad_output)
        if ctx.needs_input_grad[1]:
            grad_std = vec.mul(grad_output)
            # grad_std = vec.mul(-1*grad_output)
            # grad_std = vec.mul(torch.exp(-1*grad_output))
        return grad_mean, grad_std, grad_vec


if __name__ == '__main__':
    gaussian = GaussianFunction.apply
    for i in range(1000):
        vec = torch.randn(10, 15)
        input = (torch.randn(10, 15, dtype=torch.double, requires_grad=True),
                 torch.randn(10, 15, dtype=torch.double, requires_grad=True), vec)
        test = gradcheck(gaussian, input, eps=1e-6, atol=1e-4)
        print(test)
    torch.sigmoid()