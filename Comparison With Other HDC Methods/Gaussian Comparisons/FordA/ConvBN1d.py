import torch
import torchvision
import torchvision.transforms as transforms
import PIL
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim

class ConvBN(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size,
                 stride: int = 1,
                 padding: int = 2,
                 eps: float=1e-4,
                 q: float=1e-4,
                 initial: bool=True,
                 mean = 0,
                 std = 1,
                 seed = 13,
                 bias_par_init = 0.001):
        super().__init__()

        self.eps = eps
        self.q = q
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv = nn.Conv1d(in_channels=self.in_channels, 
                              out_channels=self.out_channels,
                              kernel_size=self.kernel_size,
                              stride=self.stride,
                              padding=self.padding,
                              bias=False)

        self.bias_trick_par = nn.Parameter(torch.tensor(bias_par_init))
        
        self.bn = nn.BatchNorm1d(self.out_channels, eps=self.eps)
        self.bn.bias.data.zero_()
        self.bn.bias.requires_grad = False
        
        kernel_shape = (1, self.in_channels, self.kernel_size)
        self.register_buffer('norm_filter', torch.ones(kernel_shape))
        
        nn.init.normal_(self.conv.weight, mean=0.0, std=std)
        nn.init.normal_(self.bn.weight, mean=0.0, std=0.05)

        self.seed = seed

    def custom_round(self, n):
        remainder = n % 1000
        base = n - remainder
        if remainder >= 101:
            return base + 1000
        elif remainder <= 100:
            return base


    def forward(self, x):
        
        x = x + (self.bias_trick_par)

        xp = F.pad(x, (self.padding, self.padding), value=0)
        normxp = F.conv1d(xp.square(), self.norm_filter, stride=self.stride, padding=0)
        normxp = (normxp + self.eps).sqrt() + self.eps
        normxp = normxp.expand(-1, self.out_channels, -1)
        
        x = self.conv(x)
        temp = x
        x = self.bn(x)

        if self.training:
            batch_mean = temp.mean(dim=(0, 2))
            batch_var = temp.var(dim=(0, 2), unbiased=False)
        else: 
            batch_mean = self.bn.running_mean
            batch_var = self.bn.running_var

        slope = self.bn.weight / torch.sqrt(batch_var + self.eps)
        w_aug = self.conv.weight * slope.view(-1, 1, 1)

        normw = torch.sum(w_aug.square(), dim=(1,2))
        normw = torch.sqrt(normw + self.eps)
        normw = normw.view(1, -1, 1)

        # print(x.shape, batch_mean.shape, slope.shape)

        x = x + (slope * batch_mean).view(1, -1, 1) # To remove the excess bias coming from batch mean
        
        x = x * (1 / normxp) * (1 / normw)
        # print(torch.min(x), 'after min')
        # print(torch.max(x), 'after max')
        # x = torch.clamp(x, -0.9999, 0.9999)
        x = torch.asin(x)
            
        return x     


    def init_hdc(self, ratio, seed):
        try:
            del self.alphag1
            del self.g
            del self.alpha1

        except UnboundLocalError:
            pass  # If 'g' was not defined, do nothing
        except AttributeError:
            pass  # Do nothing if the attribute does not exist

        slope = self.bn.weight / torch.sqrt(self.bn.running_var + self.eps)
        w_bn = self.conv.weight * slope.view(-1, 1, 1)
        w_bn = w_bn.unsqueeze(0)

        n = w_bn.shape[1:].numel()
        self.nHDC = int(self.custom_round(ratio * n)) if ratio<1000 else int(ratio)

        torch.manual_seed(seed)
        self.g = (torch.randn(self.nHDC, *w_bn.shape[1:], device=w_bn.device, dtype=w_bn.dtype))
        self.alpha1 = torch.sign((w_bn * self.g).sum(dim=(2, 3), keepdim=True))
        
        temp = (self.alpha1 * self.g)
        self.size = temp.shape
        self.alphag1 = temp.view(-1, *w_bn.shape[2:])

    def hdc(self, x):
        B, C, S = x.shape

        x = x + self.bias_trick_par
        x_p = F.pad(x, (self.padding, self.padding), value=0)
        
        out = nn.functional.conv1d(x_p, self.alphag1, stride=self.stride, padding=0)
        out = out.view(B, self.size[0], self.size[1], out.size(2))
        
        zhat = (torch.pi / (2 * self.nHDC)) * torch.sign(out).sum(dim=1)
        return zhat

        