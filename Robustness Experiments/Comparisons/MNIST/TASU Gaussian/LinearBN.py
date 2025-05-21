import torch
import torchvision
import torchvision.transforms as transforms
import PIL
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim

class LinearBN(nn.Module):
    def __init__(self, in_features, out_features, eps=1e-4, q=1e-4, std=0.1, seed = 13):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(self.in_features, self.out_features)
        self.bn = nn.BatchNorm1d(self.out_features)

        self.bn.bias.data.fill_(0)  
        self.bn.bias.requires_grad = False

        self.linear.bias.data.fill_(0)  
        self.linear.bias.requires_grad = False

        nn.init.normal_(self.linear.weight, mean=0.0, std=std) #Ali changed
        self.eps = eps
        self.q = q
        
        self.bias_trick_par = nn.Parameter(torch.tensor(0.000005))
        self.seed = seed

    def custom_round(self, n):
        remainder = n % 1000
        base = n - remainder
        if remainder >= 101:
            return base + 1000
        elif remainder <= 100:
            return base

    def forward(self, x):
        x = x + self.bias_trick_par

        temp = x
        x = self.linear(x)
        if self.training:
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
        else: 
            batch_mean = self.bn.running_mean
            batch_var = self.bn.running_var

        slope = self.bn.weight / torch.sqrt(batch_var + self.eps)
        w_comp = slope.view(-1, 1) * self.linear.weight
        # b_comp = slope * (self.linear.bias - batch_mean) + self.bn.bias

        normw = torch.norm(w_comp, p=2, dim=1) + self.eps
        normx = temp.norm(p=2, dim=1, keepdim=True) + self.q

        x = self.bn(x)

        x = x + (slope * batch_mean).view(1, -1)

        x = x * (1 / normw.view(1, -1)) * (1 / normx.view(-1, 1))
        x = (x.sign() * (x.abs() + self.eps))

        x = torch.asin(x)

        return x

    @torch.no_grad()             # no autograd graph
    def flip_sign_(self, tensor: torch.Tensor, percentage: float) -> torch.Tensor:
        """
        Flip the sign of a random subset of elements *in place*.
    
        Args:
            tensor (torch.Tensor): Any shape, modified in place.
            percentage (float): 0‒1 fraction of elements to flip.
    
        Returns:
            torch.Tensor: The same tensor object (for chaining).
        """
        if percentage <= 0.0:
            return tensor
        if percentage >= 1.0:
            tensor.mul_(-1)
            return tensor                    # all elements flipped
    
        numel = tensor.numel()
        num_to_flip = int(numel * percentage)
        if num_to_flip == 0:
            return tensor
    
        flat = tensor.view(-1)               # view ↔ no copy
        idx = torch.randint(0, numel, (num_to_flip,),
                            device=flat.device)
        flat[idx] *= -1                      # in-place sign change
        return tensor

    def init_hdc(self, ratio, seed, flip_perc=None):
        ratio = ratio

        slope = self.bn.weight / torch.sqrt(self.bn.running_var + self.eps)
        w_comp = slope.view(-1, 1) * self.linear.weight

        n = w_comp.size(1)
        self.nHDC = int(self.custom_round(ratio * n)) if ratio<1000 else int(ratio)
        try:
            del self.g
            del self.wg

        except UnboundLocalError:
            pass  # If 'g' was not defined, do nothing
        except AttributeError:
            pass  # Do nothing if the attribute does not exist
        torch.manual_seed(seed)
        self.g = (torch.randn(w_comp.size(1), self.nHDC)).to(torch.half).to(w_comp.device)

        self.wg = torch.sign(torch.matmul(self.g.t(), w_comp.to(torch.half).t()))
        if flip_perc is not None and flip_perc > 0.0:
            self.wg = self.flip_sign_(self.wg, flip_perc)

    def hdc(self, x):
        x = x + self.bias_trick_par
        x = torch.sign(torch.matmul(x.to(torch.half), self.g))
        x = x @ self.wg * (torch.pi / (2 * self.nHDC))

        return x

