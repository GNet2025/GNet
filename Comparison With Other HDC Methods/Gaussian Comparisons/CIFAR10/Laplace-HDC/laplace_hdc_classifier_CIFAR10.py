#!/usr/bin/env python
# coding: utf-8

import time
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# 4) HDC encoding utilities

def generate_base_HDVs(D, B, device):
    """Returns a (B×D) random {0,1} intTensor."""
    return (torch.rand(B, D, device=device) > 0.5).int()

@torch.no_grad()
def encode_dataset_batched(X, base_HDVs, batch_size=128):
    """
    X:        (N, B) floatTensor {0,1}
    base_HDVs: (B, D) intTensor
    → (N, D) intTensor {0,1}
    """
    N, B = X.shape
    D    = base_HDVs.shape[1]
    perm_HDVs = base_HDVs.roll(shifts=1, dims=1).unsqueeze(0)  # (1,B,D)
    base      = base_HDVs.unsqueeze(0)                         # (1,B,D)

    chunks = []
    for i in tqdm(range(0, N, batch_size), desc="Encoding"):
        xb     = X[i : i+batch_size].unsqueeze(-1)  # (b,B,1)
        weighted = xb * perm_HDVs + (1-xb)*base     # (b,B,D)
        H_float  = weighted.mean(dim=1)             # (b,D)
        chunks.append(torch.round(H_float).int())
    return torch.cat(chunks, dim=0)

def encode_class_HDVs(H_train, y_train, C):
    """Bundle per‐class mean and threshold to {0,1}."""
    class_HDVs = []
    for c in range(C):
        m = H_train[y_train==c].float().mean(dim=0)
        class_HDVs.append(torch.round(m).int())
    return torch.stack(class_HDVs, dim=0)

def predict(H_test, class_HDVs):
    """Nearest neighbor by Hamming distance."""
    diffs = H_test.unsqueeze(1) != class_HDVs.unsqueeze(0)  # (M,C,D)
    dists = diffs.sum(dim=2)                                # (M,C)
    return dists.argmin(dim=1)                              # (M,)


# 5) Classification helpers

def test_hdc_classifier(model, testloader):
    """Evaluate any model on a DataLoader, returns accuracy."""
    model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds   = outputs.argmax(dim=1)
            correct += (preds==labels).sum().item()
    return correct / len(testloader.dataset)

def train_hdc_classifier(trainloader, num_classes, mode="binary_sgd", epochs=1):
    """
    Available modes: 'binary_sgd','float_sgd','binary_majority',
                     'float_majority','cnn_sgd'
    """
    sample = next(iter(trainloader))[0]
    shape  = sample.shape[1:]              # e.g. (D,) or (C,H,W)
    hyperdim = int(np.prod(shape))

    if mode=="binary_sgd":
        model = _binary_sgd(trainloader, num_classes, epochs, hyperdim)
    elif mode=="float_sgd":
        model = _float_sgd(trainloader, num_classes, epochs, hyperdim)
    elif mode=="binary_majority":
        model = _binary_majority(trainloader, num_classes, hyperdim)
    elif mode=="float_majority":
        model = _float_majority(trainloader, num_classes, hyperdim)
    elif mode=="cnn_sgd":
        # assume shape = (in_channels, H, W)
        c,h,w = shape
        model = _cnn_sgd(trainloader, num_classes, epochs, in_channels=c, height=h, width=w)
    else:
        raise ValueError(f"Unknown mode {mode}")
    return model

# ——— SGD & Majority implementations —————————————————————————————
def _float_sgd(trainloader, num_classes, epochs, hyperdim):
    class Model(nn.Module):
        def __init__(self, hyperdim, num_classes):
            super().__init__()
            self.linear = nn.Linear(hyperdim, num_classes, bias=False)
            self.hyperdim = hyperdim
        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = 1-2*x.float()
            return self.linear(x) / (self.hyperdim**0.5)

    model    = Model(hyperdim, num_classes).to(device)
    criterion= nn.CrossEntropyLoss()
    optim    = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for _ in tqdm(range(epochs), desc="Float‐SGD"):
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            loss = criterion(model(inputs), labels)
            optim.zero_grad(); loss.backward(); optim.step()
    return model

def _binary_sgd(trainloader, num_classes, epochs, hyperdim):
    class Model(nn.Module):
        def __init__(self, hyperdim, num_classes):
            super().__init__()
            self.linear = nn.Linear(hyperdim, num_classes, bias=False)
            self.hyperdim = hyperdim
        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = 1-2*x.float()
            return self.linear(x) / (self.hyperdim**0.5)

    model    = Model(hyperdim, num_classes).to(device)
    criterion= nn.CrossEntropyLoss()
    optim    = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for _ in tqdm(range(epochs), desc="Binary‐SGD"):
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            loss = criterion(model(inputs), labels)
            optim.zero_grad(); loss.backward(); optim.step()
            with torch.no_grad():
                model.linear.weight.data.clamp_(-1,1)
    with torch.no_grad():
        model.linear.weight.data.sign_()
    return model

def _binary_majority(trainloader, num_classes, hyperdim):
    A      = torch.zeros(num_classes, hyperdim, device=device)
    counts = torch.zeros(num_classes, device=device)
    for inputs, labels in trainloader:
        x = (1-2*inputs.view(inputs.size(0), -1).float()).to(device)
        for c in range(num_classes):
            mask = (labels==c)
            if mask.any():
                A[c]    += x[mask].sum(dim=0)
                counts[c]+= mask.sum()
    A = (A/counts.unsqueeze(1)).sign()  # {−1,+1}
    class HVClassifier(nn.Module):
        def __init__(self, A): super().__init__(); self.A = A
        def forward(self, x):
            x = (1-2*x.view(x.size(0), -1).float())
            return x @ self.A.t()
    return HVClassifier(A).to(device)

def _float_majority(trainloader, num_classes, hyperdim):
    A      = torch.zeros(num_classes, hyperdim, device=device)
    counts = torch.zeros(num_classes, device=device)
    for inputs, labels in trainloader:
        x = (1-2*inputs.view(inputs.size(0), -1).float()).to(device)
        for c in range(num_classes):
            mask = (labels==c)
            if mask.any():
                A[c]    += x[mask].sum(dim=0)
                counts[c]+= mask.sum()
    A = A/counts.unsqueeze(1)
    class HVClassifier(nn.Module):
        def __init__(self, A): super().__init__(); self.A = A
        def forward(self, x):
            x = (1-2*x.view(x.size(0), -1).float())
            return x @ self.A.t()
    return HVClassifier(A).to(device)

def _cnn_sgd(trainloader, num_classes, epochs, in_channels, height, width):
    class BasicCNN(nn.Module):
        def __init__(self, in_ch, h, w, num_classes):
            super().__init__()
            self.conv  = nn.Conv2d(in_ch, 16, kernel_size=3, stride=1, padding=0)
            self.pool  = nn.MaxPool2d(2,2)
            m1 = (h-3+1); m2 = (w-3+1)
            k1 = (m1-2)//2 + 1; k2 = (m2-2)//2 + 1
            N  = k1*k2*16
            self.fc   = nn.Linear(N, num_classes)
            self.N    = N
        def forward(self, x):
            x = (1-2*x.float())
            x = self.pool(self.conv(x))
            x = x.view(x.size(0), -1)
            return self.fc(x) / (self.N**0.5)

    model    = BasicCNN(in_channels, height, width, num_classes).to(device)
    criterion= nn.CrossEntropyLoss()
    optim    = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for _ in tqdm(range(epochs), desc="CNN‐SGD"):
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            loss = criterion(model(inputs), labels)
            optim.zero_grad(); loss.backward(); optim.step()
    return model


