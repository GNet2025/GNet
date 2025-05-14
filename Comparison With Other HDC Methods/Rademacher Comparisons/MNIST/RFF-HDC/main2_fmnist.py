#!/usr/bin/env python
# coding: utf-8
import torch
import numpy as np
import time
import os
from utils import prepare_data, encode_and_save
from model import BModel, GModel
import argparse


def test(MODEL, loader, criterion, device, model_='rff-hdc'):
    MODEL.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            if model_ == 'rff-hdc':
                outputs = MODEL(2 * inputs - 1)
            else:
                outputs = MODEL(inputs)
            test_loss += criterion(outputs, labels)
            preds = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += preds.eq(labels.view_as(preds)).sum().item()
    test_loss /= len(loader.dataset)
    

    return 100. * correct / len(loader.dataset)


def train(trainloader, testloader, lr, dim):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss()
    x, _ = next(iter(testloader))
    channels = 1
    # channels = 3
    classes = 10
    model = BModel(in_dim=channels * dim, classes=classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epochs = 3
    for epoch in range(epochs):
        model.train()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(2 * inputs - 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, batch_predicted = torch.max(outputs.data, 1)
            batch_accu = 100.0 * (batch_predicted == labels).sum().item() / labels.size(0)

    return test(model, testloader, criterion, device, model_='rff-hdc')
