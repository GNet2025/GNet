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
    # print('\nTesting Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    #     test_loss, correct, len(loader.dataset),
    #     100. * correct / len(loader.dataset)))

    return 100. * correct / len(loader.dataset)


def train(trainloader, testloader, lr, dim):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # criterion = torch.nn.NLLLoss()
    criterion = torch.nn.CrossEntropyLoss()
    channels = 1
    classes = 10
    model = BModel(in_dim=channels *dim, classes=classes).to(device)
    #     optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)#, momentum=0.9, weight_decay=1.0e-5)
    #     optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epochs = 3
    for epoch in range(epochs):
        # print("Epoch:", epoch + 1)
        model.train()
        start_time = time.time()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(2 * inputs - 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, batch_predicted = torch.max(outputs.data, 1)
            batch_accu = 100.0 * (batch_predicted == labels).sum().item() / labels.size(0)
            # if i % 50 == 49:
            #     print(i, "{0:.4f}".format(loss.item()), batch_accu)
        # print('Start testing on test set')
        # test(model, testloader, criterion, device, model_='rff-hdc')
        # print("--- %s seconds ---" % (time.time() - start_time))

    return test(model, testloader, criterion, device, model_='rff-hdc')


def argument_parser():
    parser = argparse.ArgumentParser(description='HDC Encoding and Training')
    parser.add_argument('-lr', type=float, default=0.01, help="learning rate for optimizing class representative")
    parser.add_argument('-gamma', type=float, default=0.3, help='kernel parameter for computing covariance')
    parser.add_argument('-epoch', type=int, default=1, help="epochs of training")
    parser.add_argument('-gorder', type=int, default=8, help="order of the cyclic group required for G-VSA")
    parser.add_argument('-dim', type=int, default=10000, help="dimension of hypervectors")
    parser.add_argument('-seed', type=int, default=43, help="random seed for reproducing results")
    parser.add_argument('-resume', action='store_true', help='resume from existing encoded hypervectors')
    parser.add_argument('-data_dir', default='./encoded_data', type=str,
                        help='Directory used to save encoded data (hypervectors)')
    parser.add_argument('-dataset', type=str, choices=['mnist', 'fmnist', 'cifar', 'isolet', 'ucihar'], default='mnist',
                        help='dataset (mnist | fmnist | cifar | isolet | ucihar)')
    parser.add_argument('-raw_data_dir', default='./dataset', type=str, help='Raw data directory to the dataset')
    parser.add_argument('-model', type=str, choices=['rff-hdc', 'linear-hdc', 'rff-gvsa'], default='rff-gvsa',
                        help='feature and model to use: (rff-hdc | linear-hdc | rff-gvsa)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argument_parser()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if 'hdc' in args.model:
        args.gorder = 2
        print("Use binary HDC with random fourier features, ignoring gorder, set to 2.")
    args.data_dir = f'{args.data_dir}/{args.dataset}_{args.model}_order{args.gorder}_gamma{args.gamma}_dim{args.dim}'
    try:
        os.makedirs(args.data_dir)
    except FileExistsError:
        print('Encoded data folder already exists')
    if not args.resume:
        print('Encode the dataset into hypervectors and save')
        encode_and_save(args)
        print('Finish encoding and saving')
    print(f'Optimizing class representatives for {args.epoch} epochs')
    train(args)

