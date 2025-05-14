#!/usr/bin/env python
# coding: utf-8

# In[13]:


import torch
import numpy as np
import time
import os
from utils import HDDataset
from model import BModel, GModel
from torchvision import datasets, transforms
import scipy


# In[14]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[15]:


transform = transforms.Compose([
        transforms.ToTensor(),
])

trainset = datasets.MNIST(root='../../../../../', train=True, download=True, transform=transform)
testset = datasets.MNIST(root='../../../../../', train=False, download=True, transform=transform)


# In[16]:


assert len(trainset[0][0].size()) > 1


# In[17]:


channels = trainset[0][0].size(0)
print('# of channels of data', channels)


# In[18]:


input_dim = torch.prod(torch.tensor(list(trainset[0][0].size())))
print('# of training samples and test samples', len(trainset), len(testset))


# In[19]:


class RandomFourierEncoder:
    def __init__(self, input_dim, gamma, gorder=2, output_dim=10000):
        self.input_dim = input_dim
        self.output_dim = output_dim
        # kernel parameter
        self.gamma = gamma
        self.gorder = gorder
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def pts_map(self, x, r=1.0):
        theta = 2.0 * np.pi / (1.0 * self.gorder) * x
        pts = r * torch.stack([torch.cos(theta), torch.sin(theta)], -1)
        return pts

    def GroupRFF(self, x, sigma):
        intervals = sigma * torch.tensor(
            [scipy.stats.norm.ppf(i * 1.0 / self.gorder) for i in range(1, self.gorder)]).float()
        print('the threshold to discretize fourier features to group elements', intervals)
        group_index = torch.zeros_like(x)
        group_index[x <= intervals[0]] = 0
        group_index[x > intervals[-1]] = self.gorder - 1
        if self.gorder > 2:
            for i in range(1, self.gorder - 1):
                group_index[(x > intervals[i - 1]) & (x <= intervals[i])] = i
        return group_index

    def build_item_mem(self):  # create random fourier features for 256 pixel values
        # a correction factor for bias
        correction_factor = 1 / 1.4
        # covariance kernel
        x = np.linspace(0, 255, num=256)
        Cov = np.array([np.exp(-correction_factor * self.gamma ** 2 * ((x - y) / 255.0) ** 2 / 2) for y in range(256)])
        k = Cov.shape[0]
        assert Cov.shape[1] == k, "Cov is not a square matrix."
        L = np.sin(Cov * np.pi / 2.0)
        ''' Eigen decomposition: L = eigen_vectors @ np.diag(eigen_values) @ eigen_vectors.T '''
        eigen_values, eigen_vectors = np.linalg.eigh(L)
        R = eigen_vectors @ np.diag(np.maximum(0, eigen_values) ** 0.5) @ eigen_vectors.T
        item_mem = torch.from_numpy(np.random.randn(self.output_dim, k) @ R).float()  # this can be reduced for effiency
        self.item_mem = self.GroupRFF(item_mem, np.sqrt((R ** 2).sum(0).max())).T  # compatible for binary HDC (with %2)
        # for binary HDC, you can also use a bipolar representation together with prod alternatively
        # self.item_mem = (((item_mem >= 0).T * 1.0 - 0.5) * 2) # bipolar representation for binary HDC
        self.item_mem = self.item_mem.to(self.device)
        return self.item_mem

    def encode_one_img(self, x):
        '''
        x:   flattened input image. size=[#pixels,]
        rfs: random feature vectors for pixel values. size=[HDC Dim, #features]
        '''
        x = x.to(self.device).long()
        bs, channels, num_pixels = x.size()
        rv = self.item_mem[x.flatten()].view(bs, channels, num_pixels, -1).transpose(0, 2)
        # rv shape now should be [num_pixels, channels, bs, hyperD]
        for i in range(num_pixels):
            # for each pixel, shift along hyperD dimension
            rv[i] = torch.roll(rv[i], shifts=783 - i,
                               dims=-1)  # note that this batch shifting might be different from our v1
        rv = torch.sum(rv, dim=0)  # use sum, natural extends to group bind, result shape: [channels, bs, hyperD]
        #         rv = torch.fmod(torch.sum(rv, dim=0), self.gorder) # mathly same since we use cos in the GModel
        if self.gorder == 2:
            rv = rv % 2
        # the following works when bipolar representation is used
        #       # rv = torch.prod(rv, 0) > 0.
        return rv.transpose(0, 1).reshape((bs, -1))

    # returns an array of HD features for multiple inputs, together with label list
    def encode_data_extract_labels(self, datast):
        '''
        datast:   trainset or testset loaded via torch. tuple style, contains N (x,y) pair.
        rfs: random feature vectors for pixel values. shape=[HDC Dim, #features]
        return: rv -> hypervectors for images. shape=[N, HDC dim]
        '''
        channels = datast[0][0].size(0)
        n = len(datast)  # number of examples in x
        rv = torch.zeros((n, channels * self.output_dim))
        labels = torch.zeros(n).long()
        # print('Start encoding data')
        start_time = time.time()
        batch_size = 128
        data_loader = torch.utils.data.DataLoader(datast, batch_size=batch_size, shuffle=False)
        for i, batch_img in enumerate(data_loader):
            num_imgs = batch_img[0].size(0)  # in case the last batch is not equal to batch_size
            rv[i * batch_size: i * batch_size + num_imgs] = self.encode_one_img(
                (255 * batch_img[0].view(num_imgs, channels, -1)).int())
            labels[i * batch_size: i * batch_size + num_imgs] = batch_img[1]
            # if i % 100 == 99: print(
                # f"{(i + 1) * batch_size} images encoded. Total time elapse = {time.time() - start_time}")
        # print('Finish encoding data')
        return rv, labels

    def group_bind(self, lst):
        results = torch.sum(lst, dim=0)
        return results  # torch.fmod(results, self.gorder) # mathematically same

    def group_bundle(self, lst):
        intervals = torch.tensor([2 * np.pi / self.gorder * i for i in range(self.gorder)]) + np.pi / self.gorder
        pts = torch.sum(self.pts_map(lst), dim=0)
        raw_angles = 2 * np.pi + torch.arctan(pts[:, 1] / pts[:, 0]) - np.pi * (pts[:, 0] < 0).float()
        angles = torch.fmod(raw_angles, 2 * np.pi)
        return torch.floor(angles / (2.0 * np.pi) * self.gorder + 1 / 2)  # torch.fmod( , self.gorder)

    def similarity(self, x, y):
        return torch.sum(torch.sum(self.pts_map(x) * self.pts_map(y), dim=-1), dim=-1) * (1.0 / x.size(-1))


# In[20]:


lr = 0.01 
gamma = 0.3 
epochs = 3
gorder = 2 # for RFF-HDC it should be equal to 2.
dim = 10000
seed = 43 
resume = True
classes = 10
model = BModel(in_dim=channels * dim, classes=classes).to(device)


# In[21]:


def test(MODEL, loader, criterion, device, model_='rff-hdc'):
    MODEL.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = MODEL(2 * inputs - 1)
            
            test_loss += criterion(outputs, labels)
            preds = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += preds.eq(labels.view_as(preds)).sum().item()
    test_loss /= len(loader.dataset)
    # print('\nTesting Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    #     test_loss, correct, len(loader.dataset),
    return 100. * correct / len(loader.dataset)


# In[22]:


def train(model, trainloader, testloader, epochs, lr):
    # criterion = torch.nn.NLLLoss()
    criterion = torch.nn.CrossEntropyLoss()
    #     optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)#, momentum=0.9, weight_decay=1.0e-5)
    #     optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
        # print('Start testing on test set')
    acc = test(model, testloader, criterion, device)
        # print("--- %s seconds ---" % (time.time() - start_time))

    return acc


# In[23]:


# train(model, trainloader, testloader, epochs, lr)


# In[ ]:


accuracy = np.zeros(10)
for i in range(10):
    encoder = RandomFourierEncoder(
            input_dim=input_dim, 
            gamma=gamma, 
            gorder=gorder, 
            output_dim=dim)

    encoder.build_item_mem()  # Initialize item_mem
    print(f"iteration: {i}----------")
    # print("Encoding training data...")
    train_hd, y_train = encoder.encode_data_extract_labels(trainset)

    # print("Encoding test data")
    test_hd, y_test = encoder.encode_data_extract_labels(testset)

    train_dataset = HDDataset(train_hd, y_train)
    test_dataset = HDDataset(test_hd, y_test)
    trainloader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=16,
                                              shuffle=True,
                                              pin_memory=True,
                                              num_workers=1)
    testloader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=16,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=1)

    accuracy[i] = train(model, trainloader, testloader, epochs, lr)
    print(accuracy[i])
    torch.cuda.empty_cache()


# In[18]:


accuracy


# In[ ]:




