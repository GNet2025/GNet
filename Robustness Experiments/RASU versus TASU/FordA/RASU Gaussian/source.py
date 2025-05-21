import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import random
import os


  

def train(model, trainloader, criterion, optimizer, num_epochs=100):
    
    for epoch in tqdm(range(num_epochs)):
        # trainloader.dataset.reset_augmentation()
        model.cuda()
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for images, labels in trainloader:
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            
            # Calculate accuracy
            _, predicted = output.max(1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
        
        # Print metrics
        accuracy = 100 * correct_predictions / total_predictions
        if epoch == num_epochs - 1:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(trainloader):.4f}, Accuracy: {accuracy:.2f}%")



def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)  # Move data to GPU
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total
