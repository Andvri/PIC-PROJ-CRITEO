import torch
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt

data_dir = 'data/urls'

data = datasets.ImageFolder(data_dir)

classes = data.classes
num_classes = len(classes)
bins,hist = torch.unique(torch.tensor(data.targets), return_counts=True) 

hist = hist.cpu().detach().numpy()
bins = bins.cpu().detach().numpy()

plt.subplots(figsize=(20, 2))
plt.bar(classes, hist, align='center')

plt.savefig("Dataset2 number of samples per class")
