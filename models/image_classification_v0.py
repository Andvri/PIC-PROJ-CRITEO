from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

from dataloader_image import *

#data_dir = 'data/hymenoptera_data'
data_dir = 'data/google taxonomy'
train_dir = data_dir + '/train'
test_dir = data_dir + '/val'

batch_size = 4

model_name = "resnet18"
my_models = {"resnet18" : models.resnet18}
input_size = {"resnet18" : 224}



# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size[model_name]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size[model_name]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
train_dataloader, test_dataloader, dataset_sizes, class_names= dataloader(train_dir, test_dir, batch_size, data_transforms)
num_classes = len(class_names)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def categories_embedding(model, train_dataloader, num_classes, embedding_size):
    embedding = np.zeros((num_classes,embedding_size))
    label_count = [0]*num_classes
    for inputs, labels in train_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        outputs = outputs.detach().numpy()
        labels = labels.detach().numpy()
        for i in range(len(labels)):
            label = labels[i]
            embedding[label] += outputs[i]
            label_count[label] +=1
    for i in range(num_classes):
        if label_count[i]:
            embedding[i]/=label_count[i]
    return embedding

def eval(model, test_dataloader, embedding, class_names):
    correct = 0
    for inputs, labels in test_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        for i in range(len(labels)):
            output = outputs[i]
            label = labels[i]
            label_predict = 0
            similarity_max = 0
            for j in range(len(embedding)):
                category_embedding = torch.LongTensor(embedding[j])
                similarity = cos(output, category_embedding)
                if similarity > similarity_max:
                    similarity_max = similarity
                    label_predict = j
            if label==label_predict:
                correct+=1
            #print("real class : ", class_names[label])
            #print("predicted class : ", class_names[label_predict])
            #print("---------------------------------")
    print("accuracy", correct/(len(test_dataloader)*batch_size))

model = my_models[model_name](pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Sequential(*list(model.fc.children())[:-1])


model = model.to(device)

model.eval() 

#for param in model.parameters():
#    param.requires_grad = False


print("TRAIN")
embedding = categories_embedding(model, train_dataloader, num_classes, num_features)
eval(model, train_dataloader, embedding, class_names)
print("TEST")
eval(model, test_dataloader, embedding, class_names)

