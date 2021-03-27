from io_functions import *

def train_model(model, train_dataloader, test_dataloader, criterion, optimizer, scheduler, num_epochs, device, dataset_sizes, num_classes, weights_train=None, weights_test=None):
    model = model.to(device)
    since = time.time()
    weights = {'train': weights_train, 'val': weights_test}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    accuracies_train = []
    losses_train = []
    accuracies_test = []
    losses_test = []
    weighted_accuracies_test = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_dataloader
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = test_dataloader

            running_loss = 0.0
            running_corrects = 0
            epoch_weighted_accuracy = 0
            # Iterate over data.
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                for i in range(len(preds)):
                    epoch_weighted_accuracy += (1/weights[phase][labels.data[i].item()])*(preds[i]==labels.data[i]).item()
            if phase == 'train':
                scheduler.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                accuracies_train.append(epoch_acc)
                losses_train.append(epoch_loss)
            else :
                accuracies_test.append(epoch_acc)
                losses_test.append(epoch_loss)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            epoch_weighted_accuracy /= num_classes
            print('Weighted accuracy: {:.4f}'.format(epoch_weighted_accuracy))
            if phase == 'val':
                weighted_accuracies_test.append(epoch_weighted_accuracy)

            # deep copy the model
            if phase == 'val' and epoch_weighted_accuracy > best_acc:
                best_acc = epoch_weighted_accuracy
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, accuracies_train, accuracies_test, losses_train, losses_test, weighted_accuracies_test
import torch
from torch.nn import CrossEntropyLoss, Softmax, utils


import time

from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.utils import resample
import numpy as np
import sys

# Constants:
data_dir = 'data/urls'
# data_dir = 'data/google taxonomy'
# train_dir = data_dir + '/train'
# test_dir = data_dir + '/val'

k_folds = 5

version = 2 #1 to retrain all layers, 2 to retrain the last one

batch_size = 32 #not sure about it 32 or 4 for vgg11
epochs = 10
lr = 0.001
momentum = 0.9

step_size = 7
gamma = 0.1

model_name = "resnet34"
my_models = {"resnet18" : models.resnet18,
             "resnet34" : models.resnet34,
             "resnet50" : models.resnet50,
             "alexnet" : models.alexnet,
             "vgg11" : models.vgg11_bn
            }
input_size = {"resnet18" : 224, "resnet34": 224, "resnet50": 224, "alexnet": 224, "vgg11":224}

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


if __name__ == "__main__":
    import os
    import pandas as pd
    from matplotlib import pyplot as plt
    from torch.utils.data import TensorDataset,DataLoader
    
    dataset = datasets.ImageFolder(data_dir, data_transforms['train'])
    class_names = dataset.classes
    # train_dataset, test_dataset, dataset_sizes, class_names = dataloader(train_dir, test_dir, batch_size=None, data_transforms=data_transforms)
    # dataset = ConcatDataset([train_dataset, test_dataset])

    num_classes = len(class_names)


    #Kflod for cross validation
    kfold = KFold(n_splits=k_folds, shuffle=True)

    # Make tests, plot results to compare:
    plt.figure(figsize=(15, 10))
    
    accuracies = []

    crossvalidation_dataset = kfold.split(dataset)
    folds = 0
    for fold, (train_ids,validation_ids) in enumerate(crossvalidation_dataset):
        model, criterion, optimizer, scheduler, device = initiate_model(model_name, my_models, lr, momentum, step_size, gamma, num_classes, version=2)

        # Freeze the entire model except the last layer:
        # We can either re-train the whole transformer
        # or only the last classification layer:

        folds+=1
        print(f'FOLD {fold}')
        print('---------------------------------------------')
        train_ids_bootstaping = resample(train_ids, replace=True, n_samples=len(train_ids)) # bootstraping
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids_bootstaping)
        validation_subsampler = torch.utils.data.SubsetRandomSampler(validation_ids)
        # train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        # validation_subsampler = torch.utils.data.SubsetRandomSampler(validation_ids)


        train_loader = DataLoader(dataset, batch_size,sampler=train_subsampler)
        validation_loader = DataLoader(dataset, batch_size,sampler=validation_subsampler)

        dataset_sizes = {'train': 0, 'val':0}
        print("\nTraining:")
        for _,labels in train_loader :
            dataset_sizes['train'] += len(labels)
        for _,labels in validation_loader :
            dataset_sizes['val'] += len(labels)
        weights_train, weights_test = weights(train_loader, validation_loader)
        _,accuracies_train, accuracies_test,losses_train, losses_test, weighted_accuracies = train_model(model,train_loader, validation_loader, criterion, optimizer, scheduler, epochs, device, dataset_sizes, num_classes, weights_train, weights_test)
        print(dataset_sizes)

        accuracies.append(weighted_accuracies)
    accuracies_mean = np.mean(accuracies, axis = 0)
    accuracies_std = np.std(accuracies, axis = 0)

    accuracies_mean_nparray = np.array(accuracies_mean)
    accuracies_std_nparray = np.array(accuracies_std)
    # Plot results:
    plt.plot(np.arange(1,epochs+1), accuracies_mean)
    plt.fill_between(np.arange(1,epochs + 1),accuracies_mean_nparray-accuracies_std_nparray,accuracies_mean_nparray+accuracies_std_nparray, color="c")
    #plt.xticks(ticks=max_lengths[:-1] + [65], labels=max_lengths[:-1] + ["None"])
    plt.xlabel("Epochs", fontsize=15)
    plt.ylabel("Validation accuracy", fontsize=15)
    #plt.legend(list(TRANSFORMERS.keys()))
    #plt.title(f"Fine-tuning task (v2) performance", fontsize=20)
    #if os.path.exists(working_directory + f"/images"):
    #    plt.savefig(working_directory + f"/images/fine-tuning_v2.png", dpi=300)
    
    plt.savefig('v{}_model_{}.jpg'.format(version,model_name))
    # plt.show()