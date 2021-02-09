import torch
from torchvision import datasets 

def dataloader(train_dir, test_dir, batch_size, data_transforms):
    train_dataset = datasets.ImageFolder(train_dir,data_transforms['train'])
    test_dataset = datasets.ImageFolder(test_dir,data_transforms['val'])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True, num_workers=4)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=True, num_workers=4)
    dataset_sizes = {'train': len(train_dataset), 'val':len(test_dataset)}
    class_names = train_dataset.classes
    return train_dataloader, test_dataloader, dataset_sizes, class_names