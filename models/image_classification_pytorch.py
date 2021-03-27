from cross_validation import *
from train import *
#data_dir = 'data/hymenoptera_data'
data_dir = 'data/google taxonomy'
train_dir = data_dir + '/train'
test_dir = data_dir + '/val'

cross_val = False
k_folds = 5

weighted_acc = False

version = 2 #1 to retrain all layers, 2 to retrain the last one

batch_size = 4
epochs = 25
lr = 0.001
momentum = 0.9

step_size = 7
gamma = 0.1

model_name = "alexnet"
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

def main():
    if cross_val :
        print("CROSS VALIDATION with ", k_folds," folds")
        train_dataset, test_dataset, dataset_sizes, class_names = dataloader(train_dir, test_dir, batch_size=None, data_transforms=data_transforms)
    else :
        train_dataloader, test_dataloader, dataset_sizes, class_names, weights_train, weights_test = dataloader(train_dir, test_dir, batch_size, data_transforms)

    num_classes = len(class_names)
    model, criterion, optimizer, exp_lr_scheduler, device = initiate_model(model_name, my_models, lr, momentum, step_size, gamma, num_classes, version=2)

    if cross_val:
        dataset = ConcatDataset([train_dataset, test_dataset])
        train_accuracy,test_accuracy = cross_validation(model_name, my_models, lr, momentum, step_size, gamma, num_classes, dataset, batch_size, device, num_epochs = epochs, k_folds = 5, weighted_acc = weighted_acc, version = version)
    else :
        model,accuracies_train, accuracies_test,losses_train, losses_test, weight_accuracy_train, weight_accuracy_test = train_model(model,train_dataloader, test_dataloader, criterion, optimizer, exp_lr_scheduler, epochs, device, dataset_sizes, num_classes, weighted_acc, weights_train, weights_test)
        saveFig(model_name, 'accuracy', accuracies_train, accuracies_test, version)
        saveFig(model_name, 'loss', losses_train, losses_test, version)
if __name__=='__main__':
    main()
