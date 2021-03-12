from train import *
from io_functions import *
def cross_validation(model_name, my_models, lr, momentum, step_size, gamma, num_classes, dataset, batch_size, device, num_epochs = 25, k_folds = 5, weighted_acc = False, version = 1):
    
    kfold = KFold(n_splits=k_folds, shuffle=True)

    train_accuracy = 0
    test_accuracy = 0

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        print('------------',fold,'-------------------')
        #model = deepcopy(my_model)
        model, criterion, optimizer, exp_lr_scheduler, device = initiate_model(model_name, my_models, lr, momentum, step_size, gamma, num_classes, version=2)
        
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        train_dataloader = torch.utils.data.DataLoader(
                      dataset, 
                      batch_size=batch_size, sampler=train_subsampler)
        test_dataloader = torch.utils.data.DataLoader(
                      dataset,
                      batch_size=batch_size, sampler=test_subsampler)
        dataset_sizes = {'train': 0, 'val':0}
        for _,labels in train_dataloader :
            dataset_sizes['train'] += len(labels)
        for _,labels in test_dataloader :
            dataset_sizes['val'] += len(labels)
        weights_train, weights_test = weights(train_dataloader, test_dataloader)
        _,accuracies_train, accuracies_test,losses_train, losses_test, weight_accuracy_train, weight_accuracy_test = train_model(model,train_dataloader, test_dataloader, criterion, optimizer, exp_lr_scheduler, num_epochs, device, dataset_sizes, num_classes, weighted_acc, weights_train, weights_test)

        train_accuracy += accuracies_train[-1].item()
        test_accuracy += accuracies_test[-1].item()
    train_accuracy /= k_folds
    test_accuracy /= k_folds
    print('average train accuracy = ', train_accuracy)
    print('average test accuracy = ', test_accuracy)
    return train_accuracy,test_accuracy