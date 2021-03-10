import numpy as np
import torch
from torch.nn import CrossEntropyLoss, Softmax, utils
from transformers import AutoModelForSequenceClassification, AdamW,get_linear_schedule_with_warmup

from time import time

from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, balanced_accuracy_score
import sys


def train(model, train_loader, scheduler, optimizer, loss_function, checkpoint=1):
    """
    Fine-tune the given transformer with training sequences
    """

    model.train()

    avg_loss = 0

    for index, batch in enumerate(train_loader, 1):
        ids, mask, labels = batch[0].to(DEVICE), batch[1].to(DEVICE), batch[2].to(DEVICE).long()

        model.zero_grad() # re-init the gradients (otherwise they are cumulated)

        # Forward pass: Compute predicted y by passing x to the model:
        outputs = model(
            ids,
            # token_type_ids=None,
            attention_mask=mask,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits

        if not index % checkpoint:
            print(f"Batch {index}, loss: {loss.item()}")

        avg_loss += loss.item()
            
        # Perform a backward pass, and update the weights:
        loss.backward() # perform back-propagation
        
        optimizer.step() # update the weights
        scheduler.step() # update the learning rate

    avg_loss /= len(train_loader)
    
    return avg_loss


def evaluate(model, test_loader):
    """
    Evaluate performance on test data

    TODO: compute weighted accuracy
    """

    model.eval()

    avg_loss = 0
    avg_accuracy = 0
    prediction_list = []

    for batch in test_loader:
        ids, mask, labels = batch[0].to(DEVICE), batch[1].to(DEVICE), batch[2].to(DEVICE).long()

        with torch.no_grad():
            outputs = model(
                ids,
                # token_type_ids=None,
                attention_mask=mask,
                labels=labels
            )
        
        loss = outputs.loss
        logits = outputs.logits.detach().cpu()
        labels = labels.to('cpu')

        predictions = torch.argmax(Softmax(dim=1)(logits), dim=1).flatten()
        prediction_list.append(predictions)

        avg_loss += loss.item()
        # Weighted accuracy:
        avg_accuracy += balanced_accuracy_score(labels, predictions)

        # avg_accuracy += (predictions == labels).sum() / len(predictions)
        # avg_accuracy += f1_score(labels, predictions, average='weighted')

    avg_loss /= len(test_loader)
    avg_accuracy /= len(test_loader)

    return avg_loss, avg_accuracy, torch.cat(prediction_list, 0)


# Constants:
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Transformers to test
# second tuple element defines, how many initial blocks of transformer to freeze:
TRANSFORMERS = {
    "BERT": ('bert-base-uncased', 2),
    "MobileBERT": ('google/mobilebert-uncased', 2),
    "FunnelTransformer": ('funnel-transformer/small-base', 1),
    "RoBERTa": ('roberta-base', 1),
    "BART": ('facebook/bart-base', 1) # bulky
}

# Datasets to test:
DATASET = "data_train_with_parent.csv"

# Hyperparameters:
EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 1E-3

# Number of Cross-Validation splits:
K = 5


if __name__ == "__main__":
    import os
    import re
    import pandas as pd
    from matplotlib import pyplot as plt
    from IPython.display import display
    from torch.utils.data import TensorDataset,DataLoader
    
    from dataloader import tokenize_data, load_data
    from clean_data import clean_descriptions

    working_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    # Sequence lenghts to vary (careful with the long sequences):
    max_lengths = list(range(20, 110, 20))

    # Kfold for cross validation:
    kfold = KFold(n_splits=K, shuffle=False)
    
    # Load and clean data:
    print("Loading data...")
    df = pd.read_csv(working_directory + "/sources/" + DATASET, index_col=0)
    
    df = clean_descriptions(df)
    df = df.sample(frac=1, random_state=10)
    df.reset_index(drop=True, inplace=True)
    display(df)

    # Make tests, plot results to compare:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    for transformer_name, (transformer, last_child) in TRANSFORMERS.items():
        print(f"\nTransformer {transformer_name}")
        
        accuracies = []
        variances = []
        for max_len in max_lengths:
            input_ids, attention_masks, labels, num_categories, unique_categories = tokenize_data(
                df.description.values, df.category.astype("category"),
                max_len=max_len, tokenizer=transformer
            )
            
            dataset = TensorDataset(input_ids, attention_masks, labels)

            crossvalidation_dataset = kfold.split(dataset)
            fold_accuracies = []
            for fold, (train_ids,validation_ids) in enumerate(crossvalidation_dataset):
                # Download model from huggingface.co:
                model = AutoModelForSequenceClassification.from_pretrained(
                    transformer,
                    num_labels=num_categories,
                    output_attentions=False,
                    output_hidden_states=False
                )
                model.to(DEVICE)

                # Freeze the entire model except the last layer:
                # We can either re-train the whole transformer
                # or only the last classification layer:
                for idx, child in enumerate(model.children()):
                    if idx < last_child:
                        for param in child.parameters():
                            param.requires_grad = False

                print(f'FOLD {fold}')
                print('---------------------------------------------')
                train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
                validation_subsampler = torch.utils.data.SubsetRandomSampler(validation_ids)

                train_loader = DataLoader(dataset, BATCH_SIZE, sampler=train_subsampler)
                validation_loader = DataLoader(dataset, BATCH_SIZE, sampler=validation_subsampler)

                # Define optimizer, loss and scheduler:
                loss_function = CrossEntropyLoss()
                optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
                scheduler = get_linear_schedule_with_warmup(
                        optimizer,
                        num_warmup_steps=0,
                        num_training_steps=EPOCHS * len(train_loader)
                )

                start = time()
                # train:
                print("\nTraining:")
                for epoch in range(1, EPOCHS + 1):
                    avg_loss = train(model, train_loader, scheduler, optimizer, loss_function, checkpoint=20)
                    print(f"Epoch {epoch}: CrossEntropy: {avg_loss}\n")
                print(time() - start)
                
                # evaluate:
                avg_loss, avg_accuracy, predictions = evaluate(model, validation_loader)
                print("\nValidation:\n"
                    f"Val CrossEntropy: {avg_loss:.5f}; Val accuracy: {avg_accuracy:.4f}")

                # Save current fold validation accuracy:
                fold_accuracies.append(avg_accuracy)
                predicted_labels = [unique_categories[idx] for idx in predictions]

                print(predictions)
            # Compute and save average accuracy over folds and their variance:
            accuracies.append(sum(fold_accuracies) / K)
            variances.append(np.var(fold_accuracies))

        # Plot results:
        ax1.plot(max_lengths, accuracies, label=transformer_name)
        ax2.plot(max_lengths, variances, label=transformer_name)
        
    for ax in (ax1, ax2):
        ax.set_xlim(20, 100)
        ax.set(xticks=max_lengths, xticklabels=max_lengths)
        ax.set_xlabel("Sequence length", fontsize=15)
    ax1.set_ylabel("CV balanced average accuracy", fontsize=15)
    ax2.set_ylabel("CV balanced accuracies variance", fontsize=15)
    ax1.set_title("CV balanced average accuracy measure", fontsize=20)
    ax2.set_title("CV balanced accuracies variance measure", fontsize=20)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=13)
    if os.path.exists(working_directory + "/images"):
        fig.savefig(working_directory + "/images/fine-tuning_v1.png", dpi=300)
    fig.suptitle('Fine-tuning task (v1) performance', fontsize=20)
    fig.show()
