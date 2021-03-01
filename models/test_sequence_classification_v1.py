import torch
from torch.nn import CrossEntropyLoss, Softmax, utils

from transformers import AutoModelForSequenceClassification, AdamW, \
                        get_linear_schedule_with_warmup

from time import time


def train(model, train_loader, scheduler, optimizer, loss_function, checkpoint=1):
    """
    Fine-tune the given transformer with training sequences

    TODO: add cross-validation with weighted accuracy and the loss (cross-entropy)
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
        avg_accuracy += (predictions == labels).sum() / len(predictions)

    avg_loss /= len(test_loader)
    avg_accuracy /= len(test_loader)

    return avg_loss, avg_accuracy, torch.cat(prediction_list, 0)


# Constants:
MAX_LEN = 15
NUMBER_OF_SAMPLES = 100
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

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
DATASET = "data_en_train_with_parent.csv"

# Hyperparameters:
EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 1E-4


if __name__ == "__main__":
    import os
    import pandas as pd
    from matplotlib import pyplot as plt
    from torch.utils.data import TensorDataset
    
    from dataloader import tokenize_data, load_data
    
    working_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    # Sequence lenghts to vary (careful with the long sequences):
    max_lengths = list(range(25, 60, 10)) + [None]

    # Make tests, plot results to compare:
    for transformer_name, (transformer, last_child) in TRANSFORMERS.items():
        plt.figure(figsize=(15, 10))
        
        df = pd.read_csv(DATASET, index_col=0)

        accuracies = []
        for max_len in max_lengths:
            print("Loading data...")
            input_ids, attention_masks, labels, num_categories, unique_categories = tokenize_data(
                df.description.values, df.category.astype("category"),
                max_len=max_len, tokenizer=transformer
            )

            # For instance, validation and training datasets are split manually,
            # but for Cross-Validation it must be done automatically:
            # TODO
            val_dataset = TensorDataset(input_ids[:11], attention_masks[:11], labels[:11])
            train_dataset = TensorDataset(input_ids[11:], attention_masks[11:], labels[11:])

            train_loader = load_data(train_dataset, BATCH_SIZE, val=False)
            validation_loader = load_data(val_dataset, BATCH_SIZE, val=False)

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
                print(f"Child: {idx}")

                if idx < last_child:
                    for param in child.parameters():
                        param.requires_grad = False

            # Define optimizer, loss and scheduler:
            loss_function = CrossEntropyLoss()
            optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=EPOCHS * len(train_loader)
            )

            # train:
            print("\nTraining:")
            start = time()
            for epoch in range(1, EPOCHS + 1):
                avg_loss = train(model, train_loader, scheduler, optimizer, loss_function)
                print(f"Epoch {epoch}: CrossEntropy: {avg_loss}\n")
            print(time() - start)
            
            # evaluate:
            avg_loss, avg_accuracy, predictions = evaluate(model, validation_loader)

            print("\nValidation:\n"
                f"Val CrossEntropy: {avg_loss:.5f}; Val accuracy: {avg_accuracy:.4f}")

            predicted_labels = [unique_categories[idx] for idx in predictions]
            print(predictions)

            accuracies.append(avg_accuracy)

    # Plot results:
    plt.plot(max_lengths[:-1] + [65], accuracies)
    plt.xlim(25, 65)
    plt.xticks(ticks=max_lengths[:-1] + [65], labels=max_lengths[:-1] + ["None"])
    plt.xlabel("Sequence length", fontsize=15)
    plt.ylabel("Validation accuracy", fontsize=15)
    plt.legend(list(TRANSFORMERS.keys()))
    plt.title(f"Fine-tuning task (v2) performance", fontsize=20)
    if os.path.exists(working_directory + f"/images"):
        plt.savefig(working_directory + f"/images/fine-tuning_v2.png", dpi=300)
    # plt.show()
