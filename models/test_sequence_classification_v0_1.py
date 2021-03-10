import torch
from torch.nn import CosineSimilarity

from time import time

from sklearn.model_selection import KFold
from sklearn.metrics import balanced_accuracy_score


def train(model, train_loader):
    """
    Compute word embeddings for the input sequences using given pre-trained transformer
    """

    model.eval()

    outputs = []
    labels_list = []
    for batch in train_loader:
        ids, mask, labels = batch[0].to(DEVICE), batch[1].to(DEVICE), batch[2].to(DEVICE).long()

        with torch.no_grad():
            batch_outputs = model(
                ids,
                # token_type_ids=None,
                attention_mask=mask
            )
        outputs.append(batch_outputs[-1])
        labels_list.extend(labels.tolist())
    
    return torch.cat(outputs, 0), torch.tensor(labels_list)


def evaluate(label_embeddings, sample_embeddings, sample_labels):
    """
    Compute the cosine similarities with all the label_embeddings
    for each sequence embedding vector from sample_embeddings
    and take the argmax to predict sequence's category
    """
    cos = CosineSimilarity(dim=1)

    similarities = []
    # For every sequence vector in sample_embeddings:    
    for sample in sample_embeddings:
        # Compute the similarities with all the label_embeddings:
        sample_sims = cos(
            torch.vstack([sample for _ in range(label_embeddings.shape[0])]), # explode given tensor
            label_embeddings
        )
        similarities.append(sample_sims)
    # Stack similarities as one tensor:
    similarities = torch.vstack(similarities)
    # Make argmax predictions:
    predictions = torch.argmax(similarities, dim=1).flatten()
    predictions = predictions.detach().cpu()

    # Weighted accuracy:
    accuracy = balanced_accuracy_score(sample_labels, predictions)

    # Compute simple (non-weighted) accuracy:
    # accuracy = (predictions == sample_labels).sum() / sample_embeddings.shape[0]

    return predictions, accuracy


# Constants:
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")

# Transformers to test:
# The second element of the tuple defines the type of transformer's output:
# If False - transformer outputs only 1 embedding for each input sequence,
# if True - transformer outputs embeddings for each sequence token, which will be aggregated:
TRANSFORMERS = {
    "BERT": ('bert-base-uncased', False),
    "MobileBERT": ('google/mobilebert-uncased', False),
    "FunnelTransformer": ('funnel-transformer/small-base', True),
    "RoBERTa": ('roberta-base', False),
    "BART": ('facebook/bart-base', True), # bulky
}

# Dataset to test:
DATASET = "data_train_with_parent.csv"

# Hyperparameters:
BATCH_SIZE = 64

# Number of Cross-Validation splits:
K = 5


if __name__ == "__main__":
    import os
    import re
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from torch.utils.data import TensorDataset, random_split
    from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
    from transformers import AutoModel, AdamW
    from IPython.display import display
    
    from dataloader import tokenize_data, load_data
    from clean_data import clean_descriptions
    
    working_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    # Kfold for cross validation:
    kfold = KFold(n_splits=K, shuffle=False)

     # Sequence lenghts to vary (careful with the long sequences):
    max_lengths = list(range(20, 110, 20))

    # Load and clean data:
    print("Loading data...")
    df = pd.read_csv(working_directory + "/sources/" + DATASET, index_col=0)
    
    df = clean_descriptions(df)
    df = df.sample(frac=1, random_state=10)
    df.reset_index(drop=True, inplace=True)
    display(df)

    # Make tests, plot results to compare:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    for transformer_name, (transformer, mean) in TRANSFORMERS.items():
        print(f"\nTransformer {transformer_name}")
        
        accuracies = []
        variances = []
        for max_len in max_lengths:
            print("Training category embeddings...")
            input_ids, attention_masks, labels, num_categories, unique_categories = tokenize_data(
                df.description.values, df.category.astype("category").cat.codes,
                max_len=max_len, tokenizer=transformer
            )
            
            # Download model from huggingface.co:
            model = AutoModel.from_pretrained(
                transformer,
                num_labels=num_categories,
                output_attentions=False,
                output_hidden_states=False
            )
            model.to(DEVICE)
            
            dataset = TensorDataset(input_ids, attention_masks, labels)

            crossvalidation_dataset = kfold.split(dataset)
            fold_accuracies = []
            for fold, (train_ids,validation_ids) in enumerate(crossvalidation_dataset):
                # Compute categories embeddings:
                print(f'FOLD {fold}')
                print('---------------------------------------------')
                train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
                validation_subsampler = torch.utils.data.SubsetRandomSampler(validation_ids)

                train_loader = DataLoader(dataset, BATCH_SIZE, sampler=train_subsampler)
                validation_loader = DataLoader(dataset, BATCH_SIZE, sampler=validation_subsampler)

                start = time()
                # train:
                print("\nTraining:")
                # Compute embedding vectors:
                embeddings, cat_labels = train(model, train_loader)
                validation_embeddings, validation_labels = train(model, validation_loader)
                
                # For Funnel and BART - merge output embeddings with the mean:
                if mean:
                    embeddings = embeddings.mean(dim=1).squeeze()
                    validation_embeddings = validation_embeddings.mean(dim=1).squeeze()
                
                # Compute category embeddings:
                cat_embeddings = []
                for label in cat_labels.unique():
                    mean_embedding = embeddings[cat_labels == label].mean(dim=0)
                    cat_embeddings.append(mean_embedding)
                cat_embeddings = torch.vstack(cat_embeddings)
                print(time() - start)
                
                # Evaluate:
                predictions, accuracy = evaluate(
                    cat_embeddings, validation_embeddings, validation_labels
                )
                print("\nValidation:\n"
                    f"Val accuracy: {accuracy:.4f}")

                # Save current fold validation accuracy:
                fold_accuracies.append(accuracy)
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
        fig.savefig(working_directory + "/images/fine-tuning_v0_1.png", dpi=300)
    fig.suptitle('Similarity task (v0_1) performance', fontsize=20)
    fig.show()
