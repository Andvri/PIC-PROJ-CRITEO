import torch
from torch.nn import CosineSimilarity

from dataloader import tokenize_data, load_data
from time import time


def train(model, train_loader):
    """
    Compute word embeddings for the input sequences using given pre-trained transformer

    TODO: add actual training pipeline with the resulting mean embeddings for all the categories
    TODO: add cross-validation with weighted accuracy and the loss (cross-entropy)
    """

    model.eval()

    outputs = []
    for batch in train_loader:
        ids, mask = batch[0].to(DEVICE), batch[1].to(DEVICE)

        with torch.no_grad():
            batch_outputs = model(
                ids,
                # token_type_ids=None,
                attention_mask=mask
            )
        outputs.append(batch_outputs[-1])
    
    return torch.cat(outputs, 0)


def evaluate(label_embeddings, labels, sample_embeddings, sample_labels, mean=False):
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

    predicted_categories = torch.tensor([labels[index] for index in predictions])

    # Compute simple (non-weighted) accuracy:
    accuracy = (predicted_categories == sample_labels).sum() / sample_embeddings.shape[0]

    return predictions, predicted_categories, accuracy


# Constants:
MAX_LEN = 20
# NUMBER_OF_SAMPLES = 100
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

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
DATASET = "data_en_train_with_parent.csv"

# Hyperparameters:
BATCH_SIZE = 1
LEARNING_RATE = 1E-4


if __name__ == "__main__":
    import os
    import pandas as pd
    from matplotlib import pyplot as plt
    from torch.utils.data import TensorDataset
    from transformers import AutoModel, AdamW
    
    from dataloader import tokenize_data, load_data
    
    working_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    # Sequence lenghts to vary (careful with the long sequences): 
    max_lengths = list(range(5, 60, 5)) + [None]

    # Make tests, plot results to compare:
    # First tests can be done for only one transformer:
    for transformer_name, (transformer, mean) in TRANSFORMERS.items():
        plt.figure(figsize=(15, 10))

        # Load data:
        print("Loading data...")
        df = pd.read_csv(working_directory + "/sources/" + DATASET, index_col=0)
        df.category = df.category.apply(lambda x: x.replace(" >", ""))
        df.category = df.category.astype("category")

        accuracies = []
        for max_len in max_lengths:
            print("Transforming data...")
            # Create a dataset of category embeddings (see dataloader):
            input_ids, attention_masks, labels, num_categories, unique_categories = tokenize_data(
                df.category.values, df.category,
                max_len=None, tokenizer=transformer
            )
            category_dataset = TensorDataset(input_ids, attention_masks, labels)

            # Create a dataset of desctiption/sample embeddings (see dataloader):
            input_ids, attention_masks, labels, num_categories, unique_categories = tokenize_data(
                df.description.values, df.category,
                max_len=max_len, tokenizer=transformer
            )
            description_dataset = TensorDataset(input_ids, attention_masks, labels)

            # Use pythorch dataloaders:
            train_cat_loader = load_data(category_dataset, BATCH_SIZE, val=False)
            test_description_loader = load_data(description_dataset, BATCH_SIZE, val=False)

            # Download model from huggingface.co:
            model = AutoModel.from_pretrained(
                transformer,
                num_labels=num_categories,
                output_attentions=False,
                output_hidden_states=False
            )
            model.to(DEVICE)

            # Compute embedding vectors:
            cat_embeddings = train(model, train_cat_loader)
            description_embeddings = train(model, test_description_loader)

            # For Funnel and BART - merge output embeddings with the mean:
            if mean:
                cat_embeddings = cat_embeddings.mean(dim=1).squeeze()
                description_embeddings = description_embeddings.mean(dim=1).squeeze()

            # Evaluate:
            predictions, predicted_categories, accuracy = evaluate(
                cat_embeddings, df.category.cat.codes.unique().tolist(),
                description_embeddings, description_dataset.tensors[2]
            )

            accuracies.append(accuracy)

            # Output results:
            print(f"\nPredicted category indices: {predicted_categories}")
            print(f"Accuracy: {accuracy:.5f}")

    # Plot results:
    plt.plot(max_lengths[:-1] + [60], accuracies)
    plt.xlim(5, 60)
    plt.xticks(ticks=max_lengths[:-1] + [60], labels=max_lengths[:-1] + ["None"])
    plt.xlabel("Sequence length", fontsize=15)
    plt.ylabel("Accuracy", fontsize=15)
    plt.legend(list(TRANSFORMERS.keys()))
    plt.title(f"Similarity task (v0) performance", fontsize=20)
    if os.path.exists(working_directory + f"/images"):
        plt.savefig(working_directory + f"/images/similarity_v0.png", dpi=300)
    # plt.show()
