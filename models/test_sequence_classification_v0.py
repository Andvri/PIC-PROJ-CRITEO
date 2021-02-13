import torch
from torch.nn import CosineSimilarity

from dataloader import tokenize_data, load_data
from time import time


def train(model, train_loader):
    model.eval()

    outputs = []
    for batch in train_loader:
        ids, mask = batch[0].to(DEVICE), batch[1].to(DEVICE)

        with torch.no_grad():
            batch_outputs = model(
                ids,
                token_type_ids=None,
                attention_mask=mask
            )
        outputs.append(batch_outputs[-1])
    
    return torch.cat(outputs, 0)


def evaluate(label_embeddings, labels, sample_embeddings, sample_labels):
    cos = CosineSimilarity(dim=1)

    similarities = []
    for sample in sample_embeddings:
        sample_sims = cos(
            torch.vstack([sample for _ in range(label_embeddings.shape[0])]), # explode given tensor
            label_embeddings
        )
        similarities.append(sample_sims)
    similarities = torch.vstack(similarities)
    predictions = torch.argmax(similarities, dim=1).flatten()

    predicted_categories = torch.tensor([labels[index] for index in predictions])

    accuracy = (predicted_categories == sample_labels).sum() / sample_embeddings.shape[0]

    return predictions, predicted_categories, accuracy


# Constants:
MAX_LEN = None
# NUMBER_OF_SAMPLES = 100
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

# Transformers to test:
TRANSFORMER = [
    'bert-base-uncased',
    'google/mobilebert-uncased',
    'funnel-transformer/small-base',
    'roberta-base',
    'facebook/bart-base', # bulky
][0]

# Hyperparameters:
BATCH_SIZE = 1
LEARNING_RATE = 1E-4


if __name__ == "__main__":
    import os
    import pandas as pd
    from transformers import AutoModel, AdamW
    
    from dataloader import tokenize_data, load_data
    
    working_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    # Load data:
    print("Loading data...")
    df = pd.read_csv(working_directory + "/sources/data_with_parent.csv", index_col=0)
    df.category = df.category.apply(lambda x: x.replace(" >", ""))
    df.category = df.category.astype("category")
    print(df)

    print("Transforming data...")
    category_dataset, num_categories, unique_categories = tokenize_data(
        df.category.values, df.category,
        max_len=MAX_LEN, tokenizer=TRANSFORMER
    )

    description_dataset, num_categories, unique_categories = tokenize_data(
        df.description.values, df.category,
        max_len=MAX_LEN, tokenizer=TRANSFORMER
    )

    train_cat_loader = load_data(category_dataset, BATCH_SIZE, val=False)
    test_description_loader = load_data(description_dataset, BATCH_SIZE, val=False)

    # Download model from huggingface.co:
    model = AutoModel.from_pretrained(
        TRANSFORMER,
        num_labels=num_categories,
        output_attentions=False,
        output_hidden_states=False
    )
    model.to(DEVICE)

    cat_embeddings = train(model, train_cat_loader)

    description_embeddings = train(model, test_description_loader)

    predictions, predicted_categories, accuracy = evaluate(
        cat_embeddings, df.category.cat.codes.unique().tolist(),
        description_embeddings, description_dataset.tensors[2]
    )

    # Output results:
    print()
    print(f"Predicted category indices: {predicted_categories}")
    print(f"Accuracy: {accuracy:.5f}")

    df["predictions"] = [unique_categories[index] for index in predictions.numpy()]
    print(df)
