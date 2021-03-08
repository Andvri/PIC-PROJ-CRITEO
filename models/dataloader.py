from transformers import AutoTokenizer

import pandas as pd

import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

BATCH_SIZE = 32
NUMBER_OF_SAMPLES = 100
MAX_LEN = 20

def tokenize_data(sentences, categories, tokenizer='bert-base-uncased', max_len=None):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer,
        do_lower_case=True
    )

    unique_categories = categories.unique().tolist()
    num_categories = len(unique_categories)
    labels = categories.cat.codes

    if not max_len:
        max_len = 0
        # For every sentence...
        for sent in sentences:
            # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
            input_ids = tokenizer.encode(sent, add_special_tokens=True)
            # Update the maximum sentence length.
            max_len = max(max_len, len(input_ids))
    print('Max sentence length: ', max_len)

    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in sentences:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
            sent, # Sentence to encode.
            add_special_tokens=True, # Add '[CLS]' and '[SEP]'
            max_length=max_len,
            truncation=True,
            padding='max_length', # Pad & truncate all sentences.
            return_attention_mask=True, # Construct attn. masks.
            return_tensors='pt' # Return pytorch tensors.
        )
        
        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])
        
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    
    return input_ids, attention_masks, labels, num_categories, unique_categories


def load_data(train_dataset, batch_size, val=True):
    if val:
        # Create a 90-10 train-validation split.

        # Calculate the number of samples to include in each set.
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size

        # Divide the dataset by randomly selecting samples.
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

        print('{} training samples'.format(train_size))
        print('{} validation samples'.format(val_size))

        # For validation the order doesn't matter, so we'll just read them sequentially.
        validation_dataloader = DataLoader(
                val_dataset, # The validation samples.
                # sampler=SequentialSampler(val_dataset), # Pull out batches sequentially.
                batch_size=batch_size # Evaluate with this batch size.
            )
    else:
        print('{} training samples'.format(len(train_dataset)))
    
    train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            # sampler=RandomSampler(train_dataset), # Select batches randomly
            batch_size=batch_size # Trains with this batch size.
        )

    if val:
        return train_dataloader, validation_dataloader
    else:
        return train_dataloader


if __name__ == "__main__":
    # path = "./.data/yelp_review_full_csv/train.csv"
    # path = "cola_public/cola_public/raw/in_domain_train.tsv"
    # names=['label', 'sentence']
    # dataset,num_labels = tokenize_data(path, names, delimiter=',')

    path = "./cola_public/cola_public/raw/in_domain_train.tsv"
    names = ['sentence_source', 'label', 'label_notes', 'sentence']

    df, dataset, num_labels, unique_labels = tokenize_data(path, NUMBER_OF_SAMPLES,
                                                            names, max_len=MAX_LEN)
    train_loader, validation_loader = load_data(dataset, BATCH_SIZE)

    print('number of labels = ', num_labels)
    print(f"Unique labels: {unique_labels}")
    print('Number of train batches = ', len(train_loader))
    print('Batch size =', BATCH_SIZE)
