from transformers import BertTokenizer
<<<<<<< HEAD
=======
import tensorflow as tf
>>>>>>> initial dataloader
import torch
import pandas as pd
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

batch_size = 32

def tokenize_data(path, names, delimiter='\t'):
    #df = pd.read_csv(path, delimiter=delimiter, header=None, names=names)
    df = pd.read_csv(path, delimiter=delimiter, header=None, names=names)[0:100]
    #print(df.head())
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    if "sentence" in names and "label" in names:
        sentences = df.sentence.values
        labels = df.label.values
    else :
        print("EXIT")
        return None,None
    num_labels = len(set(labels))
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
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = max_len,  
                            truncation=True,         
                            pad_to_max_length = True,   # Pad & truncate all sentences.
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                    )
        
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
        
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)

<<<<<<< HEAD
    return dataset, num_labels


=======
    return dataset,num_labels
>>>>>>> initial dataloader
def load_data(batch_size):
    # Create a 90-10 train-validation split.

    # Calculate the number of samples to include in each set.
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))
    train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )
    return train_dataloader,validation_dataloader


<<<<<<< HEAD
# path = "./.data/yelp_review_full_csv/train.csv"
# path = "cola_public/cola_public/raw/in_domain_train.tsv"
# names=['label', 'sentence']
# dataset,num_labels = tokenize_data(path, names, delimiter=',')

path = "./cola_public/cola_public/raw/in_domain_train.tsv"
names = ['sentence_source', 'label', 'label_notes', 'sentence']
dataset, num_labels = tokenize_data(path, names)

train_loader, validation_loader = load_data(batch_size)

    
print('number of labels = ', num_labels)
print('Number of train batches = ', len(train_loader))
=======
path = "./.data/yelp_review_full_csv/train.csv"
names=['label', 'sentence']
dataset,num_labels = tokenize_data(path, names, delimiter=',')
'''
path = "./cola_public/cola_public/raw/in_domain_train.tsv"
names = ['sentence_source', 'label', 'label_notes', 'sentence']
dataset,num_labels = tokenize_data(path, names)
'''
train_loader,validation_loader = load_data(batch_size)

    
print('number of labels = ', num_labels)
print('Number of train batches = ',len(train_loader))
>>>>>>> initial dataloader
print('Batch size =', batch_size)
