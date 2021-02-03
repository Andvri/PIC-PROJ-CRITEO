import torch
from torch.nn import BCELoss, Softmax, utils

from transformers import AutoModelForSequenceClassification, AdamW, \
                            get_linear_schedule_with_warmup


from dataloader import tokenize_data, load_data


def train(model, train_loader, scheduler, optimizer, loss_function, checkpoint=1):
    model.train()

    avg_loss = 0

    for index, batch in enumerate(train_loader, 1):
        ids, mask, labels = batch[0].to(DEVICE), batch[1].to(DEVICE), batch[2].to(DEVICE)
        
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
        # print(logits)

        if not index % checkpoint:
            print(f"Batch {index}, loss: {loss.item()}")

        avg_loss += loss.item()
            
        # Perform a backward pass, and update the weights:
        loss.backward() # perform back-propagation

        # Gradient clipping
        utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step() # update the weights
    scheduler.step() # update the learning rate

    avg_loss /= len(train_loader)
    
    return avg_loss


def evaluate(model, test_loader):
    # model.eval()

    avg_loss = 0
    avg_accuracy = 0
    prediction_list = []

    for batch in test_loader:
        ids, mask, labels = batch[0].to(DEVICE), batch[1].to(DEVICE), batch[2].to(DEVICE)

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
        print(Softmax(dim=1)(logits))

        avg_loss += loss.item()
        avg_accuracy += (predictions == labels).sum() / len(predictions)

    avg_loss /= len(test_loader)
    avg_accuracy /= len(test_loader)

    return avg_loss, avg_accuracy, torch.cat(prediction_list, 0)


# Constants:
MAX_LEN = 15
NUMBER_OF_SAMPLES = 50
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Transformers to test:
TRANSFORMER = [
    'bert-base-uncased',
    'facebook/bart-large',
    'funnel-transformer/small-base',
    'ctrl',
    'transfo-xl-wt103'
][0]

# Hyperparameters:
EPOCHS = 5
BATCH_SIZE = 5
LEARNING_RATE = .01


if __name__ == "__main__":
    # Load and transform data:
    path = "./cola_public/cola_public/raw/in_domain_train.tsv"
    names = ['sentence_source', 'label', 'label_notes', 'sentence']

    print("Loading data...")
    df, dataset, num_labels, unique_labels = tokenize_data(
        path, NUMBER_OF_SAMPLES,
        names, max_len=MAX_LEN,
        tokenizer=TRANSFORMER
    )
    train_loader, validation_loader = load_data(dataset, BATCH_SIZE)

    # Download model from huggingface.co:
    model = AutoModelForSequenceClassification.from_pretrained(
        TRANSFORMER,
        num_labels=num_labels,
        output_attentions=False,
        output_hidden_states=False
    )
    model.to(DEVICE)

    # Define optimizer, loss and scheduler:
    loss_function = BCELoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=EPOCHS * len(train_loader)
    )

    # train:
    print("\nTraining:")
    for epoch in range(1, EPOCHS + 1):
        avg_loss = train(model, train_loader, scheduler, optimizer, loss_function)
        print(f"Epoch {epoch}: BCE: {avg_loss}\n")
    
    # evaluate:
    avg_loss, avg_accuracy, predictions = evaluate(model, validation_loader)

    print("\nValidation:\n"
          f"Val BCE: {avg_loss}; Val accuracy: {avg_accuracy}")

    predicted_labels = [unique_labels[idx] for idx in predictions]
    print(predictions)
