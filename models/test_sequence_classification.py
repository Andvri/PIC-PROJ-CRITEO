import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification


def train(model, train_loader, scheduler, optimizer, loss_function, checkpoint=50):
    model.train()

    avg_loss = 0

    for index, batch in enumerate(1, train_loader):
        ids, labels = batch[0], batch[1]
        # x, y = x.to('cuda'), y.to('cuda')
        
        model.zero_grad() # re-init the gradients (otherwise they are cumulated)

        # Forward pass: Compute predicted y by passing x to the model:
        outputs = model(
            ids,
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

    avg_loss /= train_loader.shape[0]
    
    return avg_loss


def evaluate(model, test_loader):
    model.eval()

    avg_loss = 0
    avg_accuracy = 0
    for batch in test_loader:
        ids, labels = batch[0], batch[1]

        with torch.no_grad():
            outputs = model(
                ids,
                labels=labels
            )
        
        loss = outputs.loss
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1).flatten()

        avg_loss += loss.item()
        avg_accuracy += (predictions == logits).sum() / predictions.shape[0]

    avg_loss /= test_loader.shape[0]
    avg_accuracy /= test_loader.shape[0]

    return avg_loss, avg_accuracy


# Use BERT model as an example:
TRANSFORMER = 'bert-base-uncased'


if __name__ == "__main__":
    test_categories = ["clothes", "home", "auto"]
    test_strs = [
        "This T-shirt is red",
        "We've got some fridge in here",
        "An iconic Ferrari wheel"
    ]
    max_seq_len = 10

    # Load tokenizer:
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER)
    # Download model from huggingface.co:
    model = AutoModelForSequenceClassification.from_pretrained(
        TRANSFORMER,
        num_labels=len(test_categories)
    )
    # print(model.config.output_attentions)

    inputs = [
        tokenizer(
            test_str,
            max_length=max_seq_len,
            pad_to_max_length=True,
            return_tensors="pt"
        ) for test_str in test_strs]
    input_ids = torch.cat([input_["input_ids"] for input_ in inputs], dim=0)

    model.eval()
    labels = torch.tensor([0, 1, 2]).unsqueeze(0)  # Batch size 3
    with torch.no_grad():
        outputs = model(
            input_ids,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1).flatten()

    print(predictions)
    print(loss)
    print(logits)

    for pred in predictions:
        predicted_token = test_categories[pred]
        print(pred, predicted_token)
