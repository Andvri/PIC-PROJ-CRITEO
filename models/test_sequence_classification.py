import torch
from torch.nn import BCELoss, Softmax
from torch.optim import Adam, lr_scheduler

from transformers import AutoConfig, AutoModelForSequenceClassification, \
                                get_linear_schedule_with_warmup

from dataloader import tokenize_data, load_data

from IPython.display import display


def train(model, train_loader, scheduler, optimizer, loss_function, checkpoint=1):
    model.train()

    avg_loss = 0

    for index, batch in enumerate(train_loader, 1):
        ids, mask, labels = batch[0], batch[1], batch[2]
        # x, y = x.to('cuda'), y.to('cuda')
        
        model.zero_grad() # re-init the gradients (otherwise they are cumulated)

        # Forward pass: Compute predicted y by passing x to the model:
        outputs = model(
            ids,
            # token_type_ids=None,
            # attention_mask=mask, 
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
    model.eval()

    avg_loss = 0
    avg_accuracy = 0
    prediction_list = []

    for batch in test_loader:
        ids, mask, labels = batch[0], batch[1], batch[2]

        with torch.no_grad():
            outputs = model(
                ids,
                # token_type_ids=None,
                # attention_mask=mask,
                labels=labels
            )
        
        loss = outputs.loss
        logits = outputs.logits
        softmax = Softmax(dim=1)
        predictions = torch.argmax(softmax(logits), dim=1).flatten()
        prediction_list.append(predictions)
        print(softmax(logits))

        avg_loss += loss.item()
        avg_accuracy += (predictions == labels).sum() / len(predictions)

    avg_loss /= len(test_loader)
    avg_accuracy /= len(test_loader)

    return avg_loss, avg_accuracy, torch.cat(prediction_list, 0)


# Use BERT model as an example:
TRANSFORMER = 'bert-base-uncased'
EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = .01


if __name__ == "__main__":
    # test_categories = ["clothes", "home", "auto"]
    # test_strs = [
    #     "This T-shirt is red",
    #     "We've got some fridge in here",
    #     "An iconic Ferrari wheel"
    # ]
    # max_seq_len = 10

    # Load tokenizer:
    # tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER)
    # Download model from huggingface.co:
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     TRANSFORMER,
    #     num_labels=len(test_categories)
    # )
    # print(model.config.output_attentions)

    # inputs = [
    #     tokenizer(
    #         test_str,
    #         max_length=max_seq_len,
    #         pad_to_max_length=True,
    #         return_tensors="pt"
    #     ) for test_str in test_strs]
    # input_ids = torch.cat([input_["input_ids"] for input_ in inputs], dim=0)

    # model.eval()
    # labels = torch.tensor([0, 1, 2]).unsqueeze(0)  # Batch size 3
    # with torch.no_grad():
    #     outputs = model(
    #         input_ids,
    #         labels=labels
    #     )
        
    #     loss = outputs.loss
    #     logits = outputs.logits
    #     predictions = torch.argmax(logits, dim=1).flatten()

    # print(predictions)
    # print(loss)
    # print(logits)

    # for pred in predictions:
    #     predicted_token = test_categories[pred]
    #     print(pred, predicted_token)

    # Load and transform data:
    path = "./cola_public/cola_public/raw/in_domain_train.tsv"
    names = ['sentence_source', 'label', 'label_notes', 'sentence']

    print("Loading data...")
    df, dataset, num_labels, unique_labels = tokenize_data(path, names)
    train_loader, validation_loader = load_data(dataset, BATCH_SIZE)

    # Download model from huggingface.co:
    model = AutoModelForSequenceClassification.from_pretrained(
        TRANSFORMER,
        output_attentions=True,
        num_labels=num_labels
    )

    # Define optimizer, loss and scheduler:
    loss_function = BCELoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=EPOCHS * len(train_loader)
    )

    # train:
    print("\nTraining:")
    for epoch in range(1, EPOCHS + 1):
        avg_loss = train(model, train_loader, scheduler, optimizer, loss_function)
        print(f"Epoch {epoch}: cross entropy: {avg_loss}\n")
    
    # evaluate:
    avg_loss, avg_accuracy, predictions = evaluate(model, validation_loader)

    print("\nValidation:\n"
          f"Val cross entropy: {avg_loss}; Val accuracy: {avg_accuracy}")

    predicted_labels = [unique_labels[idx] for idx in predictions]
    print(predictions)
