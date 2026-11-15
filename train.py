from torch import no_grad
from torch.utils.data import DataLoader


"""
Functions you should use.
Please avoid importing any other functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
from torch import optim, tensor
from losses import regression_loss, digitclassifier_loss, languageid_loss, digitconvolution_Loss
from torch import movedim


"""
##################
### QUESTION 1 ###
##################
"""


def train_perceptron(model, dataset):
    """
    Train the perceptron until convergence.
    You can iterate through DataLoader in order to 
    retrieve all the batches you need to train on.

    Each sample in the dataloader is in the form {'x': features, 'label': label} where label
    is the item we need to predict based off of its features.
    """
    with no_grad():
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        weights = model.get_weights()

        while True:
            made_mistake = False
            for batch in dataloader:
                features = batch["x"]
                label = batch["label"].item()
                if model.get_prediction(features) != label:
                    weights.add_(features, alpha=label)
                    made_mistake = True
            if not made_mistake:
                return


def train_regression(model, dataset):
    """
    Trains the model.

    In order to create batches, create a DataLoader object and pass in `dataset` as well as your required 
    batch size. You can look at PerceptronModel as a guideline for how you should implement the DataLoader

    Each sample in the dataloader object will be in the form {'x': features, 'label': label} where label
    is the item we need to predict based off of its features.

    Inputs:
        model: Pytorch model to use
        dataset: a PyTorch dataset object containing data to be trained on
        
    """
    dataloader = DataLoader(dataset, batch_size=50, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(300):
        total_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            loss = regression_loss(model(batch['x']), batch['label'])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if total_loss / len(dataloader) < 0.005:
            break


def train_digitclassifier(model, dataset):
    """
    Trains the model.
    """
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_accuracy = -1
    best_state = None
    for epoch in range(20):
        model.train()
        for batch in dataloader:
            optimizer.zero_grad()
            loss = digitclassifier_loss(model(batch["x"]), batch["label"])
            loss.backward()
            optimizer.step()
        model.eval()
        with no_grad():
            accuracy = dataset.get_validation_accuracy()
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        print(
            f"[Q3] Epoch {epoch + 1}/20 | validation = {accuracy:.2%} | best = {best_accuracy:.2%}",
            flush=True,
        )
        if epoch >= 4 and best_accuracy >= 0.98:
            break
    model.load_state_dict(best_state)
    model.eval()


def train_languageid(model, dataset):
    """
    Trains the model.

    Note that when you iterate through dataloader, each batch will returned as its own vector in the form
    (batch_size x length of word x self.num_chars). However, in order to run multiple samples at the same time,
    get_loss() and run() expect each batch to be in the form (length of word x batch_size x self.num_chars), meaning
    that you need to switch the first two dimensions of every batch. This can be done with the movedim() function 
    as follows:

    movedim(input_vector, initial_dimension_position, final_dimension_position)

    For more information, look at the pytorch documentation of torch.movedim()
    """
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_accuracy = -1
    best_state = None
    for epoch in range(40):
        model.train()
        if epoch == 20:
            for group in optimizer.param_groups:
                group["lr"] = 0.0003
        for batch_index, batch in enumerate(dataloader, start=1):
            xs = movedim(batch["x"], 0, 1)
            optimizer.zero_grad()
            loss = languageid_loss(model(xs), batch["label"])
            loss.backward()
            with no_grad():
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.clamp_(-1, 1)
            optimizer.step()
            if batch_index == 1 or batch_index % 50 == 0:
                print(
                    f"[Q4] Epoch {epoch + 1}/40 | batch {batch_index}/{len(dataloader)} | loss = {loss.item():.4f}",
                    flush=True,
                )
        model.eval()
        with no_grad():
            # Use dev explicitly: the backend's validation helper defaults to test.
            correct = 0
            total = 0
            for start, end in dataset.dev_buckets:
                dev_xs, dev_labels = dataset._encode(
                    dataset.dev_x[start:end], dataset.dev_y[start:end]
                )
                predicted = model(dev_xs).argmax(dim=1)
                correct += (predicted == dev_labels.argmax(dim=1)).sum().item()
                total += len(dev_labels)
            accuracy = correct / total
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        print(
            f"[Q4] Epoch {epoch + 1}/40 | dev accuracy = {accuracy:.2%} | best = {best_accuracy:.2%}",
            flush=True,
        )
        if epoch >= 9 and best_accuracy >= 0.85:
            break
    model.load_state_dict(best_state)
    model.eval()



def train_digitconvolution(model, dataset):
    """
    Trains the model.
    """
    model.train()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for _ in range(10):
        for batch in dataloader:
            optimizer.zero_grad()
            predictions = model(batch['x'].float())
            loss = digitconvolution_Loss(predictions, batch['label'].float())
            loss.backward()
            optimizer.step()
