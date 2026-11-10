from torch import no_grad, movedim
from torch.utils.data import DataLoader
from models import PerceptronModel, RegressionModel, DigitClassificationModel, LanguageIDModel, DigitConvolutionalModel
from losses import regression_loss, digitclassifier_loss, languageid_loss, digitconv_loss
from torch import optim

def train_perceptron(model, dataset):
    with no_grad():
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        "*** YOUR CODE HERE ***"
        while True:
            mistakes = 0
            for batch in dataloader:
                x = batch['x']
                y = batch['label']
                if model.get_prediction(x) != y.item():
                    model.w += x * y.item()
                    mistakes += 1
            if mistakes == 0:
                break

def train_regression(model, dataset):
    "*** YOUR CODE HERE ***"
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    while True:
        total_loss = 0.0
        count = 0
        for batch in dataloader:
            optimizer.zero_grad()
            loss = regression_loss(model(batch['x']), batch['label'])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            count += 1
        if (total_loss / count) <= 0.012:
            break

def train_digitclassifier(model, dataset):
    model.train()
    "*** YOUR CODE HERE ***"
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    while True:
        for batch in dataloader:
            optimizer.zero_grad()
            loss = digitclassifier_loss(model(batch['x']), batch['label'])
            loss.backward()
            optimizer.step()
        if dataset.get_validation_accuracy() >= 0.976:
            break

def train_languageid(model, dataset):
    model.train()
    "*** YOUR CODE HERE ***"
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    while True:
        for batch in dataloader:
            optimizer.zero_grad()
            x = movedim(batch['x'], 1, 0)
            loss = languageid_loss(model(x), batch['label'])
            loss.backward()
            optimizer.step()
        if dataset.get_validation_accuracy() >= 0.825:
            break

def train_digitconv(model, dataset):
    "*** YOUR CODE HERE ***"
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    while True:
        for batch in dataloader:
            optimizer.zero_grad()
            loss = digitconv_loss(model(batch['x']), batch['label'])
            loss.backward()
            optimizer.step()
        if dataset.get_validation_accuracy() >= 0.82:
            break
