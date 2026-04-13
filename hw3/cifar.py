import numpy as np
import torch
from torch import nn
from torch.distributed.elastic.metrics import initialize_metrics
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torchvision.utils import save_image

# Download data
training_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


# Define models
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            #TO DO: fill in this stack
            nn.Linear(32*32*3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        # TO DO: finish this function
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class Convolutional(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(64*(32//4)*(32//4), 10)

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        logits = self.linear(x)
        return logits

class Convolutional2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(128*(32//4)*(32//4), 10)

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        x = self.dropout(x)
        logits = self.linear(x)
        return logits


model_types = ["FF", "Conv", "Conv2"]

def initialize_model(model_type):
    if model_type == 0:
        return FeedForward().to(device)
    elif model_type == 1:
        return Convolutional().to(device)
    elif model_type == 2:
        return Convolutional2().to(device)

# Train models

model_type = -1

model = initialize_model(model_type)



def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


min_epochs = 20
max_epochs = 500
epoch = 0
prev_loss = float("inf")
curr_loss = float("inf")

if 0 <= model_type <= 2:
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    losses = []
    while epoch < max_epochs and (curr_loss <= prev_loss or epoch < min_epochs):
        print(f"-------------------------------\nEpoch {epoch+1}")
        train(train_dataloader, model, loss_fn, optimizer)

        model.eval()
        total_loss = 0

        with torch.no_grad():
            for X, y in train_dataloader:
                X, y = X.to(device), y.to(device)

                pred = model(X)
                loss = loss_fn(pred, y)

                total_loss += loss.item()

        prev_loss = curr_loss
        curr_loss = total_loss / len(train_dataloader)
        print(f"Loss: {curr_loss}")
        losses.append(curr_loss)

        epoch += 1

    plt.plot(range(1, len(losses)+1), losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss over Epochs")
    plt.grid(True)

    torch.save(model.state_dict(), model_types[model_type] + '.pth')
    plt.savefig(model_types[model_type] + "_loss.png")




# Testing

def test(dataloader, model, model_type):
    size = len(dataloader.dataset)
    model.eval()
    correct = 0

    pred_correct = None
    pred_incorrect = None
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()

            for i in range(len(y)):
                if pred_correct is None and pred[i] == y[i]:
                    pred_correct = (X[i], y[i], pred[i])
                if pred_incorrect is None and pred[i] != y[i]:
                    pred_incorrect = (X[i], y[i], pred[i])
                if pred_correct is not None and pred_incorrect is not None:
                    break

    accuracy = correct / size
    print(f"Accuracy: {accuracy}")
    print(f"Correct: Actual: {training_data.classes[pred_correct[1]]}, Predicted: {training_data.classes[pred_correct[2]]}")
    save_image(pred_correct[0], model_types[model_type] + "_correct.png")
    print(f"Incorrect: Actual: {training_data.classes[pred_incorrect[1]]}, Predicted: {training_data.classes[pred_incorrect[2]]}")
    save_image(pred_incorrect[0], model_types[model_type] + "_incorrect.png")



model_type = 0

if 0 <= model_type <= 2:
    model = initialize_model(model_type)
    model.load_state_dict(torch.load(model_types[model_type] + '.pth'))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    test(test_dataloader, model, model_type)