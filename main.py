import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Definitions
BATCH_SIZE = 64
INPUT_SIZE = 784
HIDDEN_SIZE = 64
OUTPUT_SIZE = 10
LEARNING_RATE = 0.005
EPOCHS = 5


# Neural Network
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x


# Datasets
mnist_train = MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
mnist_test = MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

# Network instantiation
model = NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), LEARNING_RATE)

# Training loop
mnist_dataloader = DataLoader(mnist_train, batch_size=BATCH_SIZE, drop_last=True)

for epoch in range(EPOCHS):
    total_loss = 0
    for image, labels in mnist_dataloader:
        prediction = model(image)
        loss = criterion(prediction, labels)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1} loss: {total_loss / (len(mnist_train) / BATCH_SIZE)}")

# Testing loop
mnist_dataloader = DataLoader(mnist_test, batch_size=1, drop_last=True)

with torch.no_grad():
    total_correct = 0
    for image, label in mnist_dataloader:
        prediction = model(image)
        index = torch.argmax(prediction)
        if index == label.item():
            total_correct += 1
    print(f"Percentage of correct answers: {((total_correct*100)/len(mnist_test)):.2f}%")
