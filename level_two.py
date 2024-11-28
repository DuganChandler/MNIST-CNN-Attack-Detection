import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Hyperparameters
num_clients = 5          # Number of clients
num_rounds = 10          # Number of communication rounds
client_epochs = 1        # Number of local epochs per client
batch_size = 64          # Batch size for training

# Define the ConvNet model
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Input channels changed to 1
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Output: 32 x 14 x 14
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),  # Output: 64 x 12 x 12
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Output: 64 x 6 x 6
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),  # Output: 128 x 4 x 4
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Output: 128 x 2 x 2
        )
        self.fc1 = nn.Linear(128 * 2 * 2, 256)  # Adjusted input features
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        out = self.layer1(x)  # Output size: (batch_size, 32, 14, 14)
        out = self.layer2(out)  # Output size: (batch_size, 64, 6, 6)
        out = self.layer3(out)  # Output size: (batch_size, 128, 2, 2)
        out = out.view(out.size(0), -1)  # Flatten
        out = self.dropout(self.fc1(out))
        out = self.fc2(out)
        return out

# Function to load datasets
def load_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    return train_dataset, test_dataset

# Function to partition dataset among clients
def partition_dataset(dataset, num_clients):
    data_size = len(dataset)
    shard_size = data_size // num_clients
    client_datasets = []
    indices = list(range(data_size))
    np.random.shuffle(indices)
    for i in range(num_clients):
        start = i * shard_size
        end = (i + 1) * shard_size
        client_indices = indices[start:end]
        client_datasets.append(torch.utils.data.Subset(dataset, client_indices))
    return client_datasets

# Function for client-side training
def client_update(client_model, optimizer, train_loader, device, epochs=1):
    client_model.train()
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = client_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    return client_model.state_dict()

# Function to aggregate client models on the server
def server_aggregate(global_model, client_weights):
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        param_list = [client_weights[i][k] for i in range(len(client_weights))]
        param_stack = torch.stack(param_list, dim=0)
        if param_stack.dtype.is_floating_point:
            global_dict[k] = torch.mean(param_stack, dim=0)
        else:
            global_dict[k] = param_stack[0]
    global_model.load_state_dict(global_dict)
    return global_model


# Function to evaluate the model
def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# Federated training function
def federated_training(num_rounds, num_clients, client_epochs, device):
    global_model = ConvNet().to(device)
    train_dataset, test_dataset = load_datasets()
    client_datasets = partition_dataset(train_dataset, num_clients)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for round_num in range(num_rounds):
        print(f"\n--- Round {round_num+1} of {num_rounds} ---")
        client_weights = []
        for client_num in range(num_clients):
            print(f"Client {client_num+1} training...")
            # Create a new model for each client to avoid parameter sharing issues
            client_model = ConvNet().to(device)
            client_model.load_state_dict(global_model.state_dict())
            optimizer = torch.optim.Adam(client_model.parameters(), lr=0.001)
            train_loader = DataLoader(client_datasets[client_num], batch_size=batch_size, shuffle=True)
            # Train the client model
            client_state = client_update(client_model, optimizer, train_loader, device, epochs=client_epochs)
            client_weights.append(client_state)
        # Aggregate client models
        global_model = server_aggregate(global_model, client_weights)
        # Evaluate the global model
        test_accuracy = evaluate(global_model, test_loader, device)
        print(f"Global Model Test Accuracy: {test_accuracy:.2f}%")
    return global_model

# Main function to start federated learning
def main():
    global_model = federated_training(num_rounds, num_clients, client_epochs, device)
    print("\nTraining completed.")
    torch.save(global_model.state_dict(), 'federated_mnist_model.pth')

if __name__ == '__main__':
    main()
