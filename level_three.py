import torch 
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np

# device config 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# hyper params
num_clients = 10
num_rounds = 10
client_epochs = 2
batch_size = 64

# CNN model
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

# client training
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

# aggregate func for combining clients
def server_aggregate(global_model, client_weights):
    """
    Coordinate-wise median aggregation for robustness against malicious clients.
    """
    global_dict = global_model.state_dict()
    num_clients = len(client_weights)

    for k in global_dict.keys():
        param_list = torch.stack([client_weights[i][k].flatten() for i in range(num_clients)], dim=0)

        if param_list.dtype.is_floating_point:
            median_param = torch.median(param_list, dim=0)[0]
            global_dict[k] = median_param.view(global_dict[k].shape)
        else:
            global_dict[k] = param_list[0].view(global_dict[k].shape)

    global_model.load_state_dict(global_dict)
    
    return global_model

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

def detect_malicious_clients(client_weights):
    num_clients = len(client_weights)

    # compute median model
    global_state = {}
    for key in client_weights[0].keys():
        param_list = torch.stack([client_weights[i][key].flatten() for i in range(num_clients)], dim=0)
        if param_list.dtype.is_floating_point:
            median_param = torch.median(param_list, dim=0)[0]
            global_state[key] = median_param.view(client_weights[0][key].shape)
        else:
            global_state[key] = param_list[0].view(client_weights[0][key].shape)

    # compute dist of each client from median model
    distances = []
    for i in range(num_clients):
        distance = 0.0
        for key in client_weights[0].keys():
            if client_weights[i][key].dtype.is_floating_point:
                diff = (client_weights[i][key] - global_state[key]).view(-1)
                distance += torch.norm(diff).item()
        distances.append(distance)

    # detect clients with distances significantly higher than others
    distances = np.array(distances)
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    threshold = mean_distance + 2 * std_distance
    detected_malicious_clients = np.where(distances > threshold)[0]

    return detected_malicious_clients

# training with simulated singular malicious client
def federated_training(num_rounds, num_clients, client_epochs, device):
    global_model = ConvNet().to(device)
    train_dataset, test_dataset = load_datasets()
    client_datasets = partition_dataset(train_dataset, num_clients)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    malicious_client = 0
    malicious_round = 5

    permanently_excluded_clients = set()

    for round_num in range(num_rounds):
        print(f"\n--- Round {round_num+1} of {num_rounds} ---")
        client_weights = []
        participating_clients = []
        for client_num in range(num_clients):
            # skip detected malicious client
            if client_num in permanently_excluded_clients:
                print(f"Client {client_num+1} is permanently excluded from aggregation.")
                continue

            print(f"Client {client_num+1} training...")
            client_model = ConvNet().to(device)
            client_model.load_state_dict(global_model.state_dict())
            optimizer = torch.optim.Adam(client_model.parameters(), lr=0.001)
            train_loader = DataLoader(client_datasets[client_num], batch_size=batch_size, shuffle=True)
            client_state = client_update(client_model, optimizer, train_loader, device, epochs=client_epochs)

            # simulate malicious client attack w/random weights 
            if client_num == malicious_client and round_num >= malicious_round:
                print(f"Client {client_num+1} is malicious from round {malicious_round+1} onward.")
                for key in client_state:
                    if client_state[key].dtype.is_floating_point:
                        client_state[key] = torch.randn_like(client_state[key])

            client_weights.append(client_state)
            participating_clients.append(client_num)

        detected_malicious_clients = detect_malicious_clients(client_weights)

        detected_malicious_clients = [participating_clients[i] for i in detected_malicious_clients]
        print(f"Malicious clients detected in this round: {detected_malicious_clients}")

        permanently_excluded_clients.update(detected_malicious_clients)

        client_weights_filtered = [client_weights[i] for i in range(len(client_weights)) if participating_clients[i] not in detected_malicious_clients]

        if len(client_weights_filtered) == 0:
            print("All clients detected as malicious. Skipping aggregation.")
            continue

        global_model = server_aggregate(global_model, client_weights_filtered)

        test_accuracy = evaluate(global_model, test_loader, device)
        print(f"Global Model Test Accuracy: {test_accuracy:.2f}%")
    return global_model

def main():
    global_model = federated_training(num_rounds, num_clients, client_epochs, device)
    print("\nTraining completed.")
    torch.save(global_model.state_dict(), 'federated_mnist_malicious_client_model.pth')

if __name__ == '__main__':
    main()
