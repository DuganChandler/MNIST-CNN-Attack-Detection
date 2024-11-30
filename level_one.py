import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

def dataloader(train_dataset, test_dataset):
    batch_size = 64  # Adjusted batch size for MNIST

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False  
    )

    print(f'Training has {len(train_loader)} batches of data.')
    print(f'Testing has {len(test_loader)} batches of data.')
    return train_loader, test_loader


def load_data():
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=True, download=True, transform=transform_train
    )

    test_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=False, download=True, transform=transform_test
    )

    print("Number of training samples:", len(train_dataset))
    print("Number of testing samples:", len(test_dataset))

    return dataloader(train_dataset, test_dataset)


# Building CNN model
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


def main():
    learning_rate = 0.001

    # lower since MNIST can converge pretty quick
    num_epochs = 10 

    train_loader, test_loader = load_data()

    model = ConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_accuracy = 0.0
    best_model_path = 'mnist_centralized_model.pth'

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

        train_accuracy = total_correct / total_samples
        train_loss = total_loss / total_samples

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_correct = 0
            val_total = 0
            for val_features, val_labels in test_loader:
                val_features = val_features.to(device)
                val_labels = val_labels.to(device)
                val_outputs = model(val_features)
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()
            val_accuracy = val_correct / val_total

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(model.state_dict(), best_model_path)
                print(f"Epoch [{epoch+1}/{num_epochs}], "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Train Acc: {train_accuracy:.4f}, "
                      f"Val Acc: {val_accuracy:.4f} **Best Model Saved**")
            else:
                print(f"Epoch [{epoch+1}/{num_epochs}], "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Train Acc: {train_accuracy:.4f}, "
                      f"Val Acc: {val_accuracy:.4f}")

    # Testing phase
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    model.to(device)
    with torch.no_grad():
        correct = 0
        total = 0
        for test_features, test_labels in test_loader:
            test_features = test_features.to(device)
            test_labels = test_labels.to(device)
            outputs = model(test_features)
            _, predicted = torch.max(outputs.data, 1)
            total += test_labels.size(0)
            correct += (predicted == test_labels).sum().item()
        test_accuracy = correct / total
        print(f"Test Accuracy of the model on the 10,000 test images: {100 * test_accuracy:.2f}%")


if __name__ == '__main__':
    main()
