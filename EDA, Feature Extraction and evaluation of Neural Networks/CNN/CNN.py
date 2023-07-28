import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import numpy as np
from sklearn.metrics import accuracy_score


def cnn_algorithm(x_train, y_train, x_test, y_test):
    # Convert data to tensors
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()

    # Number of classes
    n_classes = len(np.unique(y_train))

    # Create a custom InceptionTime model
    class InceptionTime(nn.Module):
        def __init__(self, in_channels, num_classes):
            super(InceptionTime, self).__init__()
            self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=(9,), stride=(1,), padding=0)
            self.conv2 = nn.Conv1d(32, 32, kernel_size=(9,), stride=(1,), padding=0)
            self.conv3 = nn.Conv1d(32, 32, kernel_size=(9,), stride=(1,), padding=0)
            self.conv4 = nn.Conv1d(32, 32, kernel_size=(9,), stride=(1,), padding=0)
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(32, num_classes)

        def forward(self, x):
            x = f.relu(self.conv1(x))
            x = f.relu(self.conv2(x))
            x = f.relu(self.conv3(x))
            x = f.relu(self.conv4(x))
            x = self.avg_pool(x).squeeze(2)
            x = self.fc(x)
            return x

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Define the model and optimizer
    model = InceptionTime(in_channels=1, num_classes=n_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    num_epochs = 20
    batch_size = 30

    train_dataset = torch.utils.data.TensorDataset(x_train.unsqueeze(1), y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.train()
    epoch_loss_list = []
    epoch_acc_list = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0
        for inputs, labels in train_loader:
            num_batches += 1
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            outputs = np.array([np.argmax(sample) for sample in outputs.detach().numpy()])
            accuracy = accuracy_score(outputs, labels.detach().numpy())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += accuracy
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/num_batches:.4f}, Acc: {epoch_acc/num_batches:.4f}")
        epoch_loss_list.append(epoch_loss/num_batches)
        epoch_acc_list.append(epoch_acc/num_batches)

    # Evaluate the model on the test data
    model.eval()

    with torch.no_grad():
        outputs = model(x_test.unsqueeze(1))
        _, y_pred = torch.max(outputs.data, 1)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy on test data: {:.2f}%".format(accuracy * 100))

    return y_pred, epoch_loss_list, epoch_acc_list, num_epochs
