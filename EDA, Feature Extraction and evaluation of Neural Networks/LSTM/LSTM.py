import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score


def lstm_algorithm(x_train, y_train, x_test, y_test):
    # Reshape the input data to match the expected dimensions
    x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
    x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])

    # Convert data to tensors
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()

    # Input size and number of classes
    in_size = x_train.shape[2]
    n_classes = len(np.unique(y_train))

    # Create a custom LSTM model
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes, num_layers=1, bidirectional=False):
            super(LSTMModel, self).__init__()
            self.num_layers = num_layers
            self.hidden_size = hidden_size
            self.bidirectional = bidirectional
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(x.device)
            out, _ = self.lstm(x, (h0.view(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).contiguous(),
                                   c0.view(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).contiguous()))
            out = out[:, -1, :]
            out = self.fc(out)
            return out

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Define the model and optimizer
    model = LSTMModel(input_size=in_size, hidden_size=64, num_classes=n_classes, num_layers=1, bidirectional=False)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    num_epochs = 20
    batch_size = 30

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
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
        outputs = model(x_test)
        _, y_pred = torch.max(outputs.data, 1)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy on test data: {:.2f}%".format(accuracy * 100))

    return y_pred, epoch_loss_list, epoch_acc_list, num_epochs
