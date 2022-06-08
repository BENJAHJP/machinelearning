import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd


class AmexData(Dataset):
    def __init__(self, train_csv, label_csv):
        train = pd.read_csv(train_csv)
        label = pd.read_csv(label_csv)

        self.X = torch.tensor(train.values, dtype=torch.float32)
        self.y = torch.tensor(label.values, dtype=torch.int8)
        self.X = self.X.reshape(-1, 1, 8)
        self.y = self.y.reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


dataset = AmexData("/home/b3njah/Documents/Book1.csv", "/home/b3njah/Documents/Book2.csv")

dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)


class BinaryClassification(nn.Module):
    def __init__(self):
        super(BinaryClassification, self).__init__()
        self.bc = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, input_data):
        return self.bc(input_data)


model = BinaryClassification()

# hyper parameters
learning_rate = 0.01
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

# train
epochs = 50
total_steps = len(dataloader)

model.train()

for epoch in range(epochs):
    for i, (features, labels) in enumerate(dataloader):
        prediction = model(features)
        labels = labels.unsqueeze(1)
        loss = criterion(prediction, labels.float())

        # backward propagate
        loss.backward()

        # update weights
        optimizer.step()
        optimizer.zero_grad()

        if (i+1) % 5 == 0:
            print(f"epoch {epoch+1}/{epochs} step {i+1}/{total_steps} loss {loss.item():.4f}")
