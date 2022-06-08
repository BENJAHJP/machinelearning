import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

df = pd.read_csv("/home/b3njah/Downloads/datasets/spaceship-titanic/train.csv")
df = df.drop(['Cabin'], axis=1)
df1 = df.dropna(axis=0)

HomePlanet_dummies = pd.get_dummies(df1['HomePlanet'])
CryoSleep = df1['CryoSleep'].apply(lambda x: 0 if x is False else 1)
Destination_dummies = pd.get_dummies(df1['Destination'])
VIP = df1['VIP'].apply(lambda x: 0 if x is False else 1)
Transported = df1['Transported'].apply(lambda x: 0 if x is False else 1)

print(Destination_dummies)
df1 = pd.concat([HomePlanet_dummies, CryoSleep, Destination_dummies, VIP, Transported, df1['Age'],
                 df1['RoomService'], df1['FoodCourt'], df1['ShoppingMall'], df1['Spa'], df1['VRDeck']], axis=1)

y = df1['Transported']
X = df1.drop(['Transported'], axis=1)

y = torch.tensor(y.values, dtype=torch.float32)
X = torch.tensor(X.values, dtype=torch.float32)


class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.l1 = nn.Linear(14, 20)
        self.B1 = nn.BatchNorm1d(20)
        self.l2 = nn.Linear(20, 40)
        self.B2 = nn.BatchNorm1d(40)
        self.l3 = nn.Linear(40, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.B1(x)
        x = F.relu(self.l2(x))
        x = self.B2(x)
        x = F.relu(self.l3(x))
        return x


model = BinaryClassifier()

# Hyper parameters
learning_rate = 0.01
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(lr=learning_rate, params=model.parameters())

# train
epochs = 100
model.train()

for epoch in range(epochs):
    prediction = model(X)
    y = y.reshape(-1, 1)
    loss = criterion(prediction, y)

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    print(f"epoch {epoch +1} loss {loss.item():.4f}")