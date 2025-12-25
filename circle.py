import torch
import sklearn.datasets
import torch.nn as nn
import torch.optim as optim

n_circles = 1000
noise = 0.3
X, y = sklearn.datasets.make_circles(n_samples=n_circles, noise=noise)
print(X, "\n")
print(y)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y)

print(X, y)

X_train = X[:800]
y_train = y[:800]

X_val = X[800:]
y_val = y[800:]

print(X_train.shape, X_val.shape)

class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=2, out_features=10)
        self.layer2 = nn.Linear(in_features=10, out_features=10)
        self.layer3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

model = CircleModelV0()
loss = nn.BCEWithLogitsLoss()
optimiser = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(1000):
    model.train()
    y_hat = model.forward(X_train).squeeze()
    lost = loss(y_hat, y_train)
    
    optimiser.zero_grad()
    loss.backward()

    optimiser.step()

    # Affichage tous les 100 tours
    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Loss: {lost.item():.5f}")


model.eval()
with torch.no_grad():
    test_logits = model(X_val).squeeze()
    # Si le score > 0, on dit que c'est la classe 1, sinon classe 0
    predictions = torch.round(torch.sigmoid(test_logits))
    # On calcule le % de bonnes réponses
    accuracy = (predictions == y_val).sum().item() / len(y_val)

print(f"\n🎉 Résultat Final sur le Test Set : {accuracy * 100:.2f}% de précision")