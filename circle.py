import torch
import sklearn.datasets
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import sklearn.model_selection
import matplotlib.pyplot as plt

SEED = 42

# 1. Fixer le hasard de PyTorch
torch.manual_seed(SEED)

# 2. Fixer le hasard de Numpy (pour make_circles)
np.random.seed(SEED)

# 3. Fixer le hasard de Python
random.seed(SEED)

n_circles = 1000
noise = 0.05
X, y = sklearn.datasets.make_circles(n_samples=n_circles, noise=noise)
print(X, "\n")
print(y)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=SEED)

class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=2, out_features=64)
        self.layer2 = nn.Linear(in_features=64, out_features=64)
        self.layer3 = nn.Linear(in_features=64, out_features=64)
        self.layer4 = nn.Linear(in_features=64, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.layer4(x)
        return x

model = CircleModelV0()
loss = nn.BCEWithLogitsLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.06)
scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=250, gamma=0.8)

for epoch in range(1000):
    model.train()
    y_hat = model.forward(X_train).squeeze()
    lost = loss(y_hat, y_train)
    
    optimiser.zero_grad()
    lost.backward()

    optimiser.step()

    scheduler.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Loss: {lost.item():.5f}")


model.eval()
with torch.no_grad():
    test_logits = model(X_test).squeeze()
    predictions = torch.round(torch.sigmoid(test_logits))
    accuracy = (predictions == y_test).sum().item() / len(y_test)

print(f"\n Résultat Final sur le Test Set : {accuracy * 100:.2f}% de précision")


def plot_decision_boundary(model, X, y):
    # Met le modèle en mode évaluation
    model.to("cpu")
    model.eval()
    
    # Crée une grille de points (le fond de la carte)
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101),
                         np.linspace(y_min, y_max, 101))
    
    # Fait prédire le modèle sur chaque pixel de la grille
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()
    
    with torch.no_grad():
        y_logits = model(X_to_pred_on)
    
    y_pred = torch.round(torch.sigmoid(y_logits))
    y_pred = y_pred.reshape(xx.shape).numpy()
    
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()

plot_decision_boundary(model, X_train.detach().numpy(), y_train.detach().numpy())