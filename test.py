import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Define the deep tanh neural network
class DeepTanhNet(nn.Module):
    def __init__(self, input_size=2, hidden_size=12, hidden_layers=10):
        super(DeepTanhNet, self).__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers += [nn.Linear(hidden_size, 1), nn.Sigmoid()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Generate 40,000 points in [-1, 1] x [-1, 1] and classify by unit circle
def generate_circle_data(n_samples=40000):
    x = np.random.uniform(-1.5, 1.5, (n_samples, 2))
    y = ((x[:, 0]**2 + x[:, 1]**2) <= 1.0).astype(np.float32)  # inside unit circle
    return x, y

# Load data
x_all, y_all = generate_circle_data()
x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.1, random_state=42)

# Convert to tensors
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Initialize model
model = DeepTanhNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        model.eval()
        with torch.no_grad():
            test_outputs = model(x_test)
            test_preds = (test_outputs > 0.5).float()
            test_acc = (test_preds == y_test).float().mean()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Test Accuracy: {test_acc.item()*100:.2f}%")

# ---- Visualization ----
def plot_decision_boundary(model, resolution=200):
    model.eval()
    x_min, x_max = -1.5, 1.5
    y_min, y_max = -1.5, 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]
    with torch.no_grad():
        preds = model(torch.tensor(grid, dtype=torch.float32)).numpy().reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, preds, levels=[0, 0.5, 1], alpha=0.6, cmap='coolwarm')
    plt.colorbar(label='Predicted Probability')

    # Plot test points with ground-truth labels
    y_test_np = y_test.numpy().ravel()
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test_np, cmap='bwr', edgecolors='k', s=20, alpha=0.7)

    # Unit circle for reference
    circle = plt.Circle((0, 0), 1, color='k', fill=False, linestyle='--')
    plt.gca().add_patch(circle)

    plt.title("Decision Boundary: Inside Unit Circle Classification")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.axis('equal')
    plt.show()

plot_decision_boundary(model)
