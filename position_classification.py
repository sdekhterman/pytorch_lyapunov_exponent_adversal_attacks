import math
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
        layers += [nn.Linear(hidden_size, 1), nn.Tanh()]
        self.model = nn.Sequential(*layers)
        
        # Custom initialization
        self._initialize_weights(hidden_size)

    def _initialize_weights(self, hidden_size):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                # Gaussian with mean=0, std = sqrt(1 / hidden_size)
                nn.init.normal_(layer.weight, mean=0.0, std=math.sqrt(1 / hidden_size))
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        return self.model(x)

# Generate 40,000 points in [-1.25, 1.25] x [-1.25, 1.25] and classify by unit circle
def generate_circle_data(n_samples=40000):
    x = np.random.uniform(-1.25, 1.25, (n_samples, 2))
    t = 2.0 * ((x[:, 0]**2 + x[:, 1]**2) <= 1.0).astype(np.float32) - 1.0
    return x, t

# Load data
x_all, t_all = generate_circle_data()
x_train, x_test, t_train, t_test = train_test_split(x_all, t_all, test_size=0.1, random_state=42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Convert to tensors
x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
t_train = torch.tensor(t_train, dtype=torch.float32).unsqueeze(1).to(device)
x_test  = torch.tensor(x_test, dtype=torch.float32).to(device)
t_test  = torch.tensor(t_test, dtype=torch.float32).unsqueeze(1).to(device)

# Initialize model
model     = DeepTanhNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.05)

# Training loop
for epoch in range(10000):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train)
    loss    = criterion(outputs, t_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        model.eval()
        with torch.no_grad():
            test_outputs = model(x_test)
            test_preds = torch.sign(test_outputs)
            test_acc = (test_preds == t_test).float().mean()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Test Accuracy: {test_acc.item()*100:.2f}%")

for i, layer in enumerate(model.model):
    if isinstance(layer, nn.Linear):
        weight_matrix = layer.weight.data.cpu().numpy()
        bias_vector   = layer.bias.data.cpu().numpy()
        
        print(f"Layer {i} — weight shape: {weight_matrix.shape}")
        print(weight_matrix)
        print(f"Layer {i} — bias shape: {bias_vector.shape}")
        print(bias_vector)
        print("-" * 40)

weights = []
for layer in model.model:
    if isinstance(layer, nn.Linear):
        weights.append(layer.weight.data.cpu().numpy())


# ---- Visualization ----
def plot_decision_boundary():
    # Choose number of test points to plot
    n_plot = 250

    # Convert test labels to NumPy
    t_test_np = t_test.detach().cpu().numpy().ravel()
    x0 = x_test[:, 0].detach().cpu().numpy()
    x1 = x_test[:, 1].detach().cpu().numpy()

    # Randomly select indices without replacement
    subset_idx = np.random.choice(len(t_test_np), size=n_plot, replace=False)

    # Subset the data
    x0_sub = x0[subset_idx]
    x1_sub = x1[subset_idx]
    t_test_sub = t_test_np[subset_idx]

    # Mask for inside/outside
    inside_mask  = t_test_sub == 1
    outside_mask = t_test_sub == -1

    plt.figure(figsize=(3.2*1.2, 2.4*1.2))

    # Inside points
    plt.scatter(
        x0_sub[inside_mask],
        x1_sub[inside_mask],
        facecolors='black',
        edgecolors='black',
        s=40,
        alpha=1.0,
        marker='s',
        label='Inside Circle',
        linewidth=1.5
    )

    # Outside points
    plt.scatter(
        x0_sub[outside_mask],
        x1_sub[outside_mask],
        facecolors='white',
        edgecolors='green',
        s=40,
        alpha=0.80,
        marker='s',
        label='Outside Circle',
        linewidth=1.5
    )

    # Unit circle for reference
    circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='-', linewidth=1.5)
    plt.gca().add_patch(circle)

    # Turn off grid and background
    ax = plt.gca()
    ax.set_facecolor('white')     # background
    ax.grid(False)                # turn off grid
    for spine in ax.spines.values():
        spine.set_visible(False)  # remove box edges

    # Remove ticks (optional)
    ax.set_xticks([])
    ax.set_yticks([])

    # Add x-axis arrow
    ax.annotate('', xy=(1.6, 0), xytext=(-1.6, 0),
                arrowprops=dict(arrowstyle='->', color='black', linewidth=1.5))

    # Add y-axis arrow
    ax.annotate('', xy=(0, 1.6), xytext=(0, -1.6),
                arrowprops=dict(arrowstyle='->', color='black', linewidth=1.5))

    # Optional: Add axis labels
    plt.text(1.4, -0.3, 'x1', fontsize=14)
    plt.text(0.1, 1.4, 'x2', fontsize=14)

    # Set limits to match arrows
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)

    # Keep aspect ratio square
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()

plot_decision_boundary()

# use later for eigens

# model.eval()
# x0_min, x0_max = -1.25, 1.25
# x1_min, x1_max = -1.25, 1.25
# x0, x1 = np.meshgrid(np.linspace(x0_min, x0_max, resolution),
#                      np.linspace(x1_min,x1_max, resolution))
# grid = np.c_[x0.ravel(), x1.ravel()]
# with torch.no_grad():
#     preds = model(torch.tensor(grid, dtype=torch.float32)).numpy().reshape(x1.shape)

# plt.contourf(x0, x1, preds, levels=[0, 0.5, 1], alpha=0.6, cmap='coolwarm')
# plt.colorbar(label='Predicted Probability')