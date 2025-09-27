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
    
    def max_lyapunov_exponents(self, x: torch.Tensor) -> list[torch.Tensor]:
        # Ensure input is a 2D tensor for batch processing
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # D_matrices    = []
        current_input = x
        jacobian = None

        # A "layer" in the formula corresponds to a (Linear, Tanh) pair.
        # We iterate through the model's layers two at a time.
        for i in range(0, len(self.model), 2):
            linear_layer     = self.model[i]
            activation_layer = self.model[i+1]

            # Weight matrix from last layer to current layer
            W_l = linear_layer.state_dict()['weight']

            # 1. Get the pre-activation value b^(l)
            # This is the output of the linear layer BEFORE the activation.
            b = linear_layer(current_input)

            # 2. Compute the derivative g'(b^(l))
            # For g(b) = tanh(b), the derivative g'(b) = 1 - tanh^2(b).
            # We can get tanh(b) by applying the activation layer.
            tanh_of_b = activation_layer(b)
            g_prime = 1 - tanh_of_b.pow(2)

            # 3. Construct the diagonal matrix D^(l)
            # torch.diag_embed creates a batch of diagonal matrices from a batch of vectors.
            D_l = torch.diag_embed(g_prime)
            # D_matrices.append(D_l)

            # The output of this layer's activation is the input for the next linear layer
            current_input = tanh_of_b

            # Compute the Jacobian (change in the network matrix) starting at the network input to layer l of the network
            temp = D_l @ W_l
            jacobian = temp if jacobian is None else temp @ jacobian
        
        cauchy_green_tensor = torch.transpose(jacobian, 1, 2) @ jacobian
        L = torch.linalg.eigvals(cauchy_green_tensor)

        # 1. Get the real part of the complex tensor L.
        L_real_parts = torch.real(L)
        
        # 2. Sort the real parts to get the sorting indices.
        # We only need the indices, so we use '_' for the sorted values.
        _, sorted_indices = torch.sort(L_real_parts, dim=-1, descending=True)

        # 3. Use torch.gather() to sort the original complex tensor L using these indices.
        sorted_L = torch.gather(L, dim=-1, index=sorted_indices)

        return sorted_L[:,0]
    


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
for epoch in range(1000):
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




# ---- Visualization ----
def plot_classification():
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

# def plot_finite_time_lyapunov_exponents():
#     Ls_gpu = model.max_lyapunov_exponents(x_test)
#     Ls = Ls_gpu.detach().cpu().numpy()
#     x0 = x_test[:, 0].detach().cpu().numpy()
#     x1 = x_test[:, 1].detach().cpu().numpy()
    
#     # background heatmap
#     pcm = plt.pcolormesh(x0, x1, Ls, cmap='RdBu_r', shading='auto')

#     plt.axis('equal')

#     # colorbar
#     plt.colorbar(pcm, label="Ls")

#     plt.show()




# plot_classification()

# plot_finite_time_lyapunov_exponents()