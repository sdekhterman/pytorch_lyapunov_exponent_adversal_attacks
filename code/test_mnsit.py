import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import math

# --------------------------------
# 1. Device Configuration
# --------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --------------------------------
# 2. Hyperparameters
# --------------------------------
INPUT_SIZE = 784  # 28x28
HIDDEN_SIZE = 20
HIDDEN_LAYER = 16
NUM_CLASSES = 10
NUM_EPOCHS = 200
BATCH_SIZE = 100
LEARNING_RATE = 0.01

# --------------------------------
# 3. MNIST Dataset
# --------------------------------
# Transformation pipeline
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # Mean and std dev of MNIST
])

# Load data
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)

# Data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --------------------------------
# 4. Model Definition (as defined above)
# --------------------------------
class TanhSoftmaxNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=20, hidden_layer=16, num_classes=10):
        super(TanhSoftmaxNet, self).__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.Tanh()]
        for _ in range(hidden_layer):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers += [nn.Linear(hidden_size, num_classes)]
        self.hidden = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for layer in self.hidden:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=math.sqrt(1 / HIDDEN_SIZE))
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        return self.hidden(x)

model = TanhSoftmaxNet(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    hidden_layer=HIDDEN_LAYER,
    num_classes=NUM_CLASSES
).to(device)

# --------------------------------
# 5. Loss and Optimizer
# --------------------------------
# nn.CrossEntropyLoss already applies Softmax internally, so we remove it from the model's forward pass.
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

# --------------------------------
# 6. Training Loop
# --------------------------------
print("Starting training...")
n_total_steps = len(train_loader)
train_losses = []
test_accuracies = []

for epoch in range(NUM_EPOCHS):
    model.train() # Set model to training mode
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        # Reshape images to (batch_size, input_size)
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Calculate and store average loss for the epoch
    avg_loss = running_loss / n_total_steps
    train_losses.append(avg_loss)

    # --------------------------------
    # 7. Testing / Evaluation
    # --------------------------------
    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)
            outputs = model(images)

            # max returns (value, index)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        test_accuracies.append(acc)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}, Accuracy: {acc:.2f} %')

print("Finished Training")

# --------------------------------
# 8. Plotting Results
# --------------------------------
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot Training Loss
ax1.plot(range(NUM_EPOCHS), train_losses, label='Training Loss', color='royalblue')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Cross-Entropy Loss')
ax1.set_title('Training Loss over Epochs')
ax1.legend()

# Plot Test Accuracy
ax2.plot(range(NUM_EPOCHS), test_accuracies, label='Test Accuracy', color='seagreen')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Test Accuracy over Epochs')
ax2.legend()
ax2.set_ylim(min(test_accuracies) - 2, 100) # Adjust y-axis for better visibility

plt.tight_layout()
plt.show()