import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Hyperparameters
INPUT_SIZE = 784
HIDDEN_SIZE = 20
HIDDEN_LAYERS = 16
OUTPUT_CLASSES = 10
EPOCHS = 200
BATCH_SIZE = 128
LEARNING_RATE = 0.001

# Data preprocessing (flatten and normalize)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(lambda x: x.view(-1))  # Flatten 28x28 to 784
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define the model with Tanh and Softmax
class TanhSoftmaxNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=20, num_layers=16, num_classes=10):
        super(TanhSoftmaxNet, self).__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers += [nn.Linear(hidden_size, num_classes)]
        self.hidden = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=1)  # Explicit Softmax

    def forward(self, x):
        x = self.hidden(x)
        x = self.softmax(x)
        return x

# Instantiate model
model = TanhSoftmaxNet()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.KLDivLoss(reduction='batchmean')  # Use with log probs

# Helper to one-hot encode labels
def one_hot(labels, num_classes=10):
    return F.one_hot(labels, num_classes=num_classes).float()

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in train_loader:
        outputs = model(images)
        targets = one_hot(labels)
        log_outputs = torch.log(outputs + 1e-9)  # avoid log(0)

        loss = loss_fn(log_outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    acc = correct / total * 100
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%")

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {correct / total * 100:.2f}%")
