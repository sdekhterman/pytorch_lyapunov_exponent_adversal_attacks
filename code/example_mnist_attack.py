import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Define the Deep Fully Connected Network ---

class DeepFCNet(nn.Module):
    def __init__(self):
        super(DeepFCNet, self).__init__()
        # MNIST images are 28x28 = 784 pixels
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10) # 10 output classes for digits 0-9

    def forward(self, x):
        # Flatten the image from [batch_size, 1, 28, 28] to [batch_size, 784]
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        # No softmax needed here, as F.cross_entropy() combines it
        return x

# --- 2. Load Data and Define Training/Testing ---

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MNIST DataLoaders
transform = transforms.ToTensor()
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True, transform=transform),
    batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=transform),
    batch_size=1, shuffle=True) # Batch size 1 for easy attack example

# --- 3. (Optional) Train the Model ---
# In a real scenario, you'd load a pre-trained model.
# For this example, we'll train for just one epoch.

model = DeepFCNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

print("Training model for 1 epoch...")
model.train() # Set model to training mode
for (data, target) in train_loader:
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
print("Training complete.")

# --- 4. Define the FGSM Attack ---

def fgsm_attack(image, epsilon, data_grad):
    # Get the sign of the gradients
    sign_data_grad = data_grad.sign()
    # Create the perturbed image
    perturbed_image = image + epsilon * sign_data_grad
    # Clamp the image to the valid range [0, 1]
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

# --- 5. Test Model Against the Attack ---

def test_attack(model, device, test_loader, epsilon):
    
    model.eval() # Set model to evaluation mode
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        
        # We need to compute gradients with respect to the input data
        data.requires_grad = True

        # Forward pass
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # Get the index of the max log-probability

        # If the initial prediction is wrong, don't bother attacking
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = loss_fn(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Get the gradients of the data
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output_adv = model(perturbed_data)
        final_pred = output_adv.max(1, keepdim=True)[1] # Get the index of the max log-probability

        if final_pred.item() == target.item():
            correct += 1
        else:
            # Save some successful adversarial examples
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                orig_ex = data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), orig_ex, adv_ex))

    # Calculate final accuracy
    final_acc = correct / float(len(test_loader))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader)} = {final_acc:.4f}")

    return final_acc, adv_examples

# --- 6. Run the Attack ---

epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
accuracies = []
all_examples = []

for eps in epsilons:
    acc, ex = test_attack(model, device, test_loader, eps)
    accuracies.append(acc)
    all_examples.append(ex)


# --- 7. Plot the Results ---

# Plotting code
plt.figure(figsize=(10, 8))

# Get examples for epsilon = 0.15
try:
    examples_to_show = all_examples[3] # Index 3 corresponds to epsilon = 0.15
    cnt = 0
    for i in range(len(examples_to_show)):
        cnt += 1
        orig_pred, adv_pred, orig_img, adv_img = examples_to_show[i]
        
        # Original Image
        plt.subplot(2, 5, cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.title(f"Original: {orig_pred}")
        plt.imshow(orig_img, cmap="gray")
        
        # Adversarial Image
        plt.subplot(2, 5, cnt + 5)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.title(f"Adversarial: {adv_pred}")
        plt.imshow(adv_img, cmap="gray")
        
        if cnt == 5:
            break
            
    plt.tight_layout()
    plt.show()

except IndexError:
    print("\nNot enough adversarial examples found to display for that epsilon.")
    print("Try training the model for more epochs or increasing epsilon.")