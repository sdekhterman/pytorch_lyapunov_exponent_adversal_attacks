import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

class DeepFCNet(nn.Module):
    def __init__(self):
        super(DeepFCNet, self).__init__()
        # MNIST images are 28x28 = 784 pixels
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10) # 10 output classes for digits 0-9

    def forward(self, x):
        # Flatten the images from [batch_size, 1, 28, 28] to [batch_size, 784]
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        # No softmax needed here, as F.cross_entropy() combines it
        return x


device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform    = transforms.ToTensor()
train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True , download=True, transform=transform), batch_size=64, shuffle=True)
test_loader  = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False, download=True, transform=transform), batch_size=1000 , shuffle=True) # Batch size 1 for easy attack example
model        = DeepFCNet().to(device)
optimizer    = optim.Adam(model.parameters(), lr=0.001)
loss_fn      = nn.CrossEntropyLoss()

print("Training model for 1 epoch...")
model.train()
for (images, labels) in train_loader:
    images = images.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()
    outputs = model(images)
    loss    = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()
print("Training complete.")


def fgsm_attack(images, epsilon, image_grads):
    sign_image_grads = image_grads.sign()
    perturbed_images = images + epsilon * sign_image_grads
    perturbed_images = torch.clamp(perturbed_images, 0, 1) # valid range [0, 1]
    return perturbed_images


def test_attack(model, device, test_loader, epsilon):
    
    model.eval() 
    n_correct = 0
    adv_examples = []

    for images, labels in test_loader:
        # compute gradients for FGSM attacks on input images
        images = images.to(device)
        labels = labels.to(device)
        images.requires_grad = True
        outputs = model(images)
        loss    = loss_fn(outputs, labels)
        model.zero_grad()
        loss.backward()
        images_grads        = images.grad.data
        _, init_predictions = torch.max(outputs.data, 1)

        # apply and evaluate FGSM attacks
        perturbed_data      = fgsm_attack(images, epsilon, images_grads)
        output_adv          = model(perturbed_data)
        _, final_predictions = torch.max(output_adv.data, 1)

        n_correct += (final_predictions == labels).sum().item()
        for index in range(len(final_predictions)):
            init_pred_item  = init_predictions[index].item()
            final_pred_item = final_predictions[index].item()
            label_item      = labels[index].item()

            was_match   = ( init_pred_item == label_item)
            is_mismatch = (final_pred_item != label_item)
            
            if (was_match and is_mismatch and (len(adv_examples) < 5)):

                adv_ex          = perturbed_data[index].squeeze().detach().cpu().numpy()
                orig_ex         = images[index].squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred_item, final_pred_item, orig_ex, adv_ex))

    final_acc = n_correct / float(len(test_loader))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {n_correct} / {len(test_loader)} = {final_acc:.4f}")

    return final_acc, adv_examples

# --- 6. Run the Attack ---
# epsilons     = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
epsilons     = [0, 0.05, 0.1, 0.15]
accuracies   = []
all_examples = []

for eps in epsilons:
    acc, ex = test_attack(model, device, test_loader, eps)
    accuracies.append(acc)
    all_examples.append(ex)


my_path        = os.path.dirname(os.path.abspath(__file__))
my_path_parent = os.path.dirname(my_path)
my_file        = "/images/fgsm_example_attack.png"
plot_path      = my_path_parent + my_file

# plotting examples of adversarial attacks
plt.figure(figsize=(10, 8))

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
    plt.savefig(plot_path, dpi=600)

except IndexError:
    print("\nNot enough adversarial examples found to display for that epsilon.")
    print("Try training the model for more epochs or increasing epsilon.")