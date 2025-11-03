import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
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


class MNISTClassification:
    def __init__(self, learning_rate: float = 0.001, number_of_epochs: int = 5, batch_size: int = 64, display_training_updates = True) -> None:
        self.batch_size               = batch_size
        self.learning_rate            = learning_rate
        self.number_of_epochs         = number_of_epochs
        self.display_training_updates = display_training_updates
        
        transform         = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) 
        train_dataset     = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        test_dataset      = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader  = torch.utils.data.DataLoader(dataset=test_dataset , batch_size=1000           , shuffle=False) 

        self.loss_function = nn.CrossEntropyLoss()
        self.device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_model(self, model):
        model.eval()
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            for images, labels in self.test_loader:
                # images  = images.reshape(-1, 28 * 28).to(self.device)
                images  = images.to(self.device)
                labels  = labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

            acc = 100.0 * n_correct / n_samples
            return acc

    def train_model(self, model, title="Training Phase"):
        print(f"--- Starting: {title} ---")
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        train_losses    = []
        test_accuracies = []
        n_total_steps   = len(self.train_loader)

        for epoch in range(self.number_of_epochs):
            model.train()
            running_loss = 0.0
            for images, labels in self.train_loader:
                # images = images.reshape(-1, 28 * 28).to(self.device)
                images  = images.to(self.device)
                labels  = labels.to(self.device)
                outputs = model(images)
                loss    = self.loss_function(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / n_total_steps
            train_losses.append(avg_loss)

            model_accuracy_percent = self.test_model(model)
            test_accuracies.append(model_accuracy_percent)

            if (((epoch + 1) % 5 == 0) and self.display_training_updates):
                print(f'Epoch [{epoch+1}/{self.number_of_epochs}], Loss: {avg_loss:.4f}, Accuracy: {model_accuracy_percent:.2f} %')
        
        return model

    def fgsm_attack(self, images, epsilon, image_grads):
        sign_image_grads = image_grads.sign()
        perturbed_images = images + epsilon * sign_image_grads
        perturbed_images = torch.clamp(perturbed_images, 0, 1) # valid range [0, 1]
        return perturbed_images

    def test_attack(self, model, epsilon):
        n_correct    = 0
        adv_examples = []
        model.eval() 

        for images, labels in self.test_loader:
            # compute gradients for Fast Gradient Sign Method (FGSM) attacks on input images
            images = images.to(self.device)
            labels = labels.to(self.device)
            images.requires_grad = True
            outputs = model(images)
            loss    = self.loss_function(outputs, labels)
            model.zero_grad()
            loss.backward()
            images_grads        = images.grad.data
            _, init_predictions = torch.max(outputs.data, 1)

            # apply and evaluate FGSM attacks
            perturbed_data      = self.fgsm_attack(images, epsilon, images_grads)
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

        final_acc = n_correct / float(len(self.test_loader.dataset.data))
        print(f"Epsilon: {epsilon}\tTest Accuracy = {n_correct} / {len(self.test_loader.dataset.data)} = {final_acc:.4f}")

        return final_acc, adv_examples
    
    def analyze_attacks(self, model, epsilons):
        accuracies   = []
        all_examples = []

        for epsilon in epsilons:
            acc, ex = self.test_attack(model, epsilon)
            accuracies.append(acc)
            all_examples.append(ex)


        my_path        = os.path.dirname(os.path.abspath(__file__))
        my_path_parent = os.path.dirname(my_path)
        my_file        = "/images/fgsm_example_attack.png"
        plot_path      = my_path_parent + my_file

        plt.figure(figsize=(10, 8))

        try:
            examples_to_show = all_examples[3]
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

def main():
    epsilons     = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    classifier      = MNISTClassification()
    untrained_model = DeepFCNet().to(classifier.device)
    trained_model   = classifier.train_model(untrained_model, title="Phase 1: Model Training")
    classifier.analyze_attacks(trained_model, epsilons)

if __name__ == "__main__":
    main()