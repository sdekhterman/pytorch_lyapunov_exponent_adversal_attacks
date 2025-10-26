import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Define the original network architecture
class TanhSoftmaxNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=20, hidden_layer=16, number_of_outputs=10):
        super(TanhSoftmaxNet, self).__init__()
        self.hidden_size = hidden_size
        
        # Build the sequential model
        layers = [nn.Linear(input_size, hidden_size), nn.Tanh()]
        for _ in range(hidden_layer - 1): # Adjust loop to build correct number of layers
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers += [nn.Linear(hidden_size, number_of_outputs)]
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(1 / self.hidden_size))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        return self.network(x)

# Define the new network with the bottleneck layer
class BottleneckNet(nn.Module):
    def __init__(self, feature_extractor, number_of_outputs=10):
        super(BottleneckNet, self).__init__()
        self.feature_extractor = feature_extractor
        
        # The new layers to be trained
        self.bottleneck = nn.Sequential(
            nn.Linear(20, 2), # Bottleneck layer with 2 neurons
            nn.Tanh(),
            nn.Linear(2, number_of_outputs) # New output layer
        )

    def forward(self, x):
        # Pass input through the frozen, pre-trained layers
        features = self.feature_extractor(x)
        # Pass the extracted features through the new bottleneck and output layers
        output = self.bottleneck(features)
        return output

class MNISTClassification:
    def __init__(self, learning_rate: float = 0.2, number_of_epochs: int = 100, batch_size: int = 64) -> None:
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.number_of_epochs = number_of_epochs
        
        # Data loading and transformation
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) 
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)

        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Models will be initialized in their respective training methods
        self.original_model = None
        self.bottleneck_model = None

    def train_model(self, model, epochs, title="Training Phase"):
        print(f"--- Starting: {title} ---")
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=self.learning_rate)
        
        train_losses = []
        test_accuracies = []
        n_total_steps = len(self.train_loader)

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for images, labels in self.train_loader:
                images = images.reshape(-1, 28 * 28).to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(images)
                loss = self.criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / n_total_steps
            train_losses.append(avg_loss)

            # Evaluate on test set
            model.eval()
            with torch.no_grad():
                n_correct = 0
                n_samples = 0
                for images, labels in self.test_loader:
                    images = images.reshape(-1, 28 * 28).to(self.device)
                    labels = labels.to(self.device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    n_samples += labels.size(0)
                    n_correct += (predicted == labels).sum().item()

                acc = 100.0 * n_correct / n_samples
                test_accuracies.append(acc)

            if (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {acc:.2f} %')
        
        self.plot_training_progress(epochs, train_losses, test_accuracies, title)
        return model

    def plot_training_progress(self, epochs, losses, accuracies, title):
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        fig.suptitle(f'Metrics for: {title}', fontsize=16)

        ax1.plot(range(epochs), losses, label='Training Loss', color='royalblue')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Cross-Entropy Loss')
        ax1.set_title('Training Loss over Epochs')
        ax1.legend()

        ax2.plot(range(epochs), accuracies, label='Test Accuracy', color='seagreen')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Test Accuracy over Epochs')
        ax2.legend()
        ax2.set_ylim(min(accuracies) - 2, 100)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def visualize_bottleneck(self):
        print("\n--- Visualizing Bottleneck Layer Output ---")
        self.bottleneck_model.eval()
        points = []
        point_labels = []

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.reshape(-1, 28 * 28).to(self.device)
                # Get the output from the feature extractor + the first layer of the bottleneck
                features = self.bottleneck_model.feature_extractor(images)
                bottleneck_output = self.bottleneck_model.bottleneck[0](features) # Output of the 2-neuron linear layer
                
                points.append(bottleneck_output.cpu().numpy())
                point_labels.append(labels.cpu().numpy())

        points = np.concatenate(points, axis=0)
        point_labels = np.concatenate(point_labels, axis=0)
        
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(points[:, 0], points[:, 1], c=point_labels, cmap='tab10', alpha=0.7, s=10)
        plt.title('2D Bottleneck Layer Activations for Test Data')
        plt.xlabel('Neuron 1 Activation')
        plt.ylabel('Neuron 2 Activation')
        plt.legend(handles=scatter.legend_elements()[0], labels=range(10), title="Digits")
        plt.grid(True)
        plt.show()


def main():
    # Phase 1: Train the original full network
    classifier = MNISTClassification(number_of_epochs=100)
    original_model = TanhSoftmaxNet().to(classifier.device)
    trained_original_model = classifier.train_model(original_model, epochs=100, title="Phase 1: Original Model Training")

    # Phase 2: Create, freeze, and retrain the bottleneck model
    # The feature extractor is all layers *except* the final classification layer
    feature_extractor = trained_original_model.network[:-1]
    
    # --- FREEZE THE WEIGHTS of the feature extractor ---
    for param in feature_extractor.parameters():
        param.requires_grad = False

    # Create the new model with the frozen feature extractor
    bottleneck_model = BottleneckNet(feature_extractor).to(classifier.device)
    
    # Train only the new bottleneck and output layers
    classifier.bottleneck_model = classifier.train_model(bottleneck_model, epochs=50, title="Phase 2: Bottleneck Fine-Tuning")
    
    # Visualize the 2D representation from the bottleneck layer
    classifier.visualize_bottleneck()


if __name__ == "__main__":
    main()