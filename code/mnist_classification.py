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
        for _ in range(hidden_layer): # Adjust loop to build correct number of layers
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
    
    def max_finite_time_lyapunov_exponents(self, x: torch.Tensor) -> list[torch.Tensor]:
        if x.dim() == 1:
            x = x.unsqueeze(0)

        current_input = x
        jacobian = None

        for i in range(0, len(self.network)-1, 2):
            linear_layer = self.network[i]
            activation_layer = self.network[i + 1]
            W_l = linear_layer.state_dict()['weight']

            b = linear_layer(current_input)
            tanh_of_b = activation_layer(b)
            g_prime = 1 - tanh_of_b.pow(2)
            D_l = torch.diag_embed(g_prime)

            current_input = tanh_of_b
            temp = D_l @ W_l
            jacobian = temp if jacobian is None else temp @ jacobian
        
        cauchy_green_tensor = torch.transpose(jacobian, 1, 2) @ jacobian
        singular_values = torch.linalg.svdvals(cauchy_green_tensor)
        max_singular_values = singular_values[:, 0]
        max_lyapunov_exponents = torch.log10(max_singular_values)

        return max_lyapunov_exponents

# Define the new network with the bottleneck layer
class BottleneckNet(nn.Module):
    def __init__(self, feature_extractor, number_of_outputs=10):
        super(BottleneckNet, self).__init__()
        self.feature_extractor = feature_extractor
        
        # The new layers to be trained
        self.bottleneck = nn.Sequential(
            nn.Linear(20, 2),
            nn.Tanh(),
            nn.Linear(2, number_of_outputs)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        self.model = self.bottleneck(features)
        return self.model
    
    def get_bottleneck_activations(self, x):
        """Helper function to get the 2D activations for visualization."""
        with torch.no_grad():
            features = self.feature_extractor(x)
            output = self.bottleneck(features)
            return output
    

class MNISTClassification:
    def __init__(self, learning_rate: float = 0.2, number_of_epochs: int = 10, batch_size: int = 64) -> None:
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

    def train_model(self, model, title="Training Phase"):
        print(f"--- Starting: {title} ---")
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),lr=3e-3, momentum=0.9)
        
        train_losses = []
        test_accuracies = []
        n_total_steps = len(self.train_loader)

        for epoch in range(self.number_of_epochs):
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
                print(f'Epoch [{epoch+1}/{self.number_of_epochs}], Loss: {avg_loss:.4f}, Accuracy: {acc:.2f} %')
        
        return model

    def visualize_ftle_on_data_points(self):
            """
            Visualizes bottleneck activations, coloring each point by its own FTLE value.
            """
            print("\n--- Visualizing FTLE Value for Each Data Point ---")
            if self.bottleneck_model is None:
                print("Bottleneck model not trained. Cannot visualize.")
                return

            self.bottleneck_model.eval()
            points = []
            point_labels = [] # We can still collect labels if we want to compare later
            max_lyapunov_exponents_list = []

            # Step 1: Get the bottleneck activation points from the test set
            with torch.no_grad():
                for images, labels in self.test_loader:
                    images = images.reshape(-1, 28 * 28).to(self.device)
                    max_lyapunov_exponents_list.append(self.original_model.max_finite_time_lyapunov_exponents(images).cpu().detach().numpy())
                    bottleneck_output = self.bottleneck_model.get_bottleneck_activations(images)
                    points.append(bottleneck_output.cpu().numpy())
                    point_labels.append(labels.cpu().numpy())

                    print(f"Processed batch with {images.size(0)} samples. Total samples processed: {len(points)}")

            points                      = np.concatenate(points, axis=0)
            point_labels                = np.concatenate(point_labels, axis=0)
            max_lyapunov_exponents_list = np.concatenate(max_lyapunov_exponents_list, axis=0)

            # Step 3: Plotting
            plt.figure(figsize=(12, 10))
            
            # Create a scatter plot where color is determined by the FTLE value
            scatter = plt.scatter(points[:, 0], points[:, 1], c=max_lyapunov_exponents_list, cmap='plasma',
                                alpha=0.8, s=15)
            
            plt.colorbar(scatter, label="Max FTLE ($\log_{10}$ scale)")
            plt.title('Bottleneck Activations Colored by FTLE Value')
            plt.xlabel('Neuron 1 Activation')
            plt.ylabel('Neuron 2 Activation')
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.axis('equal')
            plt.show()


def main():
    # Phase 1: Train the original full network
    classifier                = MNISTClassification()
    original_model            = TanhSoftmaxNet().to(classifier.device)
    classifier.original_model = original_model
    trained_original_model    = classifier.train_model(original_model, title="Phase 1: Original Model Training")

    # Phase 2: Create, freeze, and retrain the bottleneck model
    # The feature extractor is all layers *except* the final classification layer
    feature_extractor = trained_original_model.network[:-1]
    
    # --- FREEZE THE WEIGHTS of the feature extractor ---
    for param in feature_extractor.parameters():
        param.requires_grad = False

    # Create the new model with the frozen feature extractor
    bottleneck_model = BottleneckNet(feature_extractor).to(classifier.device)
    
    # Train only the new bottleneck and output layers
    classifier.bottleneck_model = classifier.train_model(bottleneck_model, title="Phase 2: Bottleneck Fine-Tuning")
    
    # Phase 3: Visualize the FTLE value for each data point in the bottleneck space
    classifier.visualize_ftle_on_data_points()


if __name__ == "__main__":
    main()