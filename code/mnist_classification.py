import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import math


class DeepTanhNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=20, hidden_layer=16, number_of_outputs=10):
        super(DeepTanhNet, self).__init__()
        self.hidden_size = hidden_size
        layers = [nn.Linear(input_size, hidden_size), nn.Tanh()]
        for _ in range(hidden_layer):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers += [nn.Linear(hidden_size, number_of_outputs)]
        self.hidden = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for layer in self.hidden:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=math.sqrt(1 / self.hidden_size))
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        return self.hidden(x)


class BottleneckNet(nn.Module):
    def __init__(self, feature_extractor, hidden_size=20, bottleneck_size=2, number_of_outputs=10):
        super(BottleneckNet, self).__init__()
        self.feature_extractor = feature_extractor
        self.bottleneck = nn.Sequential(nn.Tanh(),nn.Linear(hidden_size, bottleneck_size), 
                                        nn.Tanh(), nn.Linear(bottleneck_size, number_of_outputs))

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.bottleneck_linear(features)

    def get_bottleneck_activations(self, x):
        with torch.no_grad():
            features = self.feature_extractor(x)
            bottleneck_out = self.bottleneck_linear(features)
            return bottleneck_out


class MNISTClassifcation:
    def __init__(self, learning_rate: float = 0.01, number_of_epochs: int = 200, epoch_print_period: int = 10, batch_size: int = 64) -> None:
        self.batch_size         = batch_size
        self.learning_rate      = learning_rate
        self.number_of_epochs   = number_of_epochs
        self.epoch_print_period = epoch_print_period

        #Normalizing the data helps with training convergence and stability. The mean and standard deviation are calculated from the MNIST dataset, which has pixel values in the range [0, 1] after applying ToTensor(). Normalization centers the data around zero and scales it to have a standard deviation of 1.
        transform         = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) 
        train_dataset     = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        test_dataset      = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader  = torch.utils.data.DataLoader(dataset=test_dataset , batch_size=self.batch_size, shuffle=False)

        # nn.CrossEntropyLoss already applies Softmax internally, so we remove it from the model's forward pass.
        self.criterion = nn.CrossEntropyLoss()
        self.device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model     = DeepTanhNet().to(self.device)


        self.n_total_steps   = len(self.train_loader)
        self.train_losses    = []
        self.test_accuracies = []

    def train_model(self) -> None:
        optimizer = optim.SGD(self.model.parameters(), lr=3e-3, momentum=0.9)

        for epoch in range(self.number_of_epochs):
            self.model.train() # Set model to training mode
            running_loss = 0.0
            for i, (images, labels) in enumerate(self.train_loader):
                # Reshape images to (batch_size, input_size)
                images  = images.reshape(-1, 28 * 28).to(self.device)
                labels  = labels.to(self.device)
                outputs = self.model(images)
                loss    = self.criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / self.n_total_steps
            self.train_losses.append(avg_loss)

            self.model.eval()
            with torch.no_grad():
                n_correct = 0
                n_samples = 0
                for images, labels in self.test_loader:
                    images  = images.reshape(-1, 28 * 28).to(self.device)
                    labels  = labels.to(self.device)
                    outputs = self.model(images)

                    # max returns (value, index)
                    _, predicted = torch.max(outputs.data, 1)
                    n_samples += labels.size(0)
                    n_correct += (predicted == labels).sum().item()

                acc = 100.0 * n_correct / n_samples
                self.test_accuracies.append(acc)

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.number_of_epochs}], Loss: {avg_loss:.4f}, Accuracy: {acc:.2f} %')

    def plot_training_loss(self):
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        ax1.plot(range(self.number_of_epochs), self.train_losses, label='Training Loss', color='royalblue')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Cross-Entropy Loss')
        ax1.set_title('Training Loss over Epochs')
        ax1.legend()

        ax2.plot(range(self.number_of_epochs), self.test_accuracies, label='Test Accuracy', color='seagreen')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Test Accuracy over Epochs')
        ax2.legend()
        ax2.set_ylim(min(self.test_accuracies) - 2, 100)

        plt.tight_layout()
        plt.show()


def main():
    mnist_class_object = MNISTClassifcation()
    mnist_class_object.train_model()
    mnist_class_object.plot_training_loss()

if __name__=="__main__":
    main()