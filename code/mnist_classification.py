import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import statistics
import numpy as np
import os
from enum import Enum
from torch.utils.data import Subset
import scienceplots
import random

class TanhSoftmaxNet(nn.Module):
    """
    A deep, feed-forward neural network (MLP) using Tanh activations.

    This network consists of an initial linear layer, a specified number of 
    repeating hidden (Linear + Tanh) layers, and a final linear output layer.
    It includes custom weight initialization and a method to calculate
    Finite-Time Lyapunov Exponents (FTLEs).
    """
    def __init__(self, input_size=784, hidden_layer_size=20, numb_hidden_layers=16, number_of_outputs=10):
        """
        Initializes the TanhSoftmaxNet architecture.

        Args:
            input_size (int): The dimensionality of the input features (e.g., 784 for flattened 28x28 MNIST).
            hidden_layer_size (int): The number of neurons in each hidden layer.
            numb_hidden_layers (int): The number of (Linear + Tanh) blocks.
            number_of_outputs (int): The dimensionality of the output (e.g., 10 for MNIST classes).
        """
        super(TanhSoftmaxNet, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.numb_hidden_layers = numb_hidden_layers

        layers = [nn.Linear(input_size, hidden_layer_size), nn.Tanh()]
        for _ in range(numb_hidden_layers): # Adjust loop to build correct number of layers
            layers += [nn.Linear(hidden_layer_size, hidden_layer_size), nn.Tanh()]
        layers += [nn.Linear(hidden_layer_size, number_of_outputs)]
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Applies custom weight and bias initialization to the linear layers.

        Weights are initialized from a normal distribution with mean 0 and std
        dev sqrt(1 / hidden_layer_size). Biases are initialized to 0.
        This is a variant of Xavier/Glorot initialization suitable for Tanh.
        """
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(1 / self.hidden_layer_size))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor (logits).
        """
        return self.network(x)
    
    def max_n_finite_time_lyapunov_exponents(self, x: torch.Tensor, num_lyap_exp: int = 1) -> list[torch.Tensor]:
        """
        Calculates the top 'n' Finite-Time Lyapunov Exponents (FTLEs) for the network.

        This method computes the input-output Jacobian of the *hidden layers* (excluding the final output layer). It then uses the singular values
        of this Jacobian to estimate the FTLEs, which measure the network's
        sensitivity to perturbations in the input.

        Args:
            x (torch.Tensor): The batch of input tensors.
            num_lyap_exp (int): The number of top exponents to return.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, num_lyap_exp) containing
                          the requested FTLEs, normalized by the number of hidden
                          layers.
        """
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
        max_singular_values = singular_values[:, 0:num_lyap_exp]
        max_lyapunov_exponents = torch.log(max_singular_values)
        
        number_of_hidden_layers_tensor = torch.tensor(self.numb_hidden_layers, dtype=torch.float64, device=max_lyapunov_exponents.device)
        output = max_lyapunov_exponents / (2*number_of_hidden_layers_tensor)
        output = torch.where(torch.isfinite(output), output, torch.zeros_like(output))

        return output


class BottleneckNet(nn.Module):
    """
    A network that adds a trainable "bottleneck" to a pre-trained feature extractor.
    
    This is often used for dimensionality reduction (e.g., to 2D for visualization)
    or for transfer learning on top of a frozen base network.
    """
    def __init__(self, feature_extractor, number_of_outputs=10):
        """
        Initializes the BottleneckNet.

        Args:
            feature_extractor (nn.Module): A pre-trained model (or part of one)
                that extracts features. Its output is assumed to be 20-dimensional.
            number_of_outputs (int): The final output dimensionality (e.g., 10 classes).
        """
        super(BottleneckNet, self).__init__()
        self.feature_extractor = feature_extractor
        
        # The new layers to be trained
        self.bottleneck = nn.Sequential(
            nn.Linear(20, 2),
            nn.Tanh(),
            nn.Linear(2, number_of_outputs)
        )

    def forward(self, x):
        """
        Defines the forward pass.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The final output (logits).
        """
        features = self.feature_extractor(x)
        self.model = self.bottleneck(features)
        return self.model


class MNISTClassification:
    """
    A controller class for managing the training, testing, and analysis 
    of models on the MNIST dataset.

    This class encapsulates data loading, training loops, test loops,
    FTLE calculation, adversarial attack generation (FGSM), and
    plotting/visualization of results.
    """
    def __init__(self, learning_rate: float = 3e-3, momentum: float = 0.9, number_of_epochs: int = 25, batch_size: int = 64, debug: bool = False) -> None:
        """
        Initializes the MNIST workflow manager.

        Args:
            learning_rate (float): Learning rate for the SGD optimizer.
            momentum (float): Momentum for the SGD optimizer.
            number_of_epochs (int): Number of epochs to train.
            batch_size (int): Batch size for training.
            debug (bool): If True, reduces the number of epochs and test data
                          for quick debugging runs.
        """
        self.learning_rate    = learning_rate
        self.momentum         = momentum
        self.number_of_epochs = number_of_epochs
        self.batch_size       = batch_size

        if debug:
            self.number_of_epochs = 1
        
        transform         = transforms.Compose([transforms.ToTensor()]) 
        train_dataset     = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        test_dataset      = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)
        
        if debug:
            subset_indices = range(100)
            test_dataset = Subset(test_dataset, subset_indices)

        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader  = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

        self.criterion = nn.CrossEntropyLoss()
        self.device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.original_model   = None
        self.bottleneck_model = None

        my_path        = os.path.dirname(os.path.abspath(__file__))
        my_path_parent = os.path.dirname(my_path)

        self.attack_plot_path          = my_path_parent + "/images/fgsm_example_attack.png"
        self.acc_vs_num_plot_path      = my_path_parent + "/images/accuracy_vs_num.png"
        self.mnist_2d_proj_plot_path   = my_path_parent + "/images/mnist_2d_projection.png"
        self.err_ent_vs_lmbd_plot_path = my_path_parent + "/images/error_entropy_vs_lambda.png"
        self.training_acc_plot_path    = my_path_parent + "/images/training_acc.png"

        # training_model config
        self.reshape_size            = (-1, 28 * 28)
        self.unreshape_size          = (-1, 28, 28)
        self.epoch_loss_print_period = 5

        # lot_error_and_entropy_vs_lambda config
        self.epsilon = 1e-9 # for log stability
        self.correlation_list     = []
        self.atk_correlation_list = []

        # train_attack config
        self.numb_adversaila_examples = 5

        self.set_seed()

    def set_seed(self, seed: int = 42) -> None:
        """
        Sets the seed for reproducible training across PyTorch, NumPy, and Python.
        """
        random.seed(seed)                         # Python random module
        np.random.seed(seed)                      # NumPy random module
        torch.manual_seed(seed)                   # PyTorch CPU
        torch.cuda.manual_seed(seed)              # PyTorch GPU
        torch.cuda.manual_seed_all(seed)          # If multiple GPUs
        torch.backends.cudnn.deterministic = True # Ensure deterministic behavior
        torch.backends.cudnn.benchmark = False    # Disable CuDNN auto-tuner for reproducibility

    def train_model(self, model, title="Training Phase"):
        """
        Trains a given PyTorch model on the MNIST training dataset.

        Args:
            model (nn.Module): The model instance to be trained.
            title (str): A title string to print at the start of training.

        Returns:
            nn.Module: The trained model.
        """
        print(f"--- Starting: {title} ---")
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),lr=self.learning_rate, momentum=self.momentum)
        
        test_accuracies = []
        n_total_steps   = len(self.train_loader)

        for epoch in range(self.number_of_epochs):
            model.train()
            running_loss = 0.0
            for images, labels in self.train_loader:
                images = images.reshape(self.reshape_size).to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(images)
                loss = self.criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / n_total_steps

            _,_, model_accuracy_percent = self.test_model_fast(model)
            test_accuracies.append(model_accuracy_percent)

            if (((epoch + 1) % self.epoch_loss_print_period == 0)):
                print(f'Epoch [{epoch+1}/{self.number_of_epochs}], Loss: {avg_loss:.4f}, Accuracy: {model_accuracy_percent:.2f} %')
        
        return model, test_accuracies  

    def test_model(self, model, num_lyap_exp = 1):
        """
        Evaluates a model on the test dataset.
        
        Also computes the average and standard deviation of the top
        Lyapunov exponent(s) across the entire test set.

        Args:
            model (nn.Module): The model to evaluate.
            num_lyap_exp (int): The number of FTLEs to compute.

        Returns:
            tuple:
                - torch.Tensor: Average FTLE(s) over the test set.
                - torch.Tensor: Standard deviation of FTLE(s) over the test set.
                - float: Test accuracy percentage.
        """
        model.eval()
        n_correct  = 0
        n_samples  = 0
        lyaps_list = []
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images      = images.reshape(self.reshape_size).to(self.device)
                lyaps_batch = model.max_n_finite_time_lyapunov_exponents(images, num_lyap_exp)
                lyaps_list.append(lyaps_batch)
                
                labels  = labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                n_correct += (predicted == labels).sum().item()
                n_samples += labels.size(0)

                print(f"Total samples processed: {n_samples}")

        all_lyaps_gpu = torch.cat(lyaps_list, dim=0)
        average_lyap  = torch.mean(all_lyaps_gpu, dim=0)
        stddev_lyap   = torch.std(all_lyaps_gpu, dim=0)
        accuracy      = 100.0 * n_correct / n_samples

        return average_lyap, stddev_lyap, accuracy
    
    def test_model_fast(self, model):
        """
        Evaluates a model on the test dataset.
        
        Also computes the average and standard deviation of the top
        Lyapunov exponent(s) across the entire test set.

        Args:
            model (nn.Module): The model to evaluate.

        Returns:
            tuple:
                - torch.Tensor: Average FTLE(s) over the test set.
                - torch.Tensor: Standard deviation of FTLE(s) over the test set.
                - float: Test accuracy percentage.
        """
        model.eval()
        n_correct  = 0
        n_samples  = 0
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images      = images.reshape(self.reshape_size).to(self.device)
                
                labels  = labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                n_correct += (predicted == labels).sum().item()
                n_samples += labels.size(0)

                # print(f"Total samples processed: {n_samples}")

        accuracy = 100.0 * n_correct / n_samples

        return 0, 0, accuracy

    def visualize_ftle_on_data_points(self):
        """
        Generates a 2D scatter plot of the bottleneck layer's activations.
        
        Each point in the plot corresponds to a test set image, and its
        color represents the FTLE value calculated by the original model
        for that same image.
        
        Saves the plot to `self.mnist_2d_proj_plot_path`.
        """
        print("\n--- Visualizing FTLE Value for Each Data Point ---")
        if self.bottleneck_model is None:
            print("Bottleneck model not trained. Cannot visualize.")
            return
        
        self.original_model.eval()
        self.bottleneck_model.eval()
        
        lyaps_list        = []
        points_list       = []
        samples_processed = 0

        # disabling gradient speeds things up
        with torch.no_grad():
            for images, _ in self.test_loader:
                images = images.reshape(self.reshape_size).to(self.device)
                lyaps_batch      = self.original_model.max_n_finite_time_lyapunov_exponents(images)
                features_batch   = self.bottleneck_model.feature_extractor(images)
                bottleneck_batch = self.bottleneck_model.bottleneck[0](features_batch)
                
                lyaps_list.append(lyaps_batch)
                points_list.append(bottleneck_batch)

                samples_processed += images.size(0)
                print(f"Total samples processed: {samples_processed}")

        all_max_lyaps_gpu = torch.cat(lyaps_list, dim=0).cpu().numpy()
        all_points_gpu    = torch.cat(points_list, dim=0).cpu().numpy()

        with plt.style.context(["science"]):
            plt.rcParams['text.usetex'] = False  # <-- ADD THIS LINE
            plt.figure(figsize=(6, 4))

            scatter = plt.scatter(all_points_gpu[:, 0], all_points_gpu[:, 1], c=all_max_lyaps_gpu, cmap='coolwarm', alpha=0.8, s=1)

            plt.colorbar(scatter, label=r"Max FTLE ($\ln$ scale)")
            plt.title('Bottleneck Activations Colored by FTLE Value')
            plt.xlabel('Neuron 1 Activation')
            plt.ylabel('Neuron 2 Activation')
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.axis('equal')
            plt.savefig(self.mnist_2d_proj_plot_path, dpi=600)
            plt.close()
            print("Plot saved. :)")
    
    def plot_error_and_entropy_vs_lambda(self, ensemble_models, num_lyap_exp = 1, bin_edges=50):
        """
        Generates a plot of test error and predictive entropy vs. FTLE (lambda)
        for both clean and (optionally) adversarially attacked data.

        This function processes the test set. For each image, it calculates:
        1. The average FTLE (lambda) across an ensemble of models.
        2. The average predictive probability distribution across the ensemble.
        3. The predictive entropy (uncertainty) from this average distribution.
        4. The ensemble's prediction error (if the max avg. prob. is wrong).

        If `attack_size` is non-zero, it also performs an FGSM attack on
        the images and calculates these same four metrics for the
        perturbed data.

        It then bins the clean data and attacked data separately by their
        respective lambda values and plots the mean error and mean entropy
        for both sets on the same axes for comparison.

        Args:
            ensemble_models (list[nn.Module]): A list of trained models.
            num_lyap_exp (int): The number of FTLEs to analyze (e.g., 1 for lambda_1).
            bin_edges (int): The number of bins to use for bucketing lambda values.
            attack_size (float, optional): The epsilon value for the FGSM
                attack. If 0 (default), no attack is performed and only
                clean data is processed and plotted.
        """
        print("\n--- Generating Error and Entropy vs. Lambda Plot ---")
        if not ensemble_models:
            print("No models provided for ensemble. Cannot generate plot.")
            return

        all_lambdas   = []
        all_errors    = []
        all_entropies = []
        softmax       = nn.Softmax(dim=1)

        samples_processed = 0
        # with torch.no_grad():
        for images, labels in self.test_loader:
            images = images.reshape(self.reshape_size).to(self.device)
            labels = labels.to(self.device)

            batch_lambdas        = []
            batch_ensemble_probs = []

            for model in ensemble_models:
                model.eval()
                
                lyap_exp = model.max_n_finite_time_lyapunov_exponents(images, num_lyap_exp)
                batch_lambdas.append(lyap_exp.cpu().detach().numpy())
                outputs = model(images)
                probs   = softmax(outputs)
                batch_ensemble_probs.append(probs)

                samples_processed += images.size(0)
                print(f"Total samples processed: {samples_processed}")

            # compute the mean lyapunov exponenet(s) and entropy across the ensamle of models, per image
            avg_batch_lambda = np.mean(np.array(batch_lambdas), axis=0)
            all_lambdas.extend(avg_batch_lambda)

            ensemble_probs_for_batch =  torch.stack(batch_ensemble_probs)
            avg_probs_per_image      =  torch.mean(ensemble_probs_for_batch, dim=0) 
            entropies_per_image      = -torch.sum(avg_probs_per_image * torch.log(avg_probs_per_image + self.epsilon), dim=1)
            all_entropies.extend(entropies_per_image.cpu().detach().numpy())

            # compute model ensemble errors rates
            _, predicted = torch.max(avg_probs_per_image, 1)
            is_error = (predicted != labels)
            all_errors.extend((is_error).cpu().detach().numpy())

        all_errors    = np.array(all_errors).astype(float) # convert bool to float for averaging
        all_entropies = np.array(all_entropies)
        all_lambdas   = np.stack(all_lambdas)
        
        if num_lyap_exp == 1:     
            all_lambdas = all_lambdas.reshape(-1, 1)
        with plt.style.context(["science"]):
            plt.rcParams['text.usetex'] = False  # <-- ADD THIS LINE
            fig, axes = plt.subplots(num_lyap_exp, 1, figsize=(8, 2.4))

            # If only one subplot, axes is not iterable — make it a list
            if num_lyap_exp == 1:
                axes = [axes]

            fig.suptitle(r'Classification Error and Predictive Uncertainty vs. $\lambda_1^{(L)}(\mathbf{x})$')

            for i, ax in enumerate(axes):
                
                lambdas_i = all_lambdas[:, i] 

                min_lambda  = np.min(lambdas_i)
                max_lambda  = np.max(lambdas_i)
                lambda_bins = np.linspace(min_lambda, max_lambda, bin_edges)

                binned_lambda    = []
                binned_errors    = []
                binned_entropies = []

                # Iterate through bins and calculate average error and entropy + errors
                for j in range(len(lambda_bins) - 1):
                    lower_bound = lambda_bins[j]
                    upper_bound = lambda_bins[j+1]

                    if j == len(lambda_bins) - 2:
                        bin_indices = np.where((lambdas_i >= lower_bound) & (lambdas_i <= upper_bound))
                    else:
                        bin_indices = np.where((lambdas_i >= lower_bound) & (lambdas_i < upper_bound))

                    idx = bin_indices[0]
                    n = len(idx)
                    if n > 0:
                        # mean lambda in bin
                        avg_lambda_in_bin  = np.mean(lambdas_i[idx])

                        # error: all_errors are 0/1 floats; compute proportion p
                        p = np.mean(all_errors[idx])     # proportion of errors in bin (0..1)
                        avg_error_in_bin = p * 100.0     # convert to percent

                        # entropy: compute mean and standard error
                        ent_vals = all_entropies[idx]
                        avg_entropy_in_bin = np.mean(ent_vals)


                        binned_lambda.append(avg_lambda_in_bin)
                        binned_errors.append(avg_error_in_bin)

                        binned_entropies.append(avg_entropy_in_bin)

                # Plotting: error on left y-axis, entropy on right y-axis
                color_error = 'black'
                ax.set_xlabel(r'$\lambda_1^{(L)}(\mathbf{x})$')
                ax.set_ylabel('Test error ' +  r'$(\%)$', color=color_error)

                # convert to numpy arrays for plotting
                binned_lambda = np.array(binned_lambda)
                binned_errors = np.array(binned_errors)

                # main (clean) plot style: markers + errorbars with capsize
                if binned_lambda.size > 0:
                    ax.errorbar(binned_lambda, binned_errors, label='Error (%)', color=color_error, linewidth=1.5)
                
                ax.tick_params(axis='y', labelcolor=color_error)
                # ax.set_ylim(bottom=0)

                ax2 = ax.twinx()
                color_entropy = 'green'
                ax2.set_ylabel(r'$H$', color=color_entropy)

                binned_entropies = np.array(binned_entropies)

                if binned_lambda.size > 0:
                    ax2.errorbar(binned_lambda, binned_entropies, label='H', color=color_entropy, linewidth=1.5)

                ax2.tick_params(axis='y', labelcolor=color_entropy)
                # ax2.set_ylim(bottom=0)

                # ax.grid(True, linestyle='--', alpha=0.6)

                binned_lambda_array = np.array(binned_lambda)
                binned_errors_array = np.array(binned_errors)
                correlation_arg     = np.vstack([binned_lambda_array, binned_errors_array])
                correlation_array   = np.corrcoef(correlation_arg)
                self.correlation_list.append(correlation_array[0,1])

            plt.savefig(self.err_ent_vs_lmbd_plot_path, dpi=600)
            plt.close()
        print("Plot saved. :)")
        print(self.correlation_list)

    # def plot_error_and_entropy_vs_lambda_atk(self, ensemble_models, num_lyap_exp = 1, bin_edges=50, attack_size = 0):
    #     """
    #     Generates a plot of test error and predictive entropy vs. FTLE (lambda)
    #     for both clean and (optionally) adversarially attacked data.

    #     This function processes the test set. For each image, it calculates:
    #     1. The average FTLE (lambda) across an ensemble of models.
    #     2. The average predictive probability distribution across the ensemble.
    #     3. The predictive entropy (uncertainty) from this average distribution.
    #     4. The ensemble's prediction error (if the max avg. prob. is wrong).

    #     If `attack_size` is non-zero, it also performs an FGSM attack on
    #     the images and calculates these same four metrics for the
    #     perturbed data.

    #     It then bins the clean data and attacked data separately by their
    #     respective lambda values and plots the mean error and mean entropy
    #     for both sets on the same axes for comparison.

    #     Args:
    #         ensemble_models (list[nn.Module]): A list of trained models.
    #         num_lyap_exp (int): The number of FTLEs to analyze (e.g., 1 for lambda_1).
    #         bin_edges (int): The number of bins to use for bucketing lambda values.
    #         attack_size (float, optional): The epsilon value for the FGSM
    #             attack. If 0 (default), no attack is performed and only
    #             clean data is processed and plotted.
    #     """
    #     print("\n--- Generating Error and Entropy vs. Lambda Plot ---")
    #     if not ensemble_models:
    #         print("No models provided for ensemble. Cannot generate plot.")
    #         return

    #     all_lambdas   = []
    #     all_errors    = []
    #     all_entropies = []
    #     softmax       = nn.Softmax(dim=1)
        
    #     all_atk_lambdas   = []
    #     all_atk_errors    = []
    #     all_atk_entropies = []

    #     samples_processed = 0
    #     # with torch.no_grad():
    #     for images, labels in self.test_loader:
    #         images = images.reshape(self.reshape_size).to(self.device)
    #         labels = labels.to(self.device)

    #         batch_lambdas            = []
    #         batch_ensemble_probs     = []
    #         atk_batch_lambdas        = []
    #         atk_batch_ensemble_probs = []

    #         for model in ensemble_models:
    #             model.eval()
                
    #             lyap_exp = model.max_n_finite_time_lyapunov_exponents(images, num_lyap_exp)
    #             batch_lambdas.append(lyap_exp.cpu().detach().numpy())
    #             outputs = model(images)
    #             probs   = softmax(outputs)
    #             batch_ensemble_probs.append(probs)

    #             if(attack_size != 0):
    #                 images.requires_grad = True
    #                 outputs = model(images)
    #                 loss    = self.criterion(outputs, labels)
    #                 model.zero_grad()
    #                 loss.backward()
    #                 images_grads        = images.grad.data

    #                 # apply and evaluate FGSM attacks
    #                 perturbed_images    = self.fgsm_attack(images, attack_size, images_grads)
    #                 output_adv          = model(perturbed_images)

    #                 lyaps_batch = model.max_n_finite_time_lyapunov_exponents(perturbed_images, num_lyap_exp)
    #                 atk_batch_lambdas.append(lyaps_batch.cpu().detach().numpy())

    #                 atk_probs   = softmax(output_adv)
    #                 atk_batch_ensemble_probs.append(atk_probs)

    #             samples_processed += images.size(0)
    #             print(f"Total samples processed: {samples_processed}")

    #         # compute the mean lyapunov exponenet(s) and entropy across the ensamle of models, per image
    #         avg_batch_lambda = np.mean(np.array(batch_lambdas), axis=0)
    #         all_lambdas.extend(avg_batch_lambda)

    #         atk_avg_batch_lambda = np.mean(np.array(atk_batch_lambdas), axis=0)
    #         all_atk_lambdas.extend(atk_avg_batch_lambda)


    #         ensemble_probs_for_batch =  torch.stack(batch_ensemble_probs)
    #         avg_probs_per_image      =  torch.mean(ensemble_probs_for_batch, dim=0) 
    #         entropies_per_image      = -torch.sum(avg_probs_per_image * torch.log(avg_probs_per_image + self.epsilon), dim=1)
    #         all_entropies.extend(entropies_per_image.cpu().detach().numpy())

    #         atk_ensemble_probs_for_batch =  torch.stack(atk_batch_ensemble_probs)
    #         atk_avg_probs_per_image      =  torch.mean(atk_ensemble_probs_for_batch, dim=0) 
    #         atk_entropies_per_image      = -torch.sum(atk_avg_probs_per_image * torch.log(atk_avg_probs_per_image + self.epsilon), dim=1)
    #         all_atk_entropies.extend(atk_entropies_per_image.cpu().detach().numpy())


    #         # compute model ensemble errors rates
    #         _, predicted = torch.max(avg_probs_per_image, 1)
    #         is_error = (predicted != labels)
    #         all_errors.extend((is_error).cpu().detach().numpy())
    #         _, atk_predicted = torch.max(atk_avg_probs_per_image, 1)
    #         is_atk_error = (atk_predicted != labels)
    #         all_atk_errors.extend((is_atk_error).cpu().detach().numpy())

    #     all_errors    = np.array(all_errors).astype(float) # convert bool to float for averaging
    #     all_entropies = np.array(all_entropies)
    #     all_lambdas   = np.stack(all_lambdas)

    #     all_atk_errors    = np.array(all_atk_errors).astype(float) # convert bool to float for averaging
    #     all_atk_entropies = np.array(all_atk_entropies)
    #     all_atk_lambdas   = np.stack(all_atk_lambdas)
        
    #     if num_lyap_exp == 1:     
    #         all_lambdas = all_lambdas.reshape(-1, 1)
    #         all_atk_lambdas = all_atk_lambdas.reshape(-1, 1)
    #     with plt.style.context(["science"]):
    #         plt.rcParams['text.usetex'] = False  # <-- ADD THIS LINE
    #         fig, axes = plt.subplots(num_lyap_exp, 1, figsize=(10, 6))

    #         # If only one subplot, axes is not iterable — make it a list
    #         if num_lyap_exp == 1:
    #             axes = [axes]

    #         fig.suptitle(r'Classification Error and Predictive Uncertainty vs. $\lambda_i^{(L)}(\mathbf{x})$')

    #         for i, ax in enumerate(axes):
                
    #             lambdas_i = all_lambdas[:, i] 
    #             atk_lambdas_i = all_atk_lambdas[:, i] 

    #             min_lambda  = np.min(lambdas_i)
    #             max_lambda  = np.max(lambdas_i)
    #             lambda_bins = np.linspace(min_lambda, max_lambda, bin_edges)

    #             atk_min_lambda  = np.min(atk_lambdas_i)
    #             atk_max_lambda  = np.max(atk_lambdas_i)
    #             atk_lambda_bins = np.linspace(atk_min_lambda, atk_max_lambda, bin_edges)

    #             binned_lambda    = []
    #             binned_errors    = []
    #             binned_errors_se = []   # <-- standard error for error %
    #             binned_entropies = []
    #             binned_entropy_se = []  # <-- standard error for entropy

    #             atk_binned_lambda    = []
    #             atk_binned_errors    = []
    #             atk_binned_errors_se = []
    #             atk_binned_entropies = []
    #             atk_binned_entropy_se = []

    #             # Iterate through bins and calculate average error and entropy + errors
    #             for j in range(len(lambda_bins) - 1):
    #                 lower_bound = lambda_bins[j]
    #                 upper_bound = lambda_bins[j+1]

    #                 if j == len(lambda_bins) - 2:
    #                     bin_indices = np.where((lambdas_i >= lower_bound) & (lambdas_i <= upper_bound))
    #                 else:
    #                     bin_indices = np.where((lambdas_i >= lower_bound) & (lambdas_i < upper_bound))

    #                 idx = bin_indices[0]
    #                 n = len(idx)
    #                 if n > 0:
    #                     # mean lambda in bin
    #                     avg_lambda_in_bin  = np.mean(lambdas_i[idx])

    #                     # error: all_errors are 0/1 floats; compute proportion p
    #                     p = np.mean(all_errors[idx])     # proportion of errors in bin (0..1)
    #                     avg_error_in_bin = p * 100.0     # convert to percent

    #                     # binomial standard error for proportion (as percent)
    #                     # if n==1 this becomes 0; avoids divide-by-zero because n>0
    #                     error_se_pct = math.sqrt(p * (1.0 - p) / n) * 100.0

    #                     # entropy: compute mean and standard error
    #                     ent_vals = all_entropies[idx]
    #                     avg_entropy_in_bin = np.mean(ent_vals)
    #                     entropy_se = np.std(ent_vals, ddof=0) / math.sqrt(n)  # std error of mean

    #                     binned_lambda.append(avg_lambda_in_bin)
    #                     binned_errors.append(avg_error_in_bin)
    #                     binned_errors_se.append(error_se_pct)
    #                     binned_entropies.append(avg_entropy_in_bin)
    #                     binned_entropy_se.append(entropy_se)

    #             # same for attacked bins
    #             for j in range(len(atk_lambda_bins) - 1):
    #                 atk_lower_bound = atk_lambda_bins[j]
    #                 atk_upper_bound = atk_lambda_bins[j+1]

    #                 if j == len(atk_lambda_bins) - 2:
    #                     atk_bin_indices = np.where((atk_lambdas_i >= atk_lower_bound) & (atk_lambdas_i <= atk_upper_bound))
    #                 else:
    #                     atk_bin_indices = np.where((atk_lambdas_i >= atk_lower_bound) & (atk_lambdas_i < atk_upper_bound))

    #                 idx = atk_bin_indices[0]
    #                 n = len(idx)
    #                 if n > 0:
    #                     atk_avg_lambda_in_bin  = np.mean(atk_lambdas_i[idx])
    #                     p = np.mean(all_atk_errors[idx])
    #                     atk_avg_error_in_bin = p * 100.0
    #                     atk_error_se_pct = math.sqrt(p * (1.0 - p) / n) * 100.0

    #                     ent_vals = all_atk_entropies[idx]
    #                     atk_avg_entropy_in_bin = np.mean(ent_vals)
    #                     atk_entropy_se = np.std(ent_vals, ddof=0) / math.sqrt(n)

    #                     atk_binned_lambda.append(atk_avg_lambda_in_bin)
    #                     atk_binned_errors.append(atk_avg_error_in_bin)
    #                     atk_binned_errors_se.append(atk_error_se_pct)
    #                     atk_binned_entropies.append(atk_avg_entropy_in_bin)
    #                     atk_binned_entropy_se.append(atk_entropy_se)

    #             # Plotting: error on left y-axis, entropy on right y-axis
    #             color_error = 'black'
    #             atk_color_error = 'red'
    #             ax.set_xlabel(r'$\lambda_i^{(L)}(\mathbf{x})$')
    #             ax.set_ylabel('Error' +  r'$(\%)$', color=color_error)

    #             # convert to numpy arrays for plotting
    #             binned_lambda = np.array(binned_lambda)
    #             binned_errors = np.array(binned_errors)
    #             binned_errors_se = np.array(binned_errors_se)

    #             atk_binned_lambda = np.array(atk_binned_lambda)
    #             atk_binned_errors = np.array(atk_binned_errors)
    #             atk_binned_errors_se = np.array(atk_binned_errors_se)

    #             # main (clean) plot style: markers + errorbars with capsize
    #             if binned_lambda.size > 0:
    #                 ax.errorbar(binned_lambda, binned_errors, yerr=binned_errors_se,
    #                             fmt='-o', capsize=3, label='Error (%)', color=color_error)
    #             if atk_binned_lambda.size > 0:
    #                 ax.errorbar(atk_binned_lambda, atk_binned_errors, yerr=atk_binned_errors_se,
    #                             fmt='-o', capsize=3, label='Error (attacked) (%)', color=atk_color_error)

    #             ax.tick_params(axis='y', labelcolor=color_error)
    #             ax.set_ylim(bottom=0)

    #             ax2 = ax.twinx()
    #             color_entropy = 'green'
    #             atk_color_entropy = 'blue'
    #             ax2.set_ylabel(r'$H$', color=color_entropy)

    #             binned_entropies = np.array(binned_entropies)
    #             binned_entropy_se = np.array(binned_entropy_se)
    #             atk_binned_entropies = np.array(atk_binned_entropies)
    #             atk_binned_entropy_se = np.array(atk_binned_entropy_se)

    #             if binned_lambda.size > 0:
    #                 ax2.errorbar(binned_lambda, binned_entropies, yerr=binned_entropy_se,
    #                             fmt='-s', capsize=3, label='H', color=color_entropy)
    #             if atk_binned_lambda.size > 0:
    #                 ax2.errorbar(atk_binned_lambda, atk_binned_entropies, yerr=atk_binned_entropy_se,
    #                             fmt='-s', capsize=3, label='H (attacked)', color=atk_color_entropy)

    #             ax2.tick_params(axis='y', labelcolor=color_entropy)
    #             ax2.set_ylim(bottom=0)

    #             ax.grid(True, linestyle='--', alpha=0.6)

    #             binned_lambda_array = np.array(binned_lambda)
    #             binned_errors_array = np.array(binned_errors)
    #             correlation_arg     = np.vstack([binned_lambda_array, binned_errors_array])
    #             correlation_array   = np.corrcoef(correlation_arg)
    #             self.correlation_list.append(correlation_array[0,1])

    #             atk_binned_lambda_array = np.array(atk_binned_lambda)
    #             atk_binned_errors_array = np.array(atk_binned_errors)
    #             atk_correlation_arg     = np.vstack([atk_binned_lambda_array, atk_binned_errors_array])
    #             atk_correlation_array   = np.corrcoef(atk_correlation_arg)
    #             self.atk_correlation_list.append(atk_correlation_array[0,1])

    #         plt.savefig(self.err_ent_vs_lmbd_plot_path, dpi=600)
    #         plt.close()
    #     print("Plot saved. :)")
    #     print(self.correlation_list, self.atk_correlation_list)

    def analyze_attacks(self, model, attack_sizes):
        """
        Runs adversarial attacks (FGSM) for various strengths and plots examples.

        It calls `test_attack` for each epsilon in `attack_sizes` and then
        generates a plot comparing original images to their successful
        adversarial counterparts.

        Args:
            model (nn.Module): The model to attack.
            attack_sizes (list[float]): A list of attack strengths (epsilons).
        """
        accuracies   = []
        all_examples = []

        for attack_size in attack_sizes:
            acc, ex, _, _ = self.test_attack(model, attack_size)
            accuracies.append(acc)
            all_examples.append(ex)
        with plt.style.context(["science"]):
            plt.rcParams['text.usetex'] = False  # <-- ADD THIS LINE
            plt.figure(figsize=(10, 8))

            try:
                examples_to_show = all_examples[2] #TODO have this update based on the list size
                cnt = 0
                for i in range(len(examples_to_show)):
                    cnt += 1
                    orig_pred, adv_pred, orig_img, adv_img = examples_to_show[i]
                    
                    # original image
                    plt.subplot(2, self.numb_adversaila_examples, cnt)
                    plt.xticks([], [])
                    plt.yticks([], [])
                    plt.title(f"Original: {orig_pred}")
                    plt.imshow(orig_img, cmap="gray")
                    
                    # adversarial images
                    plt.subplot(2, self.numb_adversaila_examples, cnt + self.numb_adversaila_examples)
                    plt.xticks([], [])
                    plt.yticks([], [])
                    plt.title(f"Adversarial: {adv_pred}")
                    plt.imshow(adv_img, cmap="gray")
                    
                    if cnt == self.numb_adversaila_examples:
                        break
                        
                plt.savefig(self.attack_plot_path, dpi=600)
                plt.close()
                print("Plot saved. :)")

            except IndexError:
                print("\nNot enough adversarial examples found to display for that attack size.")
                print("Try training the model for more epochs or increasing attack size.")

    def test_attack(self, model, attack_size, num_lyap_exp=1):
        """
        Tests the model's accuracy under an FGSM adversarial attack and
        computes FTLE statistics for the attacked images.

        This function iterates through the test set, performs an FGSM attack
        on each batch, and then:
        1. Calculates the model's classification accuracy on the perturbed images.
        2. Gathers a list of successful adversarial examples (where the
           original prediction was correct, but the attacked one was wrong).
        3. Computes the Finite-Time Lyapunov Exponents (FTLEs) for all
           perturbed images.

        Args:
            model (nn.Module): The model to attack.
            attack_size (float): The epsilon value (strength) of the attack.
            num_lyap_exp (int, optional): The number of FTLEs to compute for
                each attacked image. Defaults to 1.

        Returns:
            tuple:
                - float: The model's accuracy on the perturbed dataset (0.0 to 1.0).
                - list: A list of (orig_pred, adv_pred, orig_img, adv_img)
                        tuples for successful attacks.
                - torch.Tensor: A 1D tensor containing the average (mean)
                                of each of the `num_lyap_exp` FTLEs,
                                averaged across the entire test set.
                - torch.Tensor: A 1D tensor containing the standard deviation
                                of each of the `num_lyap_exp` FTLEs,
                                calculated across the entire test set.
        """
        model.eval() 
        n_correct    = 0
        adv_examples = []
        lyaps_list = []
        

        for images, labels in self.test_loader:
            # compute gradients for Fast Gradient Sign Method (FGSM) attacks on input images
            images     = images.reshape(self.reshape_size).to(self.device)
            labels     = labels.to(self.device)
            images.requires_grad = True
            outputs = model(images)
            loss    = self.criterion(outputs, labels)
            model.zero_grad()
            loss.backward()
            images_grads        = images.grad.data
            _, init_predictions = torch.max(outputs.data, 1)

            # apply and evaluate FGSM attacks
            perturbed_images    = self.fgsm_attack(images, attack_size, images_grads)
            output_adv          = model(perturbed_images)
            _, final_predictions = torch.max(output_adv.data, 1)

            lyaps_batch = model.max_n_finite_time_lyapunov_exponents(perturbed_images, num_lyap_exp)
            lyaps_list.append(lyaps_batch)

            n_correct += (final_predictions == labels).sum().item()

            for index in range(len(final_predictions)):
                init_pred_item  = init_predictions[index].item()
                final_pred_item = final_predictions[index].item()
                label_item      = labels[index].item()

                was_match   = ( init_pred_item == label_item)
                is_mismatch = (final_pred_item != label_item)
                
                if (was_match and is_mismatch and (len(adv_examples) < self.numb_adversaila_examples)):
                    image_ref     = images.reshape(self.unreshape_size).to(self.device)
                    perturbed_ref = perturbed_images.reshape(self.unreshape_size).to(self.device)
                    adv_ex        = perturbed_ref[index].squeeze().detach().cpu().numpy()
                    orig_ex       = image_ref[index].squeeze().detach().cpu().numpy()
                    adv_examples.append((init_pred_item, final_pred_item, orig_ex, adv_ex))
            
        all_lyaps_gpu = torch.cat(lyaps_list, dim=0)
        average_lyap  = torch.mean(all_lyaps_gpu, dim=0)
        stddev_lyap   = torch.std(all_lyaps_gpu, dim=0)

        final_acc = n_correct / float(len(self.test_loader.dataset))
        print(f"Epsilon: {attack_size}\tTest Accuracy = {n_correct} / {len(self.test_loader.dataset)} = {final_acc:.4f}")

        return final_acc, adv_examples, average_lyap, stddev_lyap
    
    def fgsm_attack(self, images, attack_size, image_grads):
        """
        Performs the Fast Gradient Sign Method (FGSM) attack.

        Args:
            images (torch.Tensor): The original input images.
            attack_size (float): The attack strength (epsilon).
            image_grads (torch.Tensor): The gradients of the loss w.r.t. the images.

        Returns:
            torch.Tensor: The perturbed (adversarial) images.
        """
        sign_image_grads = image_grads.sign()
        perturbed_images = images + attack_size * sign_image_grads
        perturbed_images = torch.clamp(perturbed_images, 0, 1) # valid range [0, 1]
        return perturbed_images


class DesiredPlot(Enum):
    TRAINING     = 1
    FTLE_2D      = 2
    ENTROPY      = 3
    AVERAGE      = 4
    ATTACK       = 5
    STAT_TABLE   = 6
    ENTROPY_ATK  = 7

def main():
    classifier = MNISTClassification(debug=False) # set debug flag to True if you want code to run in 1x minutes instead of 10x minutes
    
    # change as desired
    desired_plot            = DesiredPlot.ATTACK #Prof Rainer Engelken try each of the options for this
    num_models_averaged     = 5
    hidden_layer_sizes_list = range(10, 120, 50)
    attack_sizes            = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    num_lyap_exp            = 3
    entropy_attack          = 0.2                  # set to zero for no attack plots

    if desired_plot == DesiredPlot.TRAINING:                
        classifier.number_of_epochs = 200
        model                       = TanhSoftmaxNet().to(classifier.device)
        _, train_losses             = classifier.train_model(model, title="Phase 1: Original Model Training")
        with plt.style.context(["science"]):
            plt.rcParams['text.usetex'] = False  # <-- ADD THIS LINE

            plt.figure(figsize=(8,5))
            plt.plot(range(1, len(train_losses)+1), train_losses)
            plt.xlabel('Epoch')
            plt.ylabel('Classifcation Accuracy')
            plt.title('Classifcation Accuracy per Epoch')
            plt.grid(True)
            plt.savefig(classifier.training_acc_plot_path, dpi=600)
            plt.close()
            print("Plot saved. :)")

    if desired_plot == DesiredPlot.FTLE_2D:
        original_model               = TanhSoftmaxNet().to(classifier.device)
        classifier.original_model    = original_model
        trained_original_model, _    = classifier.train_model(original_model, title="Phase 1: Original Model Training")

        # last layer not included since that's the classification layer
        feature_extractor = trained_original_model.network[:-1]

        # freeze weights
        for param in feature_extractor.parameters():
            param.requires_grad = False

        bottleneck_model                = BottleneckNet(feature_extractor).to(classifier.device)
        classifier.bottleneck_model, _  = classifier.train_model(bottleneck_model, title="Phase 2: Bottleneck Fine-Tuning")
        classifier.visualize_ftle_on_data_points()
    
    if desired_plot == DesiredPlot.ENTROPY:
        ensemble_models = []
        for num in range(num_models_averaged):
            print(f"\n--- Training Ensemble Model {num+1}/{num_models_averaged} ---")
            untrained_model     = TanhSoftmaxNet().to(classifier.device)
            trained_model ,_    = classifier.train_model(untrained_model, title=f"Ensemble Model {num+1} Training")
            ensemble_models.append(trained_model)
        
        classifier.plot_error_and_entropy_vs_lambda(ensemble_models)
        # classifier.plot_error_and_entropy_vs_lambda_atk(ensemble_models, num_lyap_exp, attack_size=entropy_attack)

    if desired_plot == DesiredPlot.AVERAGE:
        avg_avg_eig1_list      = []
        avg_std_dev_eig1_list  = []

        for hidden_layer_size in hidden_layer_sizes_list:
            print(f"\n--- Training and Testing with {hidden_layer_size} Nodes Per Layer Hidden---")
            average_eig1_size_i_list = []
            std_eig1_size_i_list     = []

            for num in range(num_models_averaged):
                untrained_model   = TanhSoftmaxNet(hidden_layer_size = hidden_layer_size).to(classifier.device)
                trained_model ,_  = classifier.train_model(untrained_model, title=f"Model {num+1}/{num_models_averaged} Training")
                average_eig1, standard_dev_eig1, _ = classifier.test_model(trained_model)
                
                average_eig1_size_i_list.append(average_eig1.item())
                std_eig1_size_i_list.append(standard_dev_eig1.item())

            avg_avg_eig1     = sum(average_eig1_size_i_list) / num_models_averaged
            avg_std_dev_eig1 = sum(std_eig1_size_i_list) / num_models_averaged
            
            avg_avg_eig1_list.append(avg_avg_eig1)
            avg_std_dev_eig1_list.append(avg_std_dev_eig1)
        
        with plt.style.context(["science"]):
            plt.rcParams['text.usetex'] = False  # <-- ADD THIS LINE
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 2))
            ax1.semilogx(hidden_layer_sizes_list,    avg_avg_eig1_list, label='Average Accuracy')
            ax2.semilogx(hidden_layer_sizes_list, avg_std_dev_eig1_list, label='Std Dev of Accuracy')

            ax1.set_xlabel('N')
            ax2.set_xlabel('N')
            ax1.set_ylabel(r'$\langle \lambda_1^{(L)}(\bf{x}$' + r'$) \rangle$')
            ax2.set_ylabel(r'Std[$\lambda_1^{(L)}(\bf{x}$)]')

            plt.savefig(classifier.acc_vs_num_plot_path, dpi=600)
            plt.close()
            print("Plot saved. :)")

    if desired_plot == DesiredPlot.ATTACK:
        untrained_model  = TanhSoftmaxNet().to(classifier.device)
        trained_model ,_ = classifier.train_model(untrained_model, title="Phase 1: Model Training")
        classifier.analyze_attacks(trained_model, attack_sizes)

    if desired_plot == DesiredPlot.STAT_TABLE:
        average_eig_list = []
        std_eig_list     = []
        accuracy_list    = []
        atk_average_eig_list = []
        atk_std_eig_list     = []
        atk_accuracy_list    = []

        for num in range(num_models_averaged):
            untrained_model   = TanhSoftmaxNet().to(classifier.device)
            trained_model ,_  = classifier.train_model(untrained_model, title=f"Model {num+1}/{num_models_averaged} Training")
            average_eig, standard_dev_eig, accuracy = classifier.test_model(trained_model, num_lyap_exp)
            
            sub_atk_average_eig_list = []
            sub_atk_std_eig_list     = []
            sub_atk_accuracy_list    = []
            for attack_size in attack_sizes:
                atk_accuracy, _, atk_average_lyap, atk_stddev_lyap = classifier.test_attack(trained_model, attack_size, num_lyap_exp)

                sub_atk_average_eig_list.append(atk_average_lyap.tolist())
                sub_atk_std_eig_list.append(atk_stddev_lyap.tolist())
                sub_atk_accuracy_list.append(atk_accuracy)
                
            average_eig_list.append(average_eig)
            std_eig_list.append(standard_dev_eig)
            accuracy_list.append(accuracy)

            atk_average_eig_list.append(sub_atk_average_eig_list)
            atk_std_eig_list.append(sub_atk_std_eig_list)
            atk_accuracy_list.append(sub_atk_accuracy_list)
            
        avg_acc        = sum(accuracy_list) / num_models_averaged
        std_acc        = statistics.stdev(accuracy_list)
        atk_avg_tensor = torch.tensor(atk_accuracy_list)

        print(f'The average of {num_models_averaged} runs was an an average of {avg_acc} with a standard deviation of {std_acc} for the percent of pictures correctly classified.')
        for i in range(len(attack_sizes)):
            atk_avg_tensor_i = atk_avg_tensor[:,i]
            avg_atk_acc      = atk_avg_tensor_i.mean().item()
            std_atk_acc      = atk_avg_tensor_i.std().item()
            print(f'The average of {num_models_averaged} runs was an an average of {avg_atk_acc} with a standard deviation of {std_atk_acc} for the percent of ATTACKED pictures correctly classified.')
            print(f'Thats a percent difference of {(avg_atk_acc - avg_acc)/avg_acc} in the average and a percent difference of {(std_atk_acc - std_acc)/std_acc} from the regular to attacked images.')
            
        avg_eig_tensor         = torch.stack(average_eig_list)
        std_eig_tensor         = torch.stack(std_eig_list)
        atk_avg_atk_eig_tensor = torch.tensor(atk_average_eig_list)
        atk_std_eig_tensor     = torch.tensor(atk_std_eig_list)

        for i in range(num_lyap_exp):
            avg_avg_of_eigi     = avg_eig_tensor[:,i].mean().item()
            avg_std_dev_of_eigi = std_eig_tensor[:,i].mean().item()
            print(f'The average of {num_models_averaged} runs was an an average of {avg_avg_of_eigi} with a standard deviation of {avg_std_dev_of_eigi} for mu{i+1}.')

            for j in range(len(attack_sizes)):
                avg_atk_avg_of_eigi     = atk_avg_atk_eig_tensor[:,j, i].mean().item()
                avg_atk_std_dev_of_eigi = atk_std_eig_tensor[:,j,i].std().item()
                print(f'The average of {num_models_averaged} runs was an an average of {avg_atk_avg_of_eigi} with a standard deviation of {avg_atk_std_dev_of_eigi} for mu{i+1} and attack size of {attack_sizes[j]}.')
                print(f'Thats a percent difference of {(avg_atk_avg_of_eigi - avg_avg_of_eigi)/avg_avg_of_eigi} in the average and a percent difference of {(avg_atk_std_dev_of_eigi - avg_std_dev_of_eigi)/avg_std_dev_of_eigi} from the regular to attacked images.')

    if desired_plot == DesiredPlot.ENTROPY_ATK:
        ensemble_models = []
        for num in range(num_models_averaged):
            print(f"\n--- Training Ensemble Model {num+1}/{num_models_averaged} ---")
            untrained_model     = TanhSoftmaxNet().to(classifier.device)
            trained_model ,_    = classifier.train_model(untrained_model, title=f"Ensemble Model {num+1} Training")
            ensemble_models.append(trained_model)

        classifier.plot_error_and_entropy_vs_lambda_atk(ensemble_models, num_lyap_exp, attack_size=entropy_attack)

if __name__ == "__main__":
    main()