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
        max_lyapunov_exponents = torch.log10(max_singular_values)
        
        number_of_hidden_layers_tensor = torch.tensor(self.numb_hidden_layers, dtype=torch.float64, device=max_lyapunov_exponents.device)
        output = max_lyapunov_exponents / number_of_hidden_layers_tensor
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
    def __init__(self, learning_rate: float = 5e-3, momentum: float = 0.9, number_of_epochs: int = 25, batch_size: int = 64, debug: bool = False) -> None:
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
            self.number_of_epochs = 2
        
        transform         = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) 
        train_dataset     = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        test_dataset      = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)
        
        if debug:
            subset_indices = range(120)
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

        # training_model config
        self.reshape_size = (-1, 28 * 28)
        self.epoch_loss_print_period = 5

        # lot_error_and_entropy_vs_lambda config
        self.epsilon = 1e-9 # for log stability

        # train_attack config
        self.numb_adversaila_examples = 5

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
        
        train_losses    = []
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
            train_losses.append(avg_loss)

            _,_, model_accuracy_percent = self.test_model(model)
            test_accuracies.append(model_accuracy_percent)

            if (((epoch + 1) % self.epoch_loss_print_period == 0)):
                print(f'Epoch [{epoch+1}/{self.number_of_epochs}], Loss: {avg_loss:.4f}, Accuracy: {model_accuracy_percent:.2f} %')
        
        return model

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
            plt.figure(figsize=(6, 4))

            scatter = plt.scatter(all_points_gpu[:, 0], all_points_gpu[:, 1], c=all_max_lyaps_gpu, cmap='coolwarm', alpha=0.8, s=15)

            plt.colorbar(scatter, label="Max FTLE ($\log_{10}$ scale)")
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
        Generates a plot of test error and predictive entropy vs. FTLE (lambda).

        This function processes the test set, and for each image, it calculates:
        1. The average FTLE (lambda) across an ensemble of models.
        2. The average predictive probability distribution across the ensemble.
        3. The predictive entropy (uncertainty) from this average distribution.
        4. The ensemble's prediction error (if the max avg. prob. is wrong).

        It then bins the data by the average lambda value and plots the
        mean error and mean entropy within each bin.

        Args:
            ensemble_models (list[nn.Module]): A list of trained models.
            num_lyap_exp (int): The number of FTLEs to analyze (e.g., 1 for lambda_1).
            bin_edges (int): The number of bins to use for lambda.
        """
        print("\n--- Generating Error and Entropy vs. Lambda_1 Plot ---")
        if not ensemble_models:
            print("No models provided for ensemble. Cannot generate plot.")
            return

        all_lambdas   = []
        all_errors    = []
        all_entropies = []
        softmax       = nn.Softmax(dim=1)
        

        samples_processed = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.reshape(self.reshape_size).to(self.device)
                labels = labels.to(self.device)

                batch_lambdas        = []
                batch_ensemble_probs = []

                for model in ensemble_models:
                    model.eval()
                    lyap_exp = model.max_n_finite_time_lyapunov_exponents(images, num_lyap_exp)
                    batch_lambdas.append(lyap_exp.cpu().numpy())
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
                all_entropies.extend(entropies_per_image.cpu().numpy())

                # compute model ensemble errors rates
                _, predicted = torch.max(avg_probs_per_image, 1)
                is_error = (predicted != labels)
                all_errors.extend((is_error).cpu().numpy())

        all_errors    = np.array(all_errors).astype(float) # convert bool to float for averaging
        all_entropies = np.array(all_entropies)
        all_lambdas   = np.stack(all_lambdas)
        
        if num_lyap_exp == 1:     
            all_lambdas = all_lambdas.reshape(-1, 1)
        with plt.style.context(["science"]):
            fig, axes = plt.subplots(num_lyap_exp, 1, figsize=(10, 4))

            # If only one subplot, axes is not iterable — make it a list
            if num_lyap_exp == 1:
                axes = [axes]

            fig.suptitle('Classification Error and Predictive Uncertainty vs. $\lambda_i^{(L)}(\mathbf{x})$')

            for i, ax in enumerate(axes):
                
                lambdas_i = all_lambdas[:, i] 

                min_lambda  = np.min(lambdas_i)
                max_lambda  = np.max(lambdas_i)
                lambda_bins = np.linspace(min_lambda, max_lambda, bin_edges)

                binned_lambda    = []
                binned_errors    = []
                binned_entropies = []

                # Iterate through bins and calculate average error and entropy
                for i in range(len(lambda_bins) - 1):
                    lower_bound = lambda_bins[i]
                    upper_bound = lambda_bins[i+1]
                    
                    if i == len(lambda_bins) - 2: # Include the max value in the last bin
                        bin_indices = np.where((lambdas_i >= lower_bound) & (lambdas_i <= upper_bound))
                    else:
                        bin_indices = np.where((lambdas_i >= lower_bound) & (lambdas_i < upper_bound))
                    
                    if len(bin_indices[0]) > 0:
                        avg_lambda_in_bin  = np.mean(  lambdas_i[bin_indices])
                        avg_error_in_bin   = np.mean(   all_errors[bin_indices]) * 100 # Convert to percentage
                        avg_entropy_in_bin = np.mean(all_entropies[bin_indices])

                        binned_lambda.append(avg_lambda_in_bin)
                        binned_errors.append(avg_error_in_bin)
                        binned_entropies.append(avg_entropy_in_bin)

                color_error = 'black'
                ax.set_xlabel(r'$\lambda_i^{(L)}(\mathbf{x})$')
                ax.set_ylabel('Test Error' +  '$(\%)$', color=color_error)
                ax.plot(binned_lambda, binned_errors, color=color_error, label='Test Error' +  '$(\%)$')
                ax.tick_params(axis='y', labelcolor=color_error)
                ax.set_ylim(bottom=0) # Error should not go below 0

                ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
                color_entropy = 'green'
                ax2.set_ylabel(r'$H$', color=color_entropy)
                ax2.plot(binned_lambda, binned_entropies, color=color_entropy, label='H')
                ax2.tick_params(axis='y', labelcolor=color_entropy)
                ax2.set_ylim(bottom=0) # Entropy should not go below 0

                ax.grid(True, linestyle='--', alpha=0.6)

            plt.savefig(self.err_ent_vs_lmbd_plot_path, dpi=600)
            plt.close()
        print("Plot saved. :)")
        
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
            acc, ex = self.test_attack(model, attack_size)
            accuracies.append(acc)
            all_examples.append(ex)
        with plt.style.context(["science"]):
            plt.figure(figsize=(10, 8))

            try:
                examples_to_show = all_examples[3] #TODO have this update based on the list size
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

    def test_attack(self, model, attack_size):
        """
        Tests the model's accuracy under an FGSM adversarial attack.

        Args:
            model (nn.Module): The model to attack.
            attack_size (float): The epsilon value (strength) of the attack.

        Returns:
            tuple:
                - float: The model's accuracy on the perturbed dataset.
                - list: A list of (orig_pred, adv_pred, orig_img, adv_img)
                        tuples for successful attacks.
        """
        n_correct    = 0
        adv_examples = []
        model.eval() 

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
            perturbed_data      = self.fgsm_attack(images, attack_size, images_grads)
            output_adv          = model(perturbed_data)
            _, final_predictions = torch.max(output_adv.data, 1)

            n_correct += (final_predictions == labels).sum().item()
            for index in range(len(final_predictions)):
                init_pred_item  = init_predictions[index].item()
                final_pred_item = final_predictions[index].item()
                label_item      = labels[index].item()

                was_match   = ( init_pred_item == label_item)
                is_mismatch = (final_pred_item != label_item)
                
                if (was_match and is_mismatch and (len(adv_examples) < self.numb_adversaila_examples)):
                    image_ref     = images.reshape(self.reshape_size).to(self.device)
                    perturbed_ref = perturbed_data.reshape(self.reshape_size).to(self.device)
                    adv_ex        = perturbed_ref[index].squeeze().detach().cpu().numpy()
                    orig_ex       = image_ref[index].squeeze().detach().cpu().numpy()
                    adv_examples.append((init_pred_item, final_pred_item, orig_ex, adv_ex))

        final_acc = n_correct / float(len(self.test_loader.dataset))
        print(f"Epsilon: {attack_size}\tTest Accuracy = {n_correct} / {len(self.test_loader.dataset)} = {final_acc:.4f}")

        return final_acc, adv_examples
    
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
    FTLE_2D    = 1
    ENTROPY    = 2
    AVERAGE    = 3
    ATTACK     = 4
    STAT_TABLE = 5

def main():
    classifier = MNISTClassification(debug=True) # set debug flag to True if you want code to run in 1x minutes instead of 10x minutes
    
    # change as desired
    num_models_averaged     = 2
    hidden_layer_sizes_list = range(10, 120, 20)
    attack_sizes            = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    num_lyap_exp            = 3
    desired_plot            =  DesiredPlot.STAT_TABLE #Prof Rainer Engelken try each of the options for this
    

    if desired_plot == DesiredPlot.FTLE_2D:
        original_model            = TanhSoftmaxNet().to(classifier.device)
        classifier.original_model = original_model
        trained_original_model    = classifier.train_model(original_model, title="Phase 1: Original Model Training")

        # last layer not included since that's the classification layer
        feature_extractor = trained_original_model.network[:-1]

        # freeze weights
        for param in feature_extractor.parameters():
            param.requires_grad = False

        bottleneck_model            = BottleneckNet(feature_extractor).to(classifier.device)
        classifier.bottleneck_model = classifier.train_model(bottleneck_model, title="Phase 2: Bottleneck Fine-Tuning")
        classifier.visualize_ftle_on_data_points()
    
    if desired_plot == DesiredPlot.ENTROPY:
        ensemble_models = []
        for num in range(num_models_averaged):
            print(f"\n--- Training Ensemble Model {num+1}/{num_models_averaged} ---")
            untrained_model = TanhSoftmaxNet().to(classifier.device)
            trained_model   = classifier.train_model(untrained_model, title=f"Ensemble Model {num+1} Training")
            ensemble_models.append(trained_model)
        
        classifier.plot_error_and_entropy_vs_lambda(ensemble_models, num_lyap_exp)

    if desired_plot == DesiredPlot.AVERAGE:
        avg_avg_eig1_list      = []
        avg_std_dev_eig1_list  = []

        for hidden_layer_size in hidden_layer_sizes_list:
            print(f"\n--- Training and Testing with {hidden_layer_size} Nodes Per Layer Hidden---")
            average_eig1_size_i_list = []
            std_eig1_size_i_list     = []

            for num in range(num_models_averaged):
                untrained_model   = TanhSoftmaxNet(hidden_layer_size = hidden_layer_size).to(classifier.device)
                trained_model     = classifier.train_model(untrained_model, title=f"Model {num+1}/{num_models_averaged} Training")
                average_eig1, standard_dev_eig1, _ = classifier.test_model(trained_model)
                
                average_eig1_size_i_list.append(average_eig1.item())
                std_eig1_size_i_list.append(standard_dev_eig1.item())

            avg_avg_eig1     = sum(average_eig1_size_i_list) / num_models_averaged
            avg_std_dev_eig1 = sum(std_eig1_size_i_list) / num_models_averaged
            
            avg_avg_eig1_list.append(avg_avg_eig1)
            avg_std_dev_eig1_list.append(avg_std_dev_eig1)
        
        with plt.style.context(["science"]):
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
        untrained_model = TanhSoftmaxNet().to(classifier.device)
        trained_model   = classifier.train_model(untrained_model, title="Phase 1: Model Training")
        classifier.analyze_attacks(trained_model, attack_sizes)

    if desired_plot == DesiredPlot.STAT_TABLE:
        average_eig_list = []
        std_eig_list     = []
        accuracy_list    = []

        for num in range(num_models_averaged):
            untrained_model   = TanhSoftmaxNet().to(classifier.device)
            trained_model     = classifier.train_model(untrained_model, title=f"Model {num+1}/{num_models_averaged} Training")
            average_eig, standard_dev_eig, accuracy = classifier.test_model(trained_model, num_lyap_exp)
            
            average_eig_list.append(average_eig)
            std_eig_list.append(standard_dev_eig)
            accuracy_list.append(accuracy)

        avg_acc             = sum(accuracy_list) / num_models_averaged
        std_acc             = statistics.stdev(accuracy_list)
        print(f'The average of {num_models_averaged} runs was an an average of {avg_acc} with a standard deviation of {std_acc} for the percent of pictures correctly classified.')
        
        average_eig_tensor = torch.stack(average_eig_list)
        std_eig_tensor     = torch.stack(std_eig_list)

        for i in range(num_lyap_exp):
            avg_avg_of_eigi     = average_eig_tensor[:,i].mean().item()
            avg_std_dev_of_eigi = std_eig_tensor[    :,i].mean().item()
            print(f'The average of {num_models_averaged} runs was an an average of {avg_avg_of_eigi} with a standard deviation of {avg_std_dev_of_eigi} for mu{i+1}.')

if __name__ == "__main__":
    main()