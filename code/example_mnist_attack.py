import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import numpy as np
import statistics
import math

class TanhSoftmaxNet(nn.Module):
    def __init__(self, input_size=784, hidden_layer_size=20, numb_hidden_layers=16, number_of_outputs=10):
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
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(1 / self.hidden_layer_size))
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
        
        number_of_hidden_layers_tensor = torch.tensor(self.numb_hidden_layers, dtype=torch.float64, device=max_lyapunov_exponents.device)
        output = max_lyapunov_exponents / number_of_hidden_layers_tensor
        output = torch.where(torch.isfinite(output), output, torch.zeros_like(output))

        return output

class MNISTClassification:
    def __init__(self, learning_rate: float = 0.001, number_of_epochs: int = 25, batch_size: int = 64, display_training_updates = True) -> None:
        self.batch_size               = batch_size
        self.learning_rate            = learning_rate
        self.number_of_epochs         = number_of_epochs
        self.display_training_updates = display_training_updates
        
        # transform         = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) 
        transform         = transforms.Compose([transforms.ToTensor()]) 
        train_dataset     = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        test_dataset      = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader  = torch.utils.data.DataLoader(dataset=test_dataset , batch_size=1000           , shuffle=False) 

        self.loss_function = nn.CrossEntropyLoss()
        self.device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.original_model   = None
        self.bottleneck_model = None

    def test_model(self, model):
        model.eval()
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            for images, labels in self.test_loader:
                images  = images.reshape(-1, 28 * 28).to(self.device)
                labels  = labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

            acc = 100.0 * n_correct / n_samples
            return acc

    def train_model(self, model, title="Training Phase"):
        print(f"--- Starting: {title} ---")
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),lr=5e-3, momentum=0.9)
        
        train_losses    = []
        test_accuracies = []
        n_total_steps   = len(self.train_loader)

        for epoch in range(self.number_of_epochs):
            model.train()
            running_loss = 0.0
            for images, labels in self.train_loader:
                images = images.reshape(-1, 28 * 28).to(self.device)
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

    def per_model_ftle(self, model):
        model.eval()
        lyaps_list = []
        samples_processed = 0
        with torch.no_grad():
            for images, _ in self.test_loader:
                images = images.reshape(-1, 28 * 28).to(self.device)
                lyaps_batch = model.max_finite_time_lyapunov_exponents(images)
                lyaps_list.append(lyaps_batch)

                samples_processed += images.size(0)
                print(f"Total samples processed: {samples_processed}")

        all_lyaps_gpu = torch.cat(lyaps_list, dim=0).cpu().numpy()
        average_lyap  = sum(all_lyaps_gpu) / len(all_lyaps_gpu)
        stddev_lyap   = statistics.stdev(all_lyaps_gpu.tolist())


        return average_lyap, stddev_lyap

    def visualize_ftle_on_data_points(self):
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
                    images = images.reshape(-1, 28 * 28).to(self.device)
                    lyaps_batch      = self.original_model.max_finite_time_lyapunov_exponents(images)
                    features_batch   = self.bottleneck_model.feature_extractor(images)
                    bottleneck_batch = self.bottleneck_model.bottleneck[0](features_batch)
                    
                    lyaps_list.append(lyaps_batch)
                    points_list.append(bottleneck_batch)

                    samples_processed += images.size(0)
                    print(f"Total samples processed: {samples_processed}")

            all_max_lyaps_gpu = torch.cat(lyaps_list, dim=0).cpu().numpy()
            all_points_gpu    = torch.cat(points_list, dim=0).cpu().numpy()

            plt.figure(figsize=(12, 10))
    
            scatter = plt.scatter(all_points_gpu[:, 0], all_points_gpu[:, 1], c=all_max_lyaps_gpu, cmap='coolwarm', alpha=0.8, s=15)

            cbar = plt.colorbar(scatter, label="Max FTLE ($\log_{10}$ scale)")
            cbar.ax.yaxis.label.set_size(14)     # label font
            cbar.ax.tick_params(labelsize=12)    # ticks
            plt.title('Bottleneck Activations Colored by FTLE Value'  , fontsize=16)
            plt.xlabel('Neuron 1 Activation', fontsize=14)
            plt.ylabel('Neuron 2 Activation', fontsize=14)
            plt.rc('axes', labelsize=14)
            plt.rc('xtick', labelsize=14)
            plt.rc('ytick', labelsize=14)
            plt.tick_params(axis='x', labelsize=14)
            plt.tick_params(axis='y', labelsize=14)
            plt.rc('font', size=14)
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.axis('equal')
            plt.savefig("mnist_2d_projection.png", dpi=600)
            plt.close()
        
    def plot_error_and_entropy_vs_lambda1(self, ensemble_models, bin_edges=50):
        """
        Replicates the plot of classification error and predictive uncertainty (H)
        as functions of lambda_1 (max Lyapunov exponent).
        """
        print("\n--- Generating Error and Entropy vs. Lambda_1 Plot ---")
        if not ensemble_models:
            print("No models provided for ensemble. Cannot generate plot.")
            return

        all_lambda1s  = []
        all_errors    = []
        all_entropies = []
        softmax = nn.Softmax(dim=1)
        epsilon = 1e-9 # for log stability

        samples_processed = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.reshape(-1, 28 * 28).to(self.device)
                labels = labels.to(self.device)

                batch_lambda1s = []
                batch_ensemble_probs = [] # To store probabilities for each image across ensemble

                for model in ensemble_models:
                    model.eval()
                    # Calculate lambda_1 for each image
                    lyap_exp = model.max_finite_time_lyapunov_exponents(images)
                    batch_lambda1s.append(lyap_exp.cpu().numpy())

                    # Get softmax probabilities for each image from this model
                    outputs = model(images)
                    probs = softmax(outputs)
                    batch_ensemble_probs.append(probs)

                    samples_processed += images.size(0)
                    print(f"Total samples processed: {samples_processed}")
                
                # Average lambda_1 across the ensemble for each image (if multiple lambda_1s per image were computed)
                # For this problem, max_finite_time_lyapunov_exponents returns one value per image.
                # So we just take the first model's lambda_1 for now, or average if desired.
                # The paper implies one lambda_1 per input x. So we use the first model's.
                avg_batch_lambda1 = np.mean(np.array(batch_lambda1s), axis=0) # average across models for stability, or just use one
                all_lambda1s.extend(avg_batch_lambda1)

                # For each image, average probabilities across the ensemble
                # stack -> [num_models, batch_size, num_classes]
                ensemble_probs_for_batch = torch.stack(batch_ensemble_probs)
                # average -> [batch_size, num_classes]
                avg_probs_per_image = torch.mean(ensemble_probs_for_batch, dim=0) 
                
                # Calculate entropy H for each image based on its averaged probabilities
                entropies_per_image = -torch.sum(avg_probs_per_image * torch.log(avg_probs_per_image + epsilon), dim=1)
                all_entropies.extend(entropies_per_image.cpu().numpy())

                # Determine error for each image using the prediction from the first model (or the ensemble majority vote)
                # For error, we usually check against a single model's prediction or a majority vote.
                # Let's use the first model's prediction for simplicity for error calculation per image.
                first_model_outputs = ensemble_models[0](images)
                _, predicted = torch.max(first_model_outputs.data, 1)
                is_correct = (predicted == labels)
                all_errors.extend((~is_correct).cpu().numpy()) # True for error, False for correct

        # Convert to numpy arrays for easier manipulation
        all_lambda1s = np.array(all_lambda1s)
        all_errors = np.array(all_errors).astype(float) # convert bool to float for averaging
        all_entropies = np.array(all_entropies)

        # Bin the data by lambda_1 values
        min_lambda1 = np.min(all_lambda1s)
        max_lambda1 = np.max(all_lambda1s)
        lambda_bins = np.linspace(min_lambda1, max_lambda1, bin_edges)

        binned_lambda1   = []
        binned_errors    = []
        binned_entropies = []

        # Iterate through bins and calculate average error and entropy
        for i in range(len(lambda_bins) - 1):
            lower_bound = lambda_bins[i]
            upper_bound = lambda_bins[i+1]
            
            # Find indices of samples falling into the current bin
            if i == len(lambda_bins) - 2: # Include the max value in the last bin
                bin_indices = np.where((all_lambda1s >= lower_bound) & (all_lambda1s <= upper_bound))
            else:
                bin_indices = np.where((all_lambda1s >= lower_bound) & (all_lambda1s < upper_bound))
            
            if len(bin_indices[0]) > 0:
                avg_lambda1_in_bin = np.mean(all_lambda1s[bin_indices])
                avg_error_in_bin   = np.mean(all_errors[bin_indices]) * 100 # Convert to percentage
                avg_entropy_in_bin = np.mean(all_entropies[bin_indices])

                binned_lambda1.append(avg_lambda1_in_bin)
                binned_errors.append(avg_error_in_bin)
                binned_entropies.append(avg_entropy_in_bin)

        # Plotting
        fig, ax1 = plt.subplots(figsize=(10, 4))

        color_error = 'black'
        ax1.set_xlabel(r'$\lambda_1^{(L)}(\mathbf{x})$')
        ax1.set_ylabel('Test error (%)', color=color_error)
        ax1.plot(binned_lambda1, binned_errors, color=color_error, label='Test error (%)')
        ax1.tick_params(axis='y', labelcolor=color_error)
        ax1.set_ylim(bottom=0) # Error should not go below 0

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color_entropy = 'green'
        ax2.set_ylabel(r'$H$', color=color_entropy)  # we already handled the x-label with ax1
        ax2.plot(binned_lambda1, binned_entropies, color=color_entropy, label='H')
        ax2.tick_params(axis='y', labelcolor=color_entropy)
        ax2.set_ylim(bottom=0) # Entropy should not go below 0

        # Add grid and title for better readability
        ax1.grid(True, linestyle='--', alpha=0.6)
        plt.title('Classification Error and Predictive Uncertainty vs. $\lambda_1^{(L)}(\mathbf{x})$')

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.savefig("error_entropy_vs_lambda1.png", dpi=300)
        plt.close()
        print("Plot saved as error_entropy_vs_lambda1.png")

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
            images     = images.reshape(-1, 28 * 28).to(self.device)
            labels     = labels.to(self.device)
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
                    image_ref     = images.reshape(-1, 28, 28).to(self.device)
                    perturbed_ref = perturbed_data.reshape(-1, 28, 28).to(self.device)
                    adv_ex        = perturbed_ref[index].squeeze().detach().cpu().numpy()
                    orig_ex       = image_ref[index].squeeze().detach().cpu().numpy()
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
    untrained_model = TanhSoftmaxNet().to(classifier.device)
    trained_model   = classifier.train_model(untrained_model, title="Phase 1: Model Training")
    classifier.analyze_attacks(trained_model, epsilons)

if __name__ == "__main__":
    main()