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
    
    def max_n_finite_time_lyapunov_exponents(self, x: torch.Tensor, num_lyap_exp: int = 1) -> list[torch.Tensor]:
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


class MNISTClassification:
    def __init__(self, learning_rate: float = 0.2, number_of_epochs: int = 1, batch_size: int = 64, display_training_updates = True) -> None:
        self.batch_size               = batch_size
        self.learning_rate            = learning_rate
        self.number_of_epochs         = number_of_epochs
        self.display_training_updates = display_training_updates
        
        transform         = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) 
        train_dataset     = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        test_dataset      = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)
        
        subset_indices = range(200)
        test_subset = Subset(test_dataset, subset_indices)
        
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader  = torch.utils.data.DataLoader(dataset=test_subset, batch_size=1000, shuffle=False, num_workers=4, pin_memory=True) # Speed up CPU to GPU transfer

        self.criterion = nn.CrossEntropyLoss()
        self.device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
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
                labels = labels.to(self.device)
                
                outputs = model(images)
                loss = self.criterion(outputs, labels)

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

    def per_model_stats(self, model, num_lyap_exp = 1):
        model.eval()
        n_correct  = 0
        n_samples  = 0
        lyaps_list = []
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images      = images.reshape(-1, 28 * 28).to(self.device)
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
                    lyaps_batch      = self.original_model.max_n_finite_time_lyapunov_exponents(images)
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
        
    def plot_error_and_entropy_vs_lambda(self, ensemble_models, num_lyap_exp = 2, bin_edges=50):
        print("\n--- Generating Error and Entropy vs. Lambda_1 Plot ---")
        if not ensemble_models:
            print("No models provided for ensemble. Cannot generate plot.")
            return

        all_lambdas   = []
        all_errors    = []
        all_entropies = []
        softmax       = nn.Softmax(dim=1)
        epsilon       = 1e-9 # for log stability

        samples_processed = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.reshape(-1, 28 * 28).to(self.device)
                labels = labels.to(self.device)

                batch_lambdas        = []
                batch_ensemble_probs = []

                for model in ensemble_models:
                    model.eval()
                    # lyap_exp = model.max_finite_time_lyapunov_exponents(images)
                    lyap_exp = model.max_n_finite_time_lyapunov_exponents(images)
                    batch_lambdas.append(lyap_exp[:, 0].cpu().numpy())
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
                entropies_per_image      = -torch.sum(avg_probs_per_image * torch.log(avg_probs_per_image + epsilon), dim=1)
                all_entropies.extend(entropies_per_image.cpu().numpy())

                # compute model ensemble errors rates
                _, predicted = torch.max(avg_probs_per_image, 1)
                is_error = (predicted != labels)
                all_errors.extend((is_error).cpu().numpy())

        all_lambdas   = np.array(all_lambdas)
        all_errors    = np.array(all_errors).astype(float) # convert bool to float for averaging
        all_entropies = np.array(all_entropies)



        min_lambda  = np.min(all_lambdas)
        max_lambda  = np.max(all_lambdas)
        lambda_bins = np.linspace(min_lambda, max_lambda, bin_edges)

        binned_lambda    = []
        binned_errors    = []
        binned_entropies = []

        # Iterate through bins and calculate average error and entropy
        for i in range(len(lambda_bins) - 1):
            lower_bound = lambda_bins[i]
            upper_bound = lambda_bins[i+1]
            
            if i == len(lambda_bins) - 2: # Include the max value in the last bin
                bin_indices = np.where((all_lambdas >= lower_bound) & (all_lambdas <= upper_bound))
            else:
                bin_indices = np.where((all_lambdas >= lower_bound) & (all_lambdas < upper_bound))
            
            if len(bin_indices[0]) > 0:
                avg_lambda_in_bin  = np.mean(  all_lambdas[bin_indices])
                avg_error_in_bin   = np.mean(   all_errors[bin_indices]) * 100 # Convert to percentage
                avg_entropy_in_bin = np.mean(all_entropies[bin_indices])

                binned_lambda.append(avg_lambda_in_bin)
                binned_errors.append(avg_error_in_bin)
                binned_entropies.append(avg_entropy_in_bin)

        # Plotting
        fig, axes = plt.subplots(num_lyap_exp, 1, figsize=(10, 4))
        fig.suptitle('Classification Error and Predictive Uncertainty vs. $\lambda_i^{(L)}(\mathbf{x})$')
        for i, ax in enumerate(axes):
            color_error = 'black'
            ax.set_xlabel(r'$\lambda_1^{(L)}(\mathbf{x})$')
            ax.set_ylabel('Test error (%)', color=color_error)
            ax.plot(binned_lambda, binned_errors, color=color_error, label='Test error (%)')
            ax.tick_params(axis='y', labelcolor=color_error)
            ax.set_ylim(bottom=0) # Error should not go below 0

            ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
            color_entropy = 'green'
            ax2.set_ylabel(r'$H$', color=color_entropy)  # we already handled the x-label with ax1
            ax2.plot(binned_lambda, binned_entropies, color=color_entropy, label='H')
            ax2.tick_params(axis='y', labelcolor=color_entropy)
            ax2.set_ylim(bottom=0) # Entropy should not go below 0

            # Add grid and title for better readability
            ax.grid(True, linestyle='--', alpha=0.6)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.savefig("error_entropy_vs_lambda.png", dpi=300)
        plt.close()
        print("Plot saved as error_entropy_vs_lambda.png")

    def fgsm_attack(self, images, attack_size, image_grads):
        sign_image_grads = image_grads.sign()
        perturbed_images = images + attack_size * sign_image_grads
        perturbed_images = torch.clamp(perturbed_images, 0, 1) # valid range [0, 1]
        return perturbed_images

    def test_attack(self, model, attack_size):
        n_correct    = 0
        adv_examples = []
        model.eval() 

        for images, labels in self.test_loader:
            # compute gradients for Fast Gradient Sign Method (FGSM) attacks on input images
            images     = images.reshape(-1, 28 * 28).to(self.device)
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
                
                if (was_match and is_mismatch and (len(adv_examples) < 5)):
                    image_ref     = images.reshape(-1, 28, 28).to(self.device)
                    perturbed_ref = perturbed_data.reshape(-1, 28, 28).to(self.device)
                    adv_ex        = perturbed_ref[index].squeeze().detach().cpu().numpy()
                    orig_ex       = image_ref[index].squeeze().detach().cpu().numpy()
                    adv_examples.append((init_pred_item, final_pred_item, orig_ex, adv_ex))

        final_acc = n_correct / float(len(self.test_loader.dataset.data))
        print(f"Epsilon: {attack_size}\tTest Accuracy = {n_correct} / {len(self.test_loader.dataset.data)} = {final_acc:.4f}")

        return final_acc, adv_examples
    
    def analyze_attacks(self, model, attack_sizes):
        accuracies   = []
        all_examples = []

        for attack_size in attack_sizes:
            acc, ex = self.test_attack(model, attack_size)
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
            print("\nNot enough adversarial examples found to display for that attack size.")
            print("Try training the model for more epochs or increasing attack size.")


class DesiredPlot(Enum):
    FTLE_2D    = 1
    ENTROPY    = 2
    AVERAGE    = 3
    ATTACK     = 4
    STAT_TABLE = 5

def main():
    classifier              = MNISTClassification()
    
    # change as desired
    num_models_averaged     = 2
    hidden_layer_sizes_list = range(10, 120, 5)
    attack_sizes            = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    num_lyap_exp            = 3
    desired_plot            =  DesiredPlot.ENTROPY
    

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
        
        classifier.plot_error_and_entropy_vs_lambda(ensemble_models)

    if desired_plot == DesiredPlot.AVERAGE:
        average_eig1_list                   = []
        standard_dev_eig1_list              = []
        # classifier.display_training_updates = False

        for hidden_layer_size in hidden_layer_sizes_list:
            print(f"\n--- Training and Testing with {hidden_layer_size} Nodes Per Layer Hidden---")
            average_eig1_size_i_list = []
            std_eig1_size_i_list = []

            for num in range(num_models_averaged):
                untrained_model   = TanhSoftmaxNet(hidden_layer_size = hidden_layer_size).to(classifier.device)
                trained_model     = classifier.train_model(untrained_model, title=f"Model {num+1}/{num_models_averaged} Training")
                average_eig1, standard_dev_eig1, _ = classifier.per_model_stats(trained_model)
                
                average_eig1_size_i_list.append(average_eig1)
                std_eig1_size_i_list.append(standard_dev_eig1)

            average_eig1         = sum(average_eig1_size_i_list) / num_models_averaged
            standard_dev_of_eig1 = sum(std_eig1_size_i_list) / num_models_averaged
            
            average_eig1_list.append(average_eig1)
            standard_dev_eig1_list.append(standard_dev_of_eig1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 2))
        ax1.semilogx(hidden_layer_sizes_list,         average_eig1_list, label='Average Accuracy')
        ax2.semilogx(hidden_layer_sizes_list, standard_dev_eig1_list, label='Std Dev of Accuracy')

        ax1.set_xlabel('N')
        ax2.set_xlabel('N')
        ax1.set_ylabel(r'$\langle \lambda_1^{(L)}(\bf{x}$' + r'$) \rangle$')
        ax2.set_ylabel(r'Std[$\lambda_1^{(L)}(\bf{x}$)]')
        plt.tight_layout()
        plt.savefig("accuracy_vs_num.png", dpi=600)

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
            average_eig, standard_dev_eig, accuracy = classifier.per_model_stats(trained_model, num_lyap_exp)
            
            average_eig_list.append(average_eig)
            std_eig_list.append(standard_dev_eig)
            accuracy_list.append(accuracy)

        avg_acc             = sum(accuracy_list)            / num_models_averaged
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