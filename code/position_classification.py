import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class DeepTanhNet(nn.Module):
    """
    Deep fully-connected neural network using Tanh activations.

    This model consists of an input layer, multiple hidden layers with Tanh
    activations, and a single Tanh output neuron. It also includes a method
    to compute Finite-Time Lyapunov Exponents (FTLEs) to assess sensitivity
    of the network to small perturbations in input space.

    Attributes:
        model (nn.Sequential): Sequential stack of Linear and Tanh layers.

    Args:
        input_size (int, optional): Number of input features. Defaults to 2.
        hidden_size (int, optional): Number of neurons per hidden layer. Defaults to 12.
        hidden_layers (int, optional): Number of hidden layers. Defaults to 10.
    """

    def __init__(self, input_size: int = 2, hidden_size: int = 12, hidden_layers: int = 10):
        super(DeepTanhNet, self).__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.Tanh()]
        for _ in range(hidden_layers):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers += [nn.Linear(hidden_size, 1), nn.Tanh()]
        self.model = nn.Sequential(*layers)
        
        self._initialize_weights(hidden_size)

    def _initialize_weights(self, hidden_size: int) -> None:
        """Initializes weights for all linear layers using a normal distribution.

        Args:
            hidden_size (int): Used to scale the initialization standard deviation.
        """
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=math.sqrt(1 / hidden_size))
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Model output tensor of shape (batch_size, 1).
        """
        return self.model(x)
    
    def max_finite_time_lyapunov_exponents(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Computes the maximum finite-time Lyapunov exponent (FTLE) for each input.

        The FTLE quantifies the local sensitivity of the network output with
        respect to small perturbations in the input space. A higher FTLE value
        indicates stronger local divergence of trajectories in the learned mapping.

        Args:
            x (torch.Tensor): Input tensor of shape (N, input_size) or (input_size,).

        Returns:
            list[torch.Tensor]: List of FTLE values (base-10 logarithm of max singular values)
            corresponding to each input in the batch.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        current_input = x
        jacobian = None

        for i in range(0, len(self.model) - 2, 2):
            linear_layer = self.model[i]
            activation_layer = self.model[i + 1]
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


class PositionClassifcation:
    """
    Trains and visualizes a deep neural network that classifies points as being
    inside or outside a unit circle. Includes tools for model training, classification
    visualization, and Finite-Time Lyapunov Exponent (FTLE) field visualization.

    Attributes:
        model (DeepTanhNet): The neural network used for classification.
        criterion (nn.MSELoss): Loss function for training.
        device (torch.device): Active device (CPU or CUDA) for training.
        Various hyperparameters and plotting configurations.

    Args:
        learning_rate (float, optional): SGD learning rate. Defaults to 0.05.
        number_of_epochs (int, optional): Number of training epochs. Defaults to 10000.
        epoch_print_period (int, optional): Epoch interval for printing progress. Defaults to 1000.
        number_of_samples (int, optional): Number of synthetic samples generated. Defaults to 40000.
        test_size (float, optional): Fraction of samples used for testing. Defaults to 0.1.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
    """

    def __init__(self, learning_rate: float = 0.05, number_of_epochs: int = 10000, epoch_print_period: int = 1000, 
                 number_of_samples: int = 40000, test_size: float = 0.1, seed: int = 42):
        # Data generation
        self.number_of_samples = number_of_samples
        self.domain_bound = 1.25

        # Training configuration
        self.learning_rate      = learning_rate
        self.number_of_epochs   = number_of_epochs
        self.epoch_print_period = epoch_print_period
        
        # Plotting configuration: classification
        self.test_data_subsample_points = 250
        self.plot_length                = 3.2 * 1.2
        self.plot_width                 = 2.4 * 1.2
        self.default_line_color         = 'black'
        self.marker_size                = 40
        self.inside_alpha               = 1.0
        self.marker_shape               = 's'
        self.inside_label               = 'Inside Circle'
        self.plot_line_width            = 1.5
        self.background_color           = 'white'
        self.outside_edgecolors         = 'green'
        self.outside_alpha              = 0.8
        self.outside_label              = 'Outside Circle'
        self.boarder_color              = 'gray'
        self.boarder_linestyle          = '-'
        self.plot_domain_bound          = 1.6
        self.arrowstyle                 = '->'
        self.x_label_position           = (1.4, -0.3)
        self.y_label_position           = (0.1, 1.4)
        self.x_label                    = 'x1'
        self.y_label                    = 'x2'
        self.plot_font_size             = 14
        self.classification_plot_name   = "circleClasfication.png"

        # Plotting configuration: FTLE
        self.domain_resolution          = 200
        self.heatmap_colors             = 'RdBu_r'
        self.heat_map_shading           = 'auto'
        self.heat_domain_bound          = 4
        self.ftle_plot_data_step        = 15
        self.gradient_line_length_scale = 20
        self.gradient_line_widths       = 3
        self.ftle_plot_grad_line_pivot  = 'middle'
        self.ftle_plot_xlabel           = "x_0"
        self.ftle_plot_ylabel           = "x_1"
        self.ftle_plot_title            = "Finite-Time Lyapunov Exponents"
        self.ftle_colorbar_label        = "Max FTLE * L"
        self.ftle_plot_name             = "FTLE.png"

        # Data generation and splitting
        x_all, t_all = self.generate_circle_data()
        x_train, x_test, t_train, t_test = train_test_split(x_all, t_all, test_size=test_size, random_state=seed)

        # Convert to tensors
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.x_train = torch.tensor(x_train, dtype=torch.float32).to(self.device)
        self.t_train = torch.tensor(t_train, dtype=torch.float32).unsqueeze(1).to(self.device)
        self.x_test  = torch.tensor( x_test, dtype=torch.float32).to(self.device)
        self.t_test  = torch.tensor( t_test, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Model setup
        self.model     = DeepTanhNet().to(self.device)
        self.criterion = nn.MSELoss()

    def generate_circle_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates labeled data for circle classification.

        Samples uniformly distributed 2D points in a square region and labels them
        as +1 if they are inside the unit circle and -1 otherwise.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - x: Array of shape (N, 2), sampled 2D coordinates.
                - t: Array of shape (N,), target labels (+1 inside, -1 outside).
        """
        x = np.random.uniform(-self.domain_bound, self.domain_bound, (self.number_of_samples, 2))
        t = 2.0 * ((x[:, 0]**2 + x[:, 1]**2) <= 1.0).astype(np.float32) - 1.0
        return x, t

    def train_model(self) -> None:
        """
        Trains the neural network using stochastic gradient descent (SGD).

        The model minimizes mean squared error (MSE) loss between predicted
        and true circle membership labels. Progress is printed every `epoch_print_period`.
        """
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        for epoch in range(self.number_of_epochs):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(self.x_train)
            loss    = self.criterion(outputs, self.t_train)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % self.epoch_print_period == 0:
                self.model.eval()
                with torch.no_grad():
                    test_outputs = self.model(self.x_test)
                    test_preds   = torch.sign(test_outputs)
                    test_acc     = (test_preds == self.t_test).float().mean()
                print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Test Accuracy: {test_acc.item()*100:.2f}%")
    
    @torch.no_grad()
    def plot_classification(self) -> None:
        """
        Visualizes the model's classification of test points.

        Creates a 2D scatter plot showing points predicted to be inside or outside
        the unit circle, along with the true circle boundary.

        Saves:
            - `self.classification_plot_name` (e.g., "circleClassification.png")
        """
        t_test_np = self.t_test.detach().cpu().numpy().ravel()
        x0        = self.x_test[:, 0].detach().cpu().numpy()
        x1        = self.x_test[:, 1].detach().cpu().numpy()

        subset_idx = np.random.choice(len(t_test_np), size=self.test_data_subsample_points, replace=False)
        x0_sub     = x0[subset_idx]
        x1_sub     = x1[subset_idx]
        t_test_sub = t_test_np[subset_idx]

        inside_mask  = t_test_sub == 1
        outside_mask = t_test_sub == -1

        plt.figure(figsize=(self.plot_length, self.plot_width))

        # Inside points
        plt.scatter(
            x0_sub[inside_mask],
            x1_sub[inside_mask],
            facecolors = self.default_line_color,
            edgecolors = self.default_line_color,
            s          = self.marker_size,
            alpha      = self.inside_alpha,
            marker     = self.marker_shape,
            label      = self.inside_label,
            linewidth  = self.plot_line_width
        )

        # Outside points
        plt.scatter(
            x0_sub[outside_mask],
            x1_sub[outside_mask],
            facecolors = self.background_color,
            edgecolors = self.outside_edgecolors,
            s          = self.marker_size,
            alpha      = self.outside_alpha,
            marker     = self.marker_shape,
            label      = self.outside_label,
            linewidth  = self.plot_line_width
        )

        # Circle boundary
        circle = plt.Circle((0, 0), 1, color=self.boarder_color, fill=False,
                            linestyle=self.boarder_linestyle, linewidth=self.plot_line_width)
        plt.gca().add_patch(circle)

        ax = plt.gca()
        ax.set_facecolor(self.background_color)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_xticks([])
        ax.set_yticks([])

        # Axis arrows
        ax.annotate('', xy=(self.plot_domain_bound, 0), xytext=(-self.plot_domain_bound, 0),
                    arrowprops=dict(arrowstyle=self.arrowstyle, color=self.default_line_color, linewidth=self.plot_line_width))
        ax.annotate('', xy=(0, self.plot_domain_bound), xytext=(0, -self.plot_domain_bound),
                    arrowprops=dict(arrowstyle=self.arrowstyle, color=self.default_line_color, linewidth=self.plot_line_width))

        x_pos1, x_pos2 = self.x_label_position
        y_pos1, y_pos2 = self.y_label_position
        plt.text(x_pos1, x_pos2, self.x_label, fontsize=self.plot_font_size)
        plt.text(y_pos1, y_pos2, self.y_label, fontsize=self.plot_font_size)

        ax.set_xlim(-self.plot_domain_bound, self.plot_domain_bound)
        ax.set_ylim(-self.plot_domain_bound, self.plot_domain_bound)
        ax.set_aspect('equal')

        plt.tight_layout()
        plt.savefig(self.classification_plot_name)
        plt.close()
    
    def plot_finite_time_lyapunov_exponents(self) -> None:
        """
        Computes and visualizes the FTLE field of the trained model.

        Generates a heatmap of maximum FTLE values across the 2D input domain,
        showing regions of sensitivity in the learned mapping. Also overlays
        local gradient direction vectors.

        Saves:
            - `self.ftle_plot_name` (e.g., "FTLE.png")
        """
        plt.figure(figsize=(self.plot_length, self.plot_width))

        domain_linespace = np.linspace(-self.domain_bound, self.domain_bound, self.domain_resolution)
        x0, x1     = np.meshgrid(domain_linespace, domain_linespace)
        grid       = np.c_[x0.ravel(), x1.ravel()]
        torch_grid = torch.tensor(grid, dtype=torch.float32).to(self.device)

        torch_exp = self.model.max_finite_time_lyapunov_exponents(torch_grid)
        exp       = torch_exp.detach().cpu().numpy().reshape(x1.shape)

        pcm = plt.pcolormesh(x0, x1, exp, cmap=self.heatmap_colors,
                             shading=self.heat_map_shading,
                             vmin=-self.heat_domain_bound, vmax=self.heat_domain_bound)

        V, U   = np.gradient(exp)
        norm   = np.sqrt(U**2 + V**2)
        U_unit = np.divide(U, norm, out=np.zeros_like(U), where=(norm != 0))
        V_unit = np.divide(V, norm, out=np.zeros_like(V), where=(norm != 0))

        x0_sub =     x0[::self.ftle_plot_data_step, ::self.ftle_plot_data_step]
        x1_sub =     x1[::self.ftle_plot_data_step, ::self.ftle_plot_data_step]
        U_sub  = U_unit[::self.ftle_plot_data_step, ::self.ftle_plot_data_step]
        V_sub  = V_unit[::self.ftle_plot_data_step, ::self.ftle_plot_data_step]

        plt.quiver(x0_sub, x1_sub, U_sub, V_sub,
                   color      = self.default_line_color,
                   pivot      = self.ftle_plot_grad_line_pivot,
                   headwidth  = 0, headlength=0, headaxislength=0,
                   scale      = self.gradient_line_length_scale,
                   linewidths = self.gradient_line_widths)
        
        plt.axis('equal')
        plt.xlabel(self.ftle_plot_xlabel)
        plt.ylabel(self.ftle_plot_ylabel)
        plt.title(self.ftle_plot_title)
        plt.colorbar(pcm, label=self.ftle_colorbar_label)
        plt.savefig(self.ftle_plot_name)
        plt.close()


def main():
    pos_class_object = PositionClassifcation(learning_rate = 0.05)
    pos_class_object.train_model()
    pos_class_object.plot_classification()
    pos_class_object.plot_finite_time_lyapunov_exponents()

if __name__=="__main__":
    main()