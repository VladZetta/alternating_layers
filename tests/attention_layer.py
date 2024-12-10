import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import yaml
import os
import matplotlib.pyplot as plt
from second_order.damped_newton import DampedNewton

torch.manual_seed(42)

# Generate synthetic dataset
def generate_large_synthetic_data(total_samples, seq_len, embed_dim):
    input_data = torch.randn(total_samples, seq_len, embed_dim)
    target_output = torch.randn(total_samples, seq_len, embed_dim)
    return input_data, target_output

# Prepare DataLoader
def prepare_dataloader(input_data, target_output, batch_size):
    dataset = TensorDataset(input_data, target_output)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the standalone attention layer
class StandaloneAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=1, batch_first=True)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        return attn_output
    
def train_attention_layer(attention_layer, dataloader, criterion, optimizer_P, num_epochs, save_dir, 
                          mixed=False, optimizer_Q=None, variant=None, L=None, reg=None, method=None, lr=None):
    """Train the attention layer and save results in organized subfolders."""
    # Create subdirectories for checkpoints and plots
    checkpoints_dir = os.path.join(save_dir, "checkpoints")
    plots_dir = os.path.join(save_dir, "plots")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    attention_layer.train()

    epoch_losses = []
    epoch_gradient_norms = []

    # Record initial loss
    with torch.no_grad():
        initial_loss = 0.0
        for input_data, target_output in dataloader:
            outputs = attention_layer(input_data)
            loss = criterion(outputs, target_output)
            initial_loss += loss.item()
        initial_loss /= len(dataloader)
    print(f"Initial Loss (before training): {initial_loss:.4f}")
    
    # Add the initial loss to epoch losses
    epoch_losses.append(initial_loss)
    # Add a placeholder (e.g., 0) for gradient norm at initialization
    epoch_gradient_norms.append(0.0)

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_grad_norm = 0.0

        for input_data, target_output in dataloader:
            if mixed:
                # Mixed Optimization: FO + SO
                def closure():
                    optimizer_P.zero_grad()
                    optimizer_Q.zero_grad()
                    outputs = attention_layer(input_data)
                    loss = criterion(outputs, target_output)
                    loss.backward(retain_graph=True)  # Retain graph for SO optimizer
                    return loss

                if isinstance(optimizer_Q, DampedNewton):
                    optimizer_Q.step(closure)
                optimizer_P.step(closure)

                loss_value = closure().item()
                running_loss += loss_value
                grad_norm = sum(p.grad.norm().item() for p in attention_layer.parameters() if p.grad is not None)
                running_grad_norm += grad_norm
            else:
                # Standard First-Order Optimization
                optimizer_P.zero_grad()
                outputs = attention_layer(input_data)
                loss = criterion(outputs, target_output)
                loss.backward()
                optimizer_P.step()

                running_loss += loss.item()
                grad_norm = sum(p.grad.norm().item() for p in attention_layer.parameters() if p.grad is not None)
                running_grad_norm += grad_norm

        # Log average loss and gradient norm
        avg_loss = running_loss / len(dataloader)
        avg_grad_norm = running_grad_norm / len(dataloader)
        epoch_losses.append(avg_loss)
        epoch_gradient_norms.append(avg_grad_norm)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Gradient Norm: {avg_grad_norm:.4f}")

    

    

    # Generate a unique filename based on hyperparameters and settings
    if not mixed:
        filename_suffix = f"method_{method}_lr_{lr}"
        results_file = os.path.join(checkpoints_dir, f"attention_layer_results_{filename_suffix}.pth")
    else:
        filename_suffix = f"method_{method}_lr_{lr}_variant_{variant}_L_{L}_reg_{reg}"
        results_file = os.path.join(checkpoints_dir, f"attention_layer_results_{filename_suffix}.pth")
                                    

    # Save all results in a single file
    results = {
        "model_state": attention_layer.state_dict(),
        "losses": epoch_losses,
        "gradient_norms": epoch_gradient_norms,
        "settings": {
            "embed_dim": attention_layer.attention.embed_dim,
            "batch_size": dataloader.batch_size,
            "L": L,
            "reg": reg,
            "mixed": mixed,
            "method": method,
            "learning_rate": lr,
            "num_epochs": num_epochs,
        },
    }
    torch.save(results, results_file)
    print(f"All results saved to {results_file}")

    # Save plots
    plot_metrics(epoch_losses, epoch_gradient_norms, plots_dir, filename_suffix)


def run(config_path, optimizer_class):
    # Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Extract parameters
    embed_dim = config["embed_dim"]
    seq_len = config["seq_len"]
    batch_size = config["batch_size"]
    total_samples = config["total_samples"]
    num_epochs = config["num_epochs"]
    learning_rate = config.get("learning_rate", 0.001)
    save_dir = config["save_dir"]
    mixed = config.get("mixed", False)
    variant = config.get("variant", None)
    method = config.get("method", None)
    L = config.get("L", 1.0)
    reg = config.get("reg", 0.0)

    # Generate synthetic data
    torch.manual_seed(42)
    input_data, target_output = generate_large_synthetic_data(total_samples, seq_len, embed_dim)

    # Prepare DataLoader
    dataloader = prepare_dataloader(input_data, target_output, batch_size)

    # Initialize attention layer
    attention_layer = StandaloneAttention(embed_dim=embed_dim)

    # Split parameters into FO and SO groups
    fo_parameters = [
        param for name, param in attention_layer.named_parameters() if "out_proj" in name
    ]
    so_parameters = [
        param for name, param in attention_layer.named_parameters() if "in_proj_weight" in name
    ]

    # Define criterion and first-order optimizer (FO)
    criterion = nn.MSELoss()
    optimizer_P = optimizer_class(fo_parameters, lr=learning_rate)

    if mixed:
        print(variant, type(variant))
        # Second-order optimizer for mixed optimization
        optimizer_Q = DampedNewton(
            so_parameters, 
            variant=variant, 
            L=L, 
            reg=reg, 
            verbose=True
        )
    else:
        optimizer_Q = None

    # Train the attention layer
    train_attention_layer(
        attention_layer, 
        dataloader, 
        criterion, 
        optimizer_P, 
        num_epochs, 
        save_dir, 
        mixed=mixed, 
        optimizer_Q=optimizer_Q, 
        variant=variant, 
        L=L, 
        reg=reg,
        method=method,
        lr=learning_rate,
    )



def plot_metrics(losses, grad_norms, plots_dir, filename_suffix):
    """Plot and save loss and gradient norm."""
    epochs = range(1, len(losses) + 1)

    # Plot loss
    plt.figure()
    plt.plot(epochs, losses, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)
    loss_plot_file = os.path.join(plots_dir, f"loss_curve_{filename_suffix}.png")
    plt.savefig(loss_plot_file)
    plt.close()

    # Plot gradient norm
    plt.figure()
    plt.plot(epochs, grad_norms, label="Gradient Norm", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Gradient Norm")
    plt.title("Gradient Norm Curve")
    plt.legend()
    plt.grid(True)
    grad_norm_plot_file = os.path.join(plots_dir, f"gradient_norm_curve_{filename_suffix}.png")
    plt.savefig(grad_norm_plot_file)
    plt.close()

    print(f"Plots saved: {loss_plot_file}, {grad_norm_plot_file}")
