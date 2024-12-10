import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os, time
import h5py
from dataset.matrix_dataset.create_data import generate_and_save_random_matrices
from second_order.damped_newton import DampedNewton
from run import load_config
import pickle

torch.manual_seed(42)

def factorize_matrix_fo(M_list, latent_dim, optimizer_class=optim.SGD, learning_rate=0.01, num_epochs=1000, lambda_reg=0.01):
    """Perform matrix factorization using first-order optimization methods."""
    d, _ = M_list[0].shape
    n = len(M_list)
    
    # Initialize latent matrices P and Q
    P = torch.randn(d, latent_dim, requires_grad=True)
    Q = torch.randn(latent_dim, d, requires_grad=True)

    # Gradient Descent optimizer
    optimizer = optimizer_class([P, Q], lr=learning_rate)

    # Loss function (MSE + regularization)
    def loss_function(M_list, P, Q):
        total_loss = 0
        for M in M_list:
            predicted = torch.matmul(P, Q)
            mse_loss = torch.norm(M - predicted, p='fro') ** 2
            total_loss += mse_loss
        avg_loss = total_loss / n
        reg_term = lambda_reg * (torch.norm(P) ** 2 + torch.norm(Q) ** 2)
        return avg_loss + reg_term

    

    # Training loop
    losses = []  # Include initial loss as the first entry
    gradient_norms = []   # Gradient norm is zero before training begins
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = loss_function(M_list, P, Q)
        loss.backward()

        # Compute the gradient norm
        grad_norm = torch.norm(P.grad).item() + torch.norm(Q.grad).item()
        gradient_norms.append(grad_norm)

        optimizer.step()
        losses.append(loss.item())

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}, Gradient Norm: {grad_norm:.6f}")

    return P, Q, losses, gradient_norms




def factorize_matrix_mixed(M_list, latent_dim, optimizer_class=optim.SGD, learning_rate=0.01, num_epochs=1000, lambda_reg=0.01,
                           L=1.0, reg=0.0, variant="GradReg"):
    """Perform matrix factorization using mixed first-order and second-order optimization methods."""
    import time

    d, _ = M_list[0].shape
    n = len(M_list)
    
    # Initialize latent matrices P and Q
    P = torch.randn(d, latent_dim, requires_grad=True)
    Q = torch.randn(latent_dim, d, requires_grad=True)

    # Optimizers
    optimizer_P = optimizer_class([P], lr=learning_rate)
    optimizer_Q = DampedNewton([Q], variant=variant, L=L, reg=reg, verbose=False)

    # Loss function (MSE + regularization)
    def loss_function(M_list, P, Q):
        total_loss = 0
        for M in M_list:
            predicted = torch.matmul(P, Q)
            mse_loss = torch.norm(M - predicted, p='fro') ** 2
            total_loss += mse_loss
        avg_loss = total_loss / n
        reg_term = lambda_reg * (torch.norm(P) ** 2 + torch.norm(Q) ** 2)
        return avg_loss + reg_term

    # Training loop
    losses = []
    gradient_norms = []

    # Evaluate initial loss and gradient norms before training
    with torch.no_grad():
        initial_loss = loss_function(M_list, P, Q).item()
        print(f"Initial Loss: {initial_loss:.6f}")
        losses.append(initial_loss)

        initial_grad_norm = (
            torch.norm(P.grad if P.grad is not None else torch.zeros_like(P)).item() +
            torch.norm(Q.grad if Q.grad is not None else torch.zeros_like(Q)).item()
        )
        gradient_norms.append(initial_grad_norm)

    for epoch in range(num_epochs):
        train_start_time = time.time()

        # Closure function for backward passes
        def closure():
            optimizer_P.zero_grad()  # Zero out gradients for P
            optimizer_Q.zero_grad()  # Zero out gradients for Q
            loss = loss_function(M_list, P, Q)
            loss.backward(retain_graph=True)  # Retain graph for second backward pass (for Q)
            return loss

        # Step for Q (second-order optimizer)
        if isinstance(optimizer_Q, DampedNewton):
            optimizer_Q.step(closure)

        # Step for P (first-order optimizer)
        optimizer_P.step(closure)

        # Log the loss
        loss_value = closure().item()
        losses.append(loss_value)

        grad_norm = (
            torch.norm(P.grad if P.grad is not None else torch.zeros_like(P)).item() +
            torch.norm(Q.grad if Q.grad is not None else torch.zeros_like(Q)).item()
        )
        gradient_norms.append(grad_norm)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss_value:.6f}, Gradient Norm: {grad_norm:.6f}")

        epoch_time = time.time() - train_start_time
        print(f"Epoch Time: {epoch_time:.2f}s")

    return P, Q, losses, gradient_norms




def matrix_factorization_experiment(matrices, method="ADAM/ADAM", 
                                    learning_rates=[0.01, 0.1], latent_dims=[5, 10, 15, 20], 
                                    num_epochs=1000, lambda_reg=0.01, 
                                    use_second_order=False, optimizer_class=optim.Adam, 
                                    variant=None,
                                    L_values=[1.0], reg_values=[0.0], save_dir="results"):
    # Create subdirectories for logs and plots
    logs_dir = os.path.join(save_dir, "MaF_logs")
    plots_dir = os.path.join(save_dir, "MaF_plots")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Placeholder for storing results
    all_results = []

    # Perform factorization for each combination of parameters
    for lr in learning_rates:
        for latent_dim in latent_dims:
            if use_second_order:
                for L in L_values:
                    for reg in reg_values:
                        print(f"Running FO+SO: LR {lr}, Latent Dim {latent_dim}, L {L}, reg {reg}")

                        M_tensors = [torch.tensor(M, dtype=torch.float32) for M in matrices]
                        # Call the mixed method (FO+SO)
                        P, Q, losses, gradient_norms = factorize_matrix_mixed(
                            M_tensors, latent_dim=latent_dim,  learning_rate=lr, num_epochs=num_epochs, 
                            lambda_reg=lambda_reg, L=L, reg=reg, variant=variant
                        )


                        # Store results for combined plotting
                        result = {
                            "method": f"{method}, LR={lr}, Latent Dim={latent_dim}, L={L}, reg={reg}",
                            "losses": losses,
                            "gradient_norms": gradient_norms
                        }   
                        all_results.append(result)

                        # Save results to logs folder
                        result_filename = os.path.join(logs_dir, f"results_{method}_LR_{lr}_LatentDim_{latent_dim}_L_{L}_reg_{reg}.pkl")
                        save_results_to_file(result, result_filename)



            else:
                print(f"Running FO: LR {lr}, Latent Dim {latent_dim}")

                
                M_tensors = [torch.tensor(M, dtype=torch.float32) for M in matrices]

                P, Q, losses, gradient_norms = factorize_matrix_fo(
                    M_tensors, latent_dim=latent_dim, optimizer_class=optimizer_class, 
                    learning_rate=lr, num_epochs=num_epochs, lambda_reg=lambda_reg
                )
                    

                

                # Store results for combined plotting
                result = {
                    "method": f"{method}, LR={lr}, Latent Dim={latent_dim}",
                    "losses": losses,
                    "gradient_norms": gradient_norms
                }
                all_results.append(result)

                # Save results to logs folder
                result_filename = os.path.join(logs_dir, f"results_{method}_{lr}_LatentDim_{latent_dim}.pkl")
                save_results_to_file(result, result_filename)



    return all_results



def run(config_path=None, dataset_path=None, optimizer_class=None):
    """Entry point for running the matrix factorization experiment."""
    print("Running Matrix Factorization Experiment")

    # Load dataset
    if dataset_path:
        matrices = load_matrices_from_h5(dataset_path)
    else:
        default_dataset_path = os.path.join(
            os.path.dirname(__file__), "../dataset/matrix_dataset/random_matrices_dataset.h5"
        )
        if not os.path.exists(default_dataset_path):
            matrices = generate_and_save_random_matrices(default_dataset_path)
        else:
            matrices = load_matrices_from_h5(default_dataset_path)

    # Load configuration
    if config_path:
        config = load_config(config_path)
    else:
        raise ValueError("A configuration file must be provided!")

    # Extract experiment parameters
    learning_rates = config.get("learning_rates", [0.01, 0.1])
    method = config.get("method", "ADAM+ADAM")
    latent_dims = config.get("latent_dims", [5, 10, 15, 20])
    num_epochs = config.get("num_epochs", 500)
    lambda_reg = config.get("lambda_reg", 0.01)
    use_second_order = config.get("mixed", False)
    L_values = config.get("L_values", [1.0])
    reg_values = config.get("reg_values", [0.0])
    save_dir = config.get("save_dir", "results")
    variant = config.get("variant", None)

    # Call the matrix factorization experiment
    results = matrix_factorization_experiment(
        matrices=matrices,
        method=method,
        learning_rates=learning_rates,
        latent_dims=latent_dims,
        num_epochs=num_epochs,
        lambda_reg=lambda_reg,
        use_second_order=use_second_order,
        optimizer_class=optimizer_class,
        variant=variant,
        L_values=L_values,
        reg_values=reg_values,
        save_dir=save_dir
    )
    print("Experiment Completed. Results saved.")



def load_matrices_from_h5(filepath):
    """Load matrices from an HDF5 file."""
    all_matrices = []
    try:
        with h5py.File(filepath, 'r') as hf:
            for key in hf.keys():
                matrix = hf[key][:]
                all_matrices.append(matrix)
        print(f"Loaded {len(all_matrices)} matrices from {filepath}.")
    except Exception as e:
        print(f"Error loading dataset from {filepath}: {e}")
        raise
    return all_matrices

def save_results_to_file(results, filename):
    """Save results to a pickle file."""
    with open(filename, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {filename}")
