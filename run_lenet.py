import argparse
import os
import pickle
import yaml
import torch
import torch.nn as nn
from torch.optim import Adam, LBFGS, SGD
from dataset.cifar10 import cifar10_dataloaders
from tests.lenet import Runner
from models.lenet import LeNet5
from second_order.damped_newton import DampedNewton

# Set default environment variables
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TORCH_DEVICE"] = "cuda"

EXPERIMENTS_DIR = "results/lenet5"

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def main(config):
    """Main function to train and save results using config."""
    # Device setup
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config["random_seed"])

    # Data setup
    if config["data"] == "cifar10":
        model = LeNet5(10, 3)
        runner = Runner(
            model=model,
            loss_fn=nn.CrossEntropyLoss(),
            metrics_fn=lambda outputs, labels: (outputs.argmax(dim=1) == labels).sum().item(),
            random_seed=config["random_seed"],
            device=device,
            verbose=config["verbose"],
        )
        train_loader, test_loader = cifar10_dataloaders(batch_size=config["batch_size"])

    # Optimizer setup
    if config["optimizer2"] is None:
        if config["optimizer1"] == "adam":
            optimizer1 = Adam(model.parameters(), lr=config["learning_rate1"])
        elif config["optimizer1"] == "sgd":
            optimizer1 = SGD(model.parameters(), lr=config["learning_rate1"], momentum=config["momentum"])
        optimizer2 = None
    else:
        fo_parameters = [param for name, param in model.named_parameters() if "last" not in name]
        so_parameters = [param for name, param in model.named_parameters() if "last" in name]

        if config["optimizer1"] == "adam":
            optimizer1 = Adam(fo_parameters, lr=config["learning_rate1"])
        elif config["optimizer1"] == "sgd":
            optimizer1 = SGD(fo_parameters, lr=config["learning_rate1"], momentum=config["momentum"])

        if config["optimizer2"] == "lbfgs":
            optimizer2 = LBFGS(so_parameters, lr=config["learning_rate2"])
        elif config["optimizer2"] == "newton":
            optimizer2 = DampedNewton(
                so_parameters, alpha=config["learning_rate2"], variant=None, reg=config["lambd"], CG_subsolver=False
            )
        elif config["optimizer2"] == "gradreg":
            optimizer2 = DampedNewton(
                so_parameters, alpha=config["learning_rate2"], variant="GradReg", L=config["lipschitz"], CG_subsolver=False
            )

    # Train the model
    results = runner.train(
        num_epochs=config["num_epochs"],
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer1=optimizer1,
        optimizer2=optimizer2,
    )

    # Save results
    if not os.path.exists(EXPERIMENTS_DIR):
        os.makedirs(EXPERIMENTS_DIR)

    task_folder = os.path.join(EXPERIMENTS_DIR, config["data"])
    if not os.path.exists(task_folder):
        os.makedirs(task_folder)

    if config["optimizer2"] is None:
        optimizer_folder = os.path.join(task_folder, config["optimizer1"])
        filename = f"lr={config['learning_rate1']}_bs={config['batch_size']}_e={config['num_epochs']}_rs={config['random_seed']}.pickle"
    else:
        optimizer_folder = os.path.join(task_folder, config["optimizer1"] + "_" + config["optimizer2"])
        filename = f"lr1={config['learning_rate1']}_lr2={config['learning_rate2']}_bs={config['batch_size']}_e={config['num_epochs']}_rs={config['random_seed']}.pickle"

    if not os.path.exists(optimizer_folder):
        os.makedirs(optimizer_folder)

    with open(os.path.join(optimizer_folder, filename), "wb") as f:
        pickle.dump(results, f)
        print("Results saved to", os.path.join(optimizer_folder, filename))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LeNet5 experiments.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration file."
    )
    args = parser.parse_args()

    config_path = args.config
    config = load_config(config_path)
    main(config)
