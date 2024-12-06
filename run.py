import argparse
import importlib
import yaml
import os
import torch.optim as optim
import sys



sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def load_config(config_path):
    """Load configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_optimizer_class(optimizer_name):
    """Get the optimizer class from its name."""
    optimizers = {
        "SGD": optim.SGD,
        "Adam": optim.Adam,
    }
    if optimizer_name not in optimizers:
        raise ValueError(f"Unsupported optimizer '{optimizer_name}'. Supported optimizers: {list(optimizers.keys())}")
    return optimizers[optimizer_name]

def main():
    print("Running the main function.")
    parser = argparse.ArgumentParser(description="Run matrix factorization experiments.")
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=["matrix_factorization", "attention_layer","lenet5"],
        help="Specify the experiment to run.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file."
    )
    args = parser.parse_args()

    # Load the configuration
    config = load_config(args.config)

    # Select the optimizer
    optimizer_name = config.get("optimizer", "Adam")
    #optimizer_params = config.get("optimizer_params", {})
    optimizer_class = get_optimizer_class(optimizer_name)

    # Extract the variant for FO + SO
    variant = config.get("variant", None)

    # Dynamically import the experiment module
    if args.experiment == "matrix_factorization":
        module = importlib.import_module("tests.matrix_factorization")
    elif args.experiment == "attention_layer":
        module = importlib.import_module("tests.attention_layer")
    elif args.experiment == "lenet5":
        module = importlib.import_module("tests.lenet")
    else:
        print(f"Error: Unsupported experiment '{args.experiment}'.")
        return
    
    if hasattr(module, "run"):
        if args.experiment == "attention_layer":
            # For attention_layer, no dataset_path is needed
            module.run(
                config_path=args.config,
                optimizer_class=optimizer_class,
                
            )
        elif args.experiment == "matrix_factorization":
            # For other experiments that require dataset_path
            module.run(
                config_path=args.config,
                dataset_path=config.get("dataset_path"),
                optimizer_class=optimizer_class,
        )
        elif args.experiment == "lenet5":
            # For other experiments that require dataset_path
            module.run(
                config_path=args.config,
                optimizer_class=optimizer_class,
        )
    else:
        print(f"Error: The 'run' function is not defined in the module for {args.experiment}.")

if __name__ == "__main__":
    main()