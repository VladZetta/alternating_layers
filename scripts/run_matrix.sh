#!/bin/bash
cd "$(dirname "$0")/.."
# Define the experiment type and directory containing configuration files
EXPERIMENT="matrix_factorization"
CONFIG_DIR="configs/matrix_factorization"

# List of YAML configuration files
CONFIG_FILES=(
    "adam+adam.yaml"
    "adam+gradreg.yaml"
    "adam+newton.yaml"
    "gd+gd.yaml"
    "gd+gradreg.yaml"
    "gd+newton.yaml"
)

# Loop through each configuration file and run the script
for CONFIG_FILE in "${CONFIG_FILES[@]}"; do
    echo "Running experiment with config: $CONFIG_FILE"
    python run.py --experiment "$EXPERIMENT" --config "$CONFIG_DIR/$CONFIG_FILE"
    if [ $? -ne 0 ]; then
        echo "Error running experiment with $CONFIG_FILE. Exiting."
        exit 1
    fi
done

echo "All experiments completed successfully."
