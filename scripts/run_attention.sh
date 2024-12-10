#!/bin/bash
cd "$(dirname "$0")/.."
# Define the experiment type and directory containing configuration files
EXPERIMENT="attention_layer"
CONFIG_DIR="configs/attention_layer"

# List of YAML configuration files
CONFIG_FILES=(
    "attention_adam.yaml"
    "attention_adam_gradreg.yaml"
    "attention_adam_newton.yaml"
    "attention_GD.yaml"
    "attention_GD_gradreg.yaml"
    "attention_GD_newton.yaml"
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
