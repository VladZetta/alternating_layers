#!/bin/bash

# Navigate to the root directory (assuming the script is called from `scripts/`)
cd "$(dirname "$0")/.."

# Define the experiment configs (relative to the root directory)
configs=(
  "configs/lenet5/full_batch/lenet5_adam_newton.yaml"
  "configs/lenet5/full_batch/lenet5_adam_gradreg_0.1.yaml"
  "configs/lenet5/full_batch/lenet5_adam_gradreg_1.0.yaml"
  "configs/lenet5/full_batch/lenet5_adam.yaml"
  "configs/lenet5/full_batch/lenet5_sgd.yaml"
)

# Path to the run_lenet.py script (in the root directory)
run_script="run_lenet.py"

# Loop through each config and run the experiment
for config in "${configs[@]}"
do
  echo "Running experiment with config: $config"
  python $run_script --config "$config"
done

echo "All experiments completed."
