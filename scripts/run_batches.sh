#!/bin/bash

# Navigate to the root directory (assuming the script is called from scripts/)
cd "$(dirname "$0")/.."

# Define the experiment configs (relative to the root directory)
configs=(
  "configs/lenet5/full_batch/lenet5_adam_gradreg_0.1yaml"

)

# Define batch sizes to test
batch_sizes=(64 128 256 50000)

# Path to the run_lenet.py script (in the root directory)
run_script="run_lenet.py"

# Loop through each batch size and config to run the experiment
for batch_size in "${batch_sizes[@]}"
do
  for config in "${configs[@]}"
  do
    echo "Running experiment with config: $config and batch size: $batch_size"

    # Dynamically override the batch size using the --batch_size argument
    python $run_script --config "$config" --batch_size "$batch_size"
  done
done

echo "All experiments completed."