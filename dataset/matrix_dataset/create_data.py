import numpy as np
import h5py
import os

def generate_and_save_random_matrices(file_path, m=100, n=100, num_matrices=40, seed=42):
    """Generate random matrices and save them to an HDF5 file."""
    np.random.seed(seed)
    matrices_dataset = [np.random.randn(m, n) for _ in range(num_matrices)]

    # Save the dataset in an HDF5 file
    with h5py.File(file_path, 'w') as hf:
        for i, matrix in enumerate(matrices_dataset):
            hf.create_dataset(f'matrix_{i}', data=matrix)
    print(f"Generated and saved {num_matrices} random matrices to {file_path}.")

    return matrices_dataset