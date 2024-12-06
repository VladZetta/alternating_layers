import torch

# Check if CUDA is available and the GPU is accessible
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available! Using device:", torch.cuda.get_device_name(0))
else:
    raise RuntimeError("CUDA is not available. Ensure that the GPU is properly installed and accessible.")

# Define two random matrices (3x3) on the GPU
a = torch.randn(3, 3, device=device)
b = torch.randn(3, 3, device=device)

# Perform matrix multiplication
c = torch.matmul(a, b)

# Print the result of the matrix multiplication
print("Matrix multiplication result:\n", c)