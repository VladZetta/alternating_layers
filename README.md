# Alternating Optimization for Neural Network Training

## Overview

This project explores an **alternating optimization approach** for training neural networks. It leverages:
- **First-order methods** for the majority of layers, ensuring computational efficiency.
- **Second-order updates** for the final layer, achieving high precision where it matters most.

### Key Highlights:
- **Scalability & Convergence Accuracy**: Balances computational efficiency and precise optimization for large-scale training.
- **Distributed Learning Potential**: Demonstrates promise for **federated learning** by reducing communication costs and enhancing robustness to data heterogeneity across clients.
- **Layered Optimization Structure**: Combines generalized feature learning across layers with layer-specific fine-tuning, inspired by the partially personalized federated learning framework proposed by **Mishchenko et al. [2]**.

---

## Running the Code

The experiments can be executed using the `run.py` file. The script supports the following experiments:
- **Matrix Factorization**
- **Attention Layer**
- **LeNet-5 Training**

### Syntax
```bash
python run.py --experiment <experiment_name> --config <path_to_config_file>
