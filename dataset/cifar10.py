import torch
import torchvision
import torchvision.transforms as transforms

def cifar10_dataloaders(root='./../data', batch_size=64, num_workers=4, pin_memory=True):
    """
    Returns train and test DataLoader objects for the CIFAR-10 dataset.

    Args:
        root (str): Path to the dataset.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of worker threads for data loading.
        pin_memory (bool): Whether to use pinned memory for faster GPU transfer.

    Returns:
        tuple: train_loader, test_loader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=root,
        train=True,
        transform=transform,
        download=True
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=root,
        train=False,
        transform=transform,
        download=True
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, test_loader
