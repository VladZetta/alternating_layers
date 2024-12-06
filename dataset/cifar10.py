import torch
import torchvision
import torchvision.transforms as transforms


def cifar10_dataloaders(root='./../data', batch_size=64):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = torchvision.datasets.CIFAR10(root=root,
                                            train=True,
                                            transform=transform,
                                            download=True)


    test_dataset = torchvision.datasets.CIFAR10(root=root,
                                            train=False,
                                            transform=transform,
                                            download=True)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True)


    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=True)
    return train_loader, test_loader