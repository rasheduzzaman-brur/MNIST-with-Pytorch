import torch
from torchvision import datasets, transforms

def dataset(train_arg,test_arg):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_arg)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_arg)

    return train_loader, test_loader
