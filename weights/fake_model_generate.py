import torch
import torchvision
import torch.nn as nn


def fake_model_generate(arch: str):
    """
    generate fake model parameters for test
    :param arch: network architecture, such as: resnet18
    """

    # build fake network
    net = torchvision.models.__dict__[arch](num_classes=128)
    if arch == 'resnet18':
        dim_mlp = net.fc.weight.shape[1]
        net.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), net.fc)
    torch.save(net.state_dict(), f'{arch}.pth')


if __name__ == '__main__':
    fake_model_generate('resnet18')
    fake_model_generate('densenet121')
