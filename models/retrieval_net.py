import torch
import torch.nn as nn
import torchvision.models.resnet
from torch import Tensor


class RetrievalNet(nn.Module):
    """
    Encoder for retrieval image
    """
    def __init__(self, model_path: str):
        super(RetrievalNet, self).__init__()
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        """
        load resnet18 model
        """

        # init model architecture
        model = torchvision.models.resnet18(num_classes=128)
        dim_mlp = model.fc.weight.shape[1]
        model.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), model.fc)

        # load model state dict
        cp = torch.load(self.model_path, map_location='cpu')
        model.load_state_dict(cp)

        # if available, move model to gpu
        model = model.cuda() if torch.cuda.is_available() else model

        # turn model to eval mode
        model.eval()

        return model

    def save_model(self):
        model = self.model
        torch.save(model.state_dict(), self.model_path)

    def forward(self, x: Tensor) -> Tensor:
        f = self.model(x)
        return f
