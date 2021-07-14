import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from models.image_set import ImageSet
from models.retrieval_net import RetrievalNet


class SearchNet(nn.Module):
    """
    Image search pipeline
    """
    def __init__(self, model_arch: str, data_root: str):
        super(SearchNet, self).__init__()

        # load pretrained model
        print('loading pretrained model')
        net = RetrievalNet(model_arch)
        self.net = net

        # load precalculated image set
        print('loading precalculated image set')
        image_list = ImageSet(data_root, False, True)
        image_set = []
        _ = [image_set.append(image) for image in image_list]
        self.image_set = image_set

        # ready for search
        print('ready for search')

    def forward(self, image: Tensor, top_num: int = 100):
        """
        return top k similarity images with input image
        """

        # load input image
        x = image.cuda() if torch.cuda.is_available() else image

        # get input image's feather
        print('calculating input image\'s feather')
        with torch.no_grad():
            f = self.net(x.unsqueeze(0))

        # calculate cosine similarity
        print('calculating cosine similarity')
        result = []
        y_set = tqdm(self.image_set)
        for y in y_set:
            s = torch.cosine_similarity(f, y.feather)
            result.append([y.id, s.item()])
            y_set.set_description('cos sim: {:.04f}'.format(s.item()))

        # sort by similarity
        result.sort(key=lambda x: x[1], reverse=True)

        return result[:top_num]
