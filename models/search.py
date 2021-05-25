import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from models.image_set import ImageSet
from models.retrieval_net import RetrievalNet


class Search(nn.Module):
    def __init__(self, model_path: str, data_root: str, use_cuda: bool):
        super(Search, self).__init__()

        # load pretrained model
        print('load pretrained model')
        net = RetrievalNet(model_path)
        if use_cuda:
            net.cuda()
        self.net = net

        # load precalculated image set
        print('load precalculated image set')
        image_list = ImageSet(data_root, False, True, use_cuda)
        image_set = []
        _ = [image_set.append(image) for image in image_list]
        self.image_set = image_set

        # set parameters
        self.use_cuda = use_cuda

        print('ready for search')

    def forward(self, image: Tensor, top_num: int = 100):

        # load input image
        x = image

        # get input image's feather
        print('get input image\'s feather')
        with torch.no_grad():
            f = self.net(x.unsqueeze(0))

        # calculate cosine similarity
        print('calculate cosine similarity')
        result = []
        y_set = tqdm(self.image_set)
        for y in y_set:
            s = torch.cosine_similarity(f, y.feather)
            result.append([y.id, s.item()])
            y_set.set_description('cos sim: {:.04f}'.format(s.item()))

        result.sort(key=lambda x: x[1], reverse=True)
        return result[:top_num]
