import os

import torch
from tqdm import tqdm

from models.image_set import ImageSet
from models.retrieval_net import RetrievalNet

net = RetrievalNet(os.path.join('weights', 'resnet50.pth'))
net.cuda()

image_set = ImageSet(os.path.join('data'), True, False, True)
loader = tqdm(image_set)

for image in loader:
    loader.set_description('ID: {}'.format(image.id))
    with torch.no_grad():
        f = net(image.image.unsqueeze(0))
        image.feather = f.cpu()
        image.save_feather()
