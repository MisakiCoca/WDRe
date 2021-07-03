import os

import cv2
import kornia.utils as utils
import numpy as np
import torch
from torch import Tensor


class Image:
    """
    Single image with raw jpg and feather
    """
    def __init__(self, root: str, id_: str, load_image: bool = True, load_feather: bool = True):

        # set parameters
        self.root = root
        self.id = id_

        # load image
        image = self.load_image() if load_image else None
        if load_image and torch.cuda.is_available():
            image = image.cuda()
        self.image = image

        # load feather
        feather = self.load_feather() if load_feather else None
        if load_feather and torch.cuda.is_available():
            feather = feather.cuda()
        self.feather = feather

    def load_image(self) -> Tensor:
        p = os.path.join(self.root, 'image', '{}.jpg'.format(self.id))
        x = cv2.imread(p)
        x = x / 255.0
        x = utils.image_to_tensor(x)
        x = x.type(torch.FloatTensor)
        return x

    def load_feather(self) -> Tensor:
        p = os.path.join(self.root, 'feather', '{}.npy'.format(self.id))
        f = np.load(p)
        f = torch.from_numpy(f)
        return f

    def save_feather(self):
        p = os.path.join(self.root, 'feather', '{}.npy'.format(self.id))
        f = self.feather.numpy()
        np.save(p, f)
