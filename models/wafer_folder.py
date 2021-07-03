import os
from glob import glob

import numpy as np
import torch
import torch.utils.data as data


class WaferFolder(data.Dataset):
    """
    Build a wafer folder with root path
    """

    def __init__(self, root, transform):
        """
        :param root: wafer folder root path and transforms
        :param transform: enhancements to be applied to the image
        """
        super(WaferFolder, self).__init__()

        # create image list
        self.imgs = glob(os.path.join(root, '*.npy'))

        # pass transforms
        self.transform = transform

    def __getitem__(self, index):
        """
        read numpy file and convert to tensor
        :return: [Tensor, Tensor], label
        """

        # get path and label
        p = self.imgs[index]
        label = os.path.basename(p).split('.')[0]

        # read numpy image and convert to Tensor
        i = np.load(p)
        i = torch.from_numpy(i).float().unsqueeze(0)
        i = torch.cat([i, i, i])

        # image transforms
        im_q = self.transform(i)
        im_k = self.transform(i)

        return [im_q, im_k], label

    def __len__(self):
        """
        length of image list
        """
        return len(self.imgs)
