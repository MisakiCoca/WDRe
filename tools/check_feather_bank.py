import os

import torch
from tqdm import tqdm

from models.image_set import ImageSet
from models.retrieval_net import RetrievalNet


def check_feather_bank(arch, data_root):
    """
    Check the integrity of the feather bank.
    If missing, compute images and save their feather.
    """

    # start check feather bank
    print(f'checking {arch} feather bank')
    folder = os.path.join(data_root, f'feather_{arch}')
    if not os.path.exists(folder):
        os.makedirs(folder)

    # lazy load search model
    net = RetrievalNet(arch)

    # load image set without feather
    image_set = ImageSet(root=data_root, load_image=True, load_feather=False, arch=arch)
    loader = tqdm(image_set)

    # check for the presence of feathers
    for image in loader:
        loader.set_description(f'checking ID: {image.id}')

        # if missing, compute
        if not os.path.exists(os.path.join(data_root, f'feather_{arch}', f'{image.id}.npy')):
            loader.set_description(f'generating ID: {image.id}')
            with torch.no_grad():
                f = net(image.image.unsqueeze(0))
                image.feather = f.cpu()
                image.save_feather()

    # feather bank check over
    print(f'{arch} feather bank is ready')
