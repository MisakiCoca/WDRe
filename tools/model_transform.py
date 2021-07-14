import os

import torch


def model_transform(moco_checkpoint, arch):
    """
    transform moco checkpoint to search checkpoint
    :param moco_checkpoint: path to moco checkpoint
    :param arch: search network architecture, resnet18 or densenet121
    """

    # transforming model
    print(f'transforming moco model to {arch} model')

    # load moco checkpoint
    moco = torch.load(moco_checkpoint)['state_dict']
    net = {}

    # filter key with 'module.encoder_q'
    for k in moco:
        if 'module.encoder_q' in k:
            x = k.replace('module.encoder_q.', '')
            net[x] = moco[k]

    # save search checkpoint
    torch.save(net, os.path.join('weights', f'{arch}.pth'))

    # transform over
    print(f'{arch} model is ready')
