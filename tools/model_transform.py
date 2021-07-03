import torch


def model_transform(moco_checkpoint, resnet18_checkpoint):
    """
    transform moco checkpoint to resnet18 checkpoint
    :param moco_checkpoint: path to moco checkpoint
    :param resnet18_checkpoint: path to save resnet18 checkpoint
    """

    # transforming model
    print('transforming moco model to resnet18 model')

    # load moco checkpoint
    moco = torch.load(moco_checkpoint)['state_dict']
    resnet18 = {}

    # filter key with 'module.encoder_q'
    for k in moco:
        if 'module.encoder_q' in k:
            x = k.replace('module.encoder_q.', '')
            resnet18[x] = moco[k]

    # save resnet18 checkpoint
    torch.save(resnet18, resnet18_checkpoint)

    # transform over
    print('resnet18 model is ready')
