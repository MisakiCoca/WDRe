import torch


def model_transform(moco_checkpoint, checkpoint):
    """
    transform moco checkpoint to resnet18 checkpoint
    :param moco_checkpoint: path to moco checkpoint
    :param checkpoint: path to save checkpoint
    """

    # transforming model
    print('transforming moco model to search model')

    # load moco checkpoint
    moco = torch.load(moco_checkpoint)['state_dict']
    net = {}

    # filter key with 'module.encoder_q'
    for k in moco:
        if 'module.encoder_q' in k:
            x = k.replace('module.encoder_q.', '')
            net[x] = moco[k]

    # save resnet18 checkpoint
    torch.save(net, checkpoint)

    # transform over
    print('search model is ready')
