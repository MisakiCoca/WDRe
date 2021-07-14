import argparse
import os
import warnings

import cv2
import kornia.utils as utils
import torch
from tqdm import tqdm

from models.search_net import SearchNet
from tools.check_feather_bank import check_feather_bank
from tools.model_transform import model_transform

warnings.filterwarnings('ignore')


def init_args():
    """
    get hyper parameters
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--image', help='input image path')
    parser.add_argument('--top', type=int, help='search top num', default=20)
    parser.add_argument('--data_root', help='data root path', default='data')
    parser.add_argument('--arch', help='network architecture', default='resnet18')
    parser.add_argument('--cp_final', help='moco checkpoint', default=os.path.join('cache', 'cp_final.pth'))
    parser.add_argument('--skip_check', help='skip feather bank check', action='store_true')

    args = parser.parse_args()
    return args


def check_pre(args):
    """
    inspection preparation
    """

    # check search model
    model_path = os.path.join('weights', f'{args.arch}.pth')
    if not os.path.exists(model_path):
        moco_checkpoint = args.cp_final
        model_transform(moco_checkpoint, args.arch)

    # check feather bank
    if not args.skip_check:
        check_feather_bank(args.arch, args.data_root)


def load_model(args):
    """
    load search net
    """
    s = SearchNet(args.arch, args.data_root)
    return s


def search(search_net, image_name, args):
    """
    search one image with image name
    """

    # load input image
    x = cv2.imread(image_name) / 255.0
    x = utils.image_to_tensor(x)
    x = x.type(torch.FloatTensor)

    # search operation
    res = search_net(x, args.top)

    # show input
    x = cv2.imread(image_name)
    cv2.imshow('input', x)

    # show search result
    res = tqdm(res)
    for id_, sim in res:
        t = 'id:{} similarity:{:.04f}'.format(id_, sim)
        res.set_description(t)
        p = os.path.join('data', 'image', '{}.jpg'.format(id_))
        y = cv2.imread(p)
        t = t.split(' ')
        cv2.putText(y, t[0], (10, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 255), 2)
        cv2.putText(y, t[1], (10, 80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 255), 2)
        cv2.imshow('similar', y)
        cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    # init hyper parameters
    args = init_args()

    # check preparation
    check_pre(args)

    # load search network
    s = load_model(args)

    # switch single mode and multi mode
    if args.image:
        print(f'Single search mode: {args.image}')
        search(s, args.image, args)
    else:
        print('Multi search mode')
        while True:
            image_name = input('Search: ')
            search(s, image_name, args)
