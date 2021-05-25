import argparse
import os

import cv2
import kornia.utils as utils
import torch

from tqdm import tqdm
from models.search import Search

# load args
parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, help='input image path')
parser.add_argument('--top', type=int, help='search top num', default=20)
args = parser.parse_args()

# load search model
mp = os.path.join('weights', 'resnet50.pth')
dp = os.path.join('data')
s = Search(mp, dp, False)

# load input image
x = cv2.imread(args.image) / 255.0
x = utils.image_to_tensor(x)
x = x.type(torch.FloatTensor)

# search
res = s(x, args.top)

# show input
x = cv2.imread(args.image)
cv2.imshow('input', x)

# show search result
res = tqdm(res)
for id_, sim in res:
    t = 'id:{} similarity:{:.04f}'.format(id_, sim)
    res.set_description(t)
    p = os.path.join('data', 'image', '{}.jpg'.format(id_))
    y = cv2.imread(p)
    t = t.split(' ')
    cv2.putText(y, t[0], (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(y, t[1], (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('similar', y)
    cv2.waitKey()
