import argparse
import math
import os
import shutil

import torch.backends.cudnn
import torch.cuda
import torch.utils.data
import torchvision.models
import torchvision.transforms as trs
from tqdm import tqdm

from models.moco import MoCo
from models.wafer_folder import WaferFolder

# init hyper parameters
parser = argparse.ArgumentParser('wafer classifier training')
# network
parser.add_argument('--arch', default='resnet18', help='model architecture')
# dataset
parser.add_argument('--data', default=os.path.join('data', 'raw'), help='path to npy dataset')
parser.add_argument('--workers', default=32, type=int, help='number of data loading workers')
parser.add_argument('--bs', '--batch_size', default=256, type=int, help='mini-batch size, total of all GPUs')
# training
parser.add_argument('--epochs', default=500, type=int, help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
# checkpoint
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint')
parser.add_argument('--cache', default='cache', type=str, help='checkpoint save folder')
# optimizer
parser.add_argument('--lr', '--learning_rate', default=0.03, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum of SGD solver')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float, help='weight decay')
# logger
parser.add_argument('--print_freq', default=10, type=int, help='print frequency')

# moco specific
parser.add_argument('--moco_dim', default=128, type=int, help='feature dimension')
parser.add_argument('--moco_k', default=2560, type=int, help='queue size; number of negative keys')
parser.add_argument('--moco_m', default=0.999, type=float, help='moco momentum of updating key encoder')
parser.add_argument('--moco_t', default=0.2, type=float, help='softmax temperature')

# parse args
args = parser.parse_args()


def main():
    """
    whole train process moco model on wafer dataset
    """

    # only support cuda training
    if not torch.cuda.is_available():
        raise Exception('You must have CUDA to train')

    # create model
    print(f'creating model {args.arch}')
    model = MoCo(torchvision.models.__dict__[args.arch], args.moco_dim, args.moco_k, args.moco_m, args.moco_t)
    print(model)

    # make model data parallel
    model = model.cuda()
    model = torch.nn.DataParallel(model)

    # define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, args.momentum, weight_decay=args.wd)

    # check cache folder
    if not os.path.exists(args.cache):
        os.makedirs(args.cache)

    # optionally resume from a checkpoint
    if args.resume:
        print(f'loading checkpoint {args.resume}')
        cp = torch.load(args.resume)  # checkpoint state dict
        args.start_epoch = cp['epoch']
        model.load_state_dict(cp['state_dict'])
        optimizer.load_state_dict(cp['optimizer'])
        print(f'loaded checkpoint {args.resume} (epoch: {cp["epoch"]})')

    # pre-optimise conv layers
    torch.backends.cudnn.benchmark = True

    # transforms & augmentation
    t = trs.Compose([
        trs.Resize((224, 224)),
        trs.RandomRotation(180),
    ])

    # data loading (trs -> torchvision.transforms)
    s = WaferFolder(args.data, t)
    if len(s) < args.bs:
        raise Exception('Since we use drop last, data size must bigger than batch size.')
    train_loader = torch.utils.data.DataLoader(
        dataset=s,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    # train for epochs
    for epoch in range(args.start_epoch, args.epochs):
        # adjust learning rate
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epochs
        train(train_loader, model, criterion, optimizer, epoch, args)

        # save if need
        n = os.path.join(args.cache, f'cp_{epoch:04d}.pth')
        save_checkpoint(model, optimizer, epoch, args.arch, is_best=False, filename=n)


def train(train_loader, model, criterion, optimizer, epoch, args):
    """
    train for one epoch
    """

    # switch to train mode
    model.train()

    # build process bar
    loader = tqdm(train_loader)

    # run one epoch
    for [im_q, im_k], _ in loader:
        # move imgs to cuda
        im_q = im_q.cuda(non_blocking=True)
        im_k = im_k.cuda(non_blocking=True)

        # compute output
        output, target = model(im_q, im_k)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update process bar
        loader.set_description(f'epoch: [{epoch}/{args.epochs}] loss:{loss.item():.2f}')


def adjust_learning_rate(optimizer, epoch, args):
    """
    change learning rate according to epoch
    """
    lr = args.lr
    # cosine lr schedule
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for group in optimizer.param_groups:
        group['lr'] = lr


def save_checkpoint(model, optimizer, epoch, arch, is_best, filename='cp.pth'):
    """
    save checkpoint of [model, optimizer, epoch, arch]
    """
    state = {'epoch': epoch + 1, 'arch': arch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'best.pth')


if __name__ == '__main__':
    main()
