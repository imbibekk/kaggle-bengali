
"""
Stochastic Weight Averaging (SWA)
Averaging Weights Leads to Wider Optima and Better Generalization
https://github.com/timgaripov/swa
"""
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

from dataset import KaggleDataset, null_collate, read_data
from augmentations import train_augment, valid_augment
from model import Serex50_Net, EfficientNet_3


def moving_average(net1, net2, alpha=1.):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0

    pbar = tqdm(loader, unit="images", unit_scale=loader.batch_size)
    for batch in pbar:
        input, truth, infor = batch
        input = input.cuda()
        b = input.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(input)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Kaggle Bengali Competition')
    parser.add_argument('--gpu', type=int, default=0, 
                    help='Choose GPU to use. This only support single GPU')
    parser.add_argument('--data_dir', default='./data',
                    help='datasest directory')              
    parser.add_argument('--df_path', default='./data/train.csv',
                    help='df_path')
    parser.add_argument('--batch_size', type=int, default=40, 
                    help='batch size')
    parser.add_argument('--use_gridmask', action='store_true', default=False, 
                    help='whether to use grid_mask augmentation or not')
    parser.add_argument('--num_snapshots', type=int, default=30, 
                    help='number of checkpoints to use for swa')
    parser.add_argument('--model_name', type=str, default='effnetb3',
                    help='name of model to use: effnetb3 or serex50')
    parser.add_argument('--out_dir', type=str, default='runs', 
                    help='logging directory')                

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']= f'{args.gpu}'

    directory = Path(args.out_dir+f'/{args.model_name}/checkpoint')
    files = [f for f in directory.iterdir() if f.suffix == ".pth"]
    files = sorted([f for f in files if 'model' in str(f)])

    files = files[-args.num_snapshots:]
    output_name = f'models_swa_{len(files)}.pth'
    print('No of snapshots: ', len(files))
    
    if args.model_name == 'serex50':
        net = Serex50_Net().cuda()
    elif args.model_name == 'effnetb3':
        net = EfficientNet_3().cuda()
    else:
        raise NotImplemented
    
    state_dict = torch.load(files[0], map_location=lambda storage, loc: storage)
    net.load_state_dict(state_dict,strict=True)  #True
        
    for i,checkpoint in enumerate(files[1:]):
        print(i, checkpoint)
        
        if args.model_name == 'serex50':
            net2 = Serex50_Net().cuda()
        elif args.model_name == 'effnetb3':
            net2 = EfficientNet_3().cuda()
        else:
            raise NotImplemented
    
        state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        net2.load_state_dict(state_dict,strict=True)

        moving_average(net, net2, 1. / (i + 2))

    ## dataset ----------------------------------------
    files_train = [f'train_image_data_{fid}.feather' for fid in range(4)]
    data = read_data(args.data_dir, files_train)
    
    df = pd.read_csv(args.df_path)
    train_split = np.load(args.data_dir + '/train_b_fold1_184855.npy').tolist()
    train_df = df[df['image_id'].isin(train_split)]

    train_dataset = KaggleDataset(
        df       = df,
        data     = data,
        idx      = train_df.index.values, 
        augment  = train_augment if args.use_gridmask else valid_augment,
    )

    train_loader  = DataLoader(
        train_dataset,
        sampler     = RandomSampler(train_dataset),
        batch_size  = args.batch_size,
        drop_last   = True,
        num_workers = 4,
        pin_memory  = True,
        collate_fn  = null_collate
    )

    net.cuda()
    bn_update(train_loader, net)
    torch.save(net.state_dict(), args.out_dir +f'/{args.model_name}/' + output_name)
