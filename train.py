import os
import numpy as np
import pandas as pd
import argparse
from timeit import default_timer as timer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from common  import COMMON_STRING, IDENTIFIER, SEED
from augmentations import GridMask, cutmix, train_augment, valid_augment

from dataset import KaggleDataset, NUM_TASK, null_collate, read_data
from schdulers import NullScheduler, CyclicScheduler0, get_learning_rate, adjust_learning_rate
from utils import Logger, metric, compute_kaggle_metric, time_to_str
from model import Serex50_Net, EfficientNet_3

def criterion(logit, truth):
    loss = []
    for i,(l,t) in enumerate(zip(logit,truth)):
        e = F.cross_entropy(l, t)
        loss.append(e)
    return loss

def cutmix_criterion(preds, targets, shuffled_targets, lam):
    loss_1 = criterion(preds, targets)
    loss_2 = criterion(preds, shuffled_targets)
    loss_1 = [lam*l for l in loss_1]
    loss_2 = [(1-lam)*l for l in loss_2]
    loss = [x+y for x,y in zip(loss_1,loss_2)]    
    return loss


def logit_to_probability(logit):
    probability=[]
    for l in logit:
        p = F.softmax(l,1)
        probability.append(p)
    return probability

#------------------------------------
def do_valid(net, valid_loader, out_dir=None):

    valid_loss = np.zeros(6, np.float32)
    valid_num  = np.zeros_like(valid_loss)

    valid_probability = [[],[],[],]
    valid_truth = [[],[],[],]

    for t, (input, truth, infor) in enumerate(valid_loader):
        batch_size = len(infor)

        net.eval()
        input = input.cuda()
        truth = [t.cuda() for t in truth]

        with torch.no_grad():
            logit = net(input)
            probability = logit_to_probability(logit)

            loss = criterion(logit, truth)
            correct = metric(probability, truth)

        loss = [l.item() for l in loss]
        l = np.array([ *loss, *correct, ])*batch_size
        n = np.array([ 1, 1, 1, 1, 1, 1  ])*batch_size
        valid_loss += l
        valid_num  += n

        for i in range(NUM_TASK):
            valid_probability[i].append(probability[i].data.cpu().numpy())
            valid_truth[i].append(truth[i].data.cpu().numpy())

        print('\r %8d /%d'%(valid_num[0], len(valid_loader.dataset)),end='',flush=True)

        pass 
    assert(valid_num[0] == len(valid_loader.dataset))
    valid_loss = valid_loss/(valid_num+1e-8)

    for i in range(NUM_TASK):
        valid_probability[i] = np.concatenate(valid_probability[i])
        valid_truth[i] = np.concatenate(valid_truth[i])
    recall, avgerage_recall = compute_kaggle_metric(valid_probability, valid_truth)


    return valid_loss, (recall, avgerage_recall)


def run_train(args):
    
    out_dir = args.out_dir + '/' + args.model_name
    use_gridmask = args.use_gridmask
    initial_checkpoint = args.initial_checkpoint
    
    if args.scheduler_name == 'null':
        schduler = NullScheduler(lr=0.001)
    else:
        schduler = CyclicScheduler0(min_lr=0.00001, max_lr=0.00005, period=750, ratio=1 )
    
    iter_accum = 1
    batch_size = args.batch_size

    # set-up directories
    for f in ['checkpoint'] : os.makedirs(out_dir +'/'+f, exist_ok=True)

    log = Logger()
    log.open(out_dir+'/log.train.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\t%s\n' % COMMON_STRING)
    log.write('\n')

    log.write('\tSEED         = %u\n' % SEED)
    log.write('\t__file__     = %s\n' % __file__)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')


    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')
    files_train = [f'train_image_data_{fid}.feather' for fid in range(4)]
    data = read_data(args.data_dir, files_train)
    
    df = pd.read_csv(args.df_path)
    train_split = np.load(args.data_dir + '/train_b_fold1_184855.npy').tolist()
    valid_split = np.load(args.data_dir + '/valid_b_fold1_15985.npy').tolist()

    train_df = df[df['image_id'].isin(train_split)]
    valid_df = df[df['image_id'].isin(valid_split)]

    train_dataset = KaggleDataset(
        df       = df,
        data     = data,
        idx      = train_df.index.values, 
        augment  = train_augment if use_gridmask else valid_augment,
    )

    train_loader  = DataLoader(
        train_dataset,
        sampler     = RandomSampler(train_dataset),
        batch_size  = batch_size,
        drop_last   = True,
        num_workers = 4,
        pin_memory  = True,
        collate_fn  = null_collate
    )

    valid_dataset = KaggleDataset(
        df       = df,
        data     = data,
        idx      = valid_df.index.values, 
        augment  = valid_augment,
    )

    valid_loader = DataLoader(
        valid_dataset,
        sampler     = SequentialSampler(valid_dataset),
        batch_size  = batch_size,
        drop_last   = False,
        num_workers = 4,
        pin_memory  = True,
        collate_fn  = null_collate
    )

    assert(len(train_dataset)>=batch_size)
    log.write('batch_size = %d\n'%(batch_size))
    log.write('\n')

    ## net ----------------------------------------
    log.write('** net setting **\n')
    
    if args.model_name == 'serex50':
        net = Serex50_Net().cuda()
    elif args.model_name == 'effnetb3':
        net = EfficientNet_3().cuda()
    else:
        raise NotImplemented
    
    log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)

    if initial_checkpoint is not None:
        state_dict = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
        net.load_state_dict(state_dict,strict=True) 
    else:
        if args.model_name == 'serex50':
            net.load_pretrain(is_print=False)
        else:
            pass

    log.write('net=%s\n'%(type(net)))
    log.write('\n')

    if args.optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()),lr=schduler(0), weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=schduler(0), momentum=0.0, weight_decay = 1e-4)
    
    num_iters   = 3000*1000
    iter_smooth = 50
    iter_log    = 250
    iter_valid  = 500
    iter_save   = [0, num_iters-1]\
                   + list(range(0, num_iters, 1000))#1*1000

    start_iter = 0
    start_epoch= 0
    rate       = 0

    if initial_checkpoint is not None:
        initial_optimizer = initial_checkpoint.replace('_model.pth','_optimizer.pth')
        if os.path.exists(initial_optimizer):
            checkpoint  = torch.load(initial_optimizer)
            start_iter  = checkpoint['iter' ]
            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
        pass

    log.write('optimizer\n  %s\n'%(optimizer))
    log.write('schduler\n  %s\n'%(schduler))
    log.write('\n')

    ## start training here! ##############################################
    log.write('** start training here! **\n')
    log.write('   batch_size=%d,  iter_accum=%d\n'%(batch_size,iter_accum))
    log.write('   experiment  = %s\n' % str(__file__.split('/')[-2:]))
    log.write('                    |----------------------- VALID------------------------------------|------- TRAIN/BATCH -----------\n')
    log.write('rate    iter  epoch | kaggle                    | loss               acc              | loss             | time       \n')
    log.write('----------------------------------------------------------------------------------------------------------------------\n')

    def message(rate, iter, epoch, kaggle, valid_loss, train_loss, batch_loss, mode='print'):
        if mode==('print'):
            asterisk = ' '
            loss = batch_loss
        if mode==('log'):
            asterisk = '*' if iter in iter_save else ' '
            loss = train_loss

        text = \
            '%0.5f %5.1f%s %4.1f | '%(rate, iter/1000, asterisk, epoch,) +\
            '%0.4f : %0.4f %0.4f %0.4f | '%(kaggle[1],*kaggle[0]) +\
            '%4.4f, %4.4f, %4.4f : %4.4f, %4.4f, %4.4f | '%(*valid_loss,) +\
            '%4.4f, %4.4f, %4.4f |'%(*loss,) +\
            '%s' % (time_to_str((timer() - start_timer),'min'))

        return text

    kaggle = (0,0,0,0)
    valid_loss = np.zeros(6,np.float32)
    train_loss = np.zeros(3,np.float32)
    batch_loss = np.zeros_like(train_loss)
    iter = 0
    i    = 0

    start_timer = timer()
    while  iter<num_iters:
        sum_train_loss = np.zeros_like(train_loss)
        sum_train = np.zeros_like(train_loss)

        optimizer.zero_grad()
        for t, (input, truth, infor) in enumerate(train_loader):

            input, truth, shuffled_truth, lam = cutmix(input, truth,alpha=0.3)

            batch_size = len(infor)
            iter  = i + start_iter
            epoch = (iter-start_iter)*batch_size/len(train_dataset) + start_epoch

            if (iter % iter_valid==0):
                valid_loss, kaggle = do_valid(net, valid_loader, out_dir) #
                pass

            if (iter % iter_log==0):
                print('\r',end='',flush=True)
                log.write(message(rate, iter, epoch, kaggle, valid_loss, train_loss, batch_loss, mode='log'))
                log.write('\n')

            if iter in iter_save:
                torch.save({
                    'optimizer': optimizer.state_dict(),
                    'iter'     : iter,
                    'epoch'    : epoch,
                }, out_dir +'/checkpoint/%08d_optimizer.pth'%(iter))
                if iter!=start_iter:
                    torch.save(net.state_dict(),out_dir +'/checkpoint/%08d_model.pth'%(iter))
                    pass

            # learning rate schduler -------------
            lr = schduler(iter)
            if lr<0 : break
            adjust_learning_rate(optimizer, lr)
            rate = get_learning_rate(optimizer)

            net.train()
            
            input = input.cuda()
            truth = [t.cuda() for t in truth]
            shuffled_truth = [t.cuda() for t in shuffled_truth]

            logit = net(input) 
            probability = logit_to_probability(logit)

            loss = cutmix_criterion(logit, truth, shuffled_truth, lam)
        
            ((loss[0]+loss[1]+loss[2] )/iter_accum).backward()
        
            if (iter % iter_accum)==0:
                optimizer.step()
                optimizer.zero_grad()

            loss = [l.item() for l in loss]
            l = np.array([ *loss, ])*batch_size
            n = np.array([ 1, 1, 1 ])*batch_size
            batch_loss      = l/(n+1e-8)
            sum_train_loss += l
            sum_train      += n
            if iter%iter_smooth == 0:
                train_loss = sum_train_loss/(sum_train+1e-12)
                sum_train_loss[...] = 0
                sum_train[...]      = 0

            print('\r',end='',flush=True)
            print(message(rate, iter, epoch, kaggle, valid_loss, train_loss, batch_loss, mode='print'), end='',flush=True)
            i=i+1

        pass  #-- end of one data loader --
    pass #-- end of all iterations --

    log.write('\n')


# main #################################################################
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
    parser.add_argument('--initial_checkpoint', type=str, default=None,
                    help='initial checkpoint to resume training')
    parser.add_argument('--model_name', type=str, default='effnetb3',
                    help='name of model to use: effnetb3 or serex50')
    parser.add_argument('--optimizer_name', type=str, default='SGD',
                    help='name of optimizer to use: SGD or AdamW')                
    parser.add_argument('--scheduler_name', type=str, default='null',
                    help='learning rate scheduler: null or ')
    parser.add_argument('--out_dir', type=str, default='runs', 
                    help='logging directory')                

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']= f'{args.gpu}'

    run_train(args)


