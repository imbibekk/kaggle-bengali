import os
import pandas as pd
import argparse
import numpy as np
from timeit import default_timer as timer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler

from common  import COMMON_STRING, IDENTIFIER, SEED
from augmentations import valid_augment

from dataset import KaggleDataset, NUM_TASK, TASK_NAME, null_collate, read_data
from utils import Logger, metric, compute_kaggle_metric, time_to_str
from utils import read_list_from_file, read_list_from_file, read_pickle_from_file, write_list_to_file, write_pickle_to_file
from model import Serex50_Net, EfficientNet_3


def criterion(logit, truth):
    loss = []
    for i,(l,t) in enumerate(zip(logit,truth)):
        e = F.cross_entropy(l, t)
        loss.append(e)
    return loss

def logit_to_probability(logit):
    probability=[]
    for l in logit:
        p = F.softmax(l,1)
        probability.append(p)
    return probability

def do_evaluate(net, test_dataset, batch_size, augment=[]):

    test_loader = DataLoader(
        test_dataset,
        sampler     = SequentialSampler(test_dataset),
        batch_size  = batch_size,
        drop_last   = False,
        num_workers = 4,
        pin_memory  = True,
        collate_fn  = null_collate
    )
    #----
    start_timer = timer()

    test_num  = 0
    test_id   = []
    test_probability = [[],[],[]]
    test_truth = [[],[],[]]

    start_timer = timer()
    for t, (input, truth, infor) in enumerate(test_loader):

        batch_size,C,H,W = input.shape
        input = input.cuda()

        with torch.no_grad():
            net.eval()

            num_augment=0
            probability=[0,0,0]
            if 'null' in augment: #null
                logit =  net(input)
                prob  = logit_to_probability(logit)

                probability = [p+q**0.5 for p,q in zip(probability,prob)]
                num_augment += 1

            probability = [p/num_augment for p in probability]

        batch_size  = len(infor)
        for i in range(NUM_TASK):
            test_probability[i].append(probability[i].data.cpu().numpy())
            test_truth[i].append(truth[i].data.cpu().numpy())

        test_id.extend([i.image_id for i in infor])
        test_num += batch_size

        print('\r %4d / %4d  %s'%(
             test_num, len(test_loader.dataset), time_to_str((timer() - start_timer),'min')
        ),end='',flush=True)

    assert(test_num == len(test_loader.dataset))
    print('')

    for i in range(NUM_TASK):
        test_probability[i] = np.concatenate(test_probability[i])
        test_truth[i] = np.concatenate(test_truth[i])

    print(time_to_str((timer() - start_timer),'sec'))
    return test_id, test_truth, test_probability


######################################################################################
def run_submit(args):    
        
    augment = ['null'] 
    out_dir = args.out_dir + f'/{args.model_name}'
    initial_checkpoint = args.initial_checkpoint
    batch_size = args.batch_size

    ## setup out_dir
    os.makedirs(out_dir +'/submit', exist_ok=True)

    log = Logger()
    log.open(out_dir+'/log.submit.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\t%s\n' % COMMON_STRING)
    log.write('\n')
    log.write('\tSEED         = %u\n' % SEED)
    log.write('\t__file__     = %s\n' % __file__)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')

    log.write('submitting .... @ %s\n'%str(augment))
    log.write('initial_checkpoint  = %s\n'%initial_checkpoint)
    log.write('\n')

    if 1: #save
        log.write('** dataset setting **\n')
        files_train = [f'train_image_data_{fid}.feather' for fid in range(4)]
        data = read_data(args.data_dir, files_train)
        
        df = pd.read_csv(args.df_path)
        valid_split = np.load(args.data_dir + '/valid_b_fold1_15985.npy').tolist()
        valid_df = df[df['image_id'].isin(valid_split)]

        test_dataset = KaggleDataset(
            df       = df,
            data     = data,
            idx      = valid_df.index.values, 
            augment  = valid_augment,
        )

        log.write('\n')

        ## net
        log.write('** net setting **\n')
        if args.model_name == 'serex50':
            net = Serex50_Net().cuda()
        elif args.model_name == 'effnetb3':
            net = EfficientNet_3().cuda()
        else:
            raise NotImplemented

        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage), strict=True)

        image_id, truth, probability = do_evaluate(net, test_dataset, batch_size,  augment)


        if 1: #save
            write_list_to_file (out_dir + '/submit/image_id.txt',image_id)
            write_pickle_to_file(out_dir + '/submit/probability.pickle', probability)
            write_pickle_to_file(out_dir + '/submit/truth.pickle', truth)

    if 1:
        image_id = read_list_from_file(out_dir + '/submit/image_id.txt')
        probability = read_pickle_from_file(out_dir + '/submit/probability.pickle')
        truth       = read_pickle_from_file(out_dir + '/submit/truth.pickle')
    num_test= len(image_id)

    if 1:
        recall, avgerage_recall = compute_kaggle_metric(probability, truth)
        log.write('avgerage_recall : %f\n'%(avgerage_recall))

        for i,name in enumerate(TASK_NAME):
            log.write('%28s  %f\n'%(name,recall[i]))
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
    parser.add_argument('--initial_checkpoint', type=str, default=None,
                    help='initial checkpoint to resume training')
    parser.add_argument('--model_name', type=str, default='effnetb3',
                    help='name of model to use: effnetb3 or serex50')
    parser.add_argument('--out_dir', type=str, default='runs', 
                    help='logging directory')                

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']= f'{args.gpu}'

    run_submit(args)


