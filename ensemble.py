import os
import argparse

from utils import Logger, compute_kaggle_metric, read_list_from_file, read_pickle_from_file
from dataset import TASK_NAME


def run_ensemble(args):
    ensemble_dir=[
    './runs/serex50/submit', 
    './runs/effnetb3/submit'
    ]
    out_dir = args.out_dir + '/ensemble'

    ############################################################
    os.makedirs(out_dir, exist_ok=True)
    log = Logger()
    log.open(out_dir+'/log.ensemble.txt',mode='a')

    if 1:
        test_probability = [0,0,0] # 8bit
        num_ensemble=0
        test_truth = None

        for t,d in enumerate(ensemble_dir):
            log.write('%d  %s\n'%(t,d))

            image_id    = read_list_from_file(d +'/image_id.txt')
            probability = read_pickle_from_file(d + '/probability.pickle')
            truth       = read_pickle_from_file(d + '/truth.pickle')

            test_probability = [p+q for p,q in zip(test_probability,probability)]
            num_ensemble += 1
        print('done')
        print('')

        probability = [p/num_ensemble for p in test_probability]

        recall, avgerage_recall = compute_kaggle_metric(probability, truth)
        log.write('avgerage_recall : %f\n'%(avgerage_recall))

        for i,name in enumerate(TASK_NAME):
            log.write('%28s  %f\n'%(name,recall[i]))
        log.write('\n')


# main #################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kaggle Bengali Competition')
    parser.add_argument('--out_dir', type=str, default='runs', 
                    help='logging directory')                

    args = parser.parse_args()
    run_ensemble(args)

