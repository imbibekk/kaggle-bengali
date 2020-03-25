import sys
import re
import numpy as np
import builtins
import pickle
import pandas as pd
from sklearn import metrics as sklearn_metrics


# http://stackoverflow.com/questions/34950201/pycharm-print-end-r-statement-not-working
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  #stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode ='w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1 ):
        if '\r' in message: is_file=0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
            #time.sleep(1)

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


class Struct(object):
    def __init__(self, is_copy=False, **kwargs):
        self.add(is_copy, **kwargs)

    def add(self, is_copy=False, **kwargs):
        #self.__dict__.update(kwargs)

        if is_copy == False:
            for key, value in kwargs.items():
                setattr(self, key, value)
        else:
            for key, value in kwargs.items():
                try:
                    setattr(self, key, copy.deepcopy(value))
                    #setattr(self, key, value.copy())
                except Exception:
                    setattr(self, key, value)

    def __str__(self):
        text =''
        for k,v in self.__dict__.items():
            text += '\t%s : %s\n'%(k, str(v))
        return text



#https://stackoverflow.com/questions/43162506/undefinedmetricwarning-f-score-is-ill-defined-and-being-set-to-0-0-in-labels-wi
def metric(probability, truth):

    correct = []
    for p,t in zip(probability,truth):
        p = p.data.cpu().numpy()
        t = t.data.cpu().numpy()
        y = p.argmax(-1)
        c = np.mean(y==t)
        correct.append(c)

    return correct


def compute_kaggle_metric(probability, truth):

    def compute_recall(probability,truth):
        num_class = probability.shape[-1]
        y = probability.argmax(-1)
        t = truth
        correct = y==t

        recall = np.zeros(num_class)
        for c in range(num_class):
            e = correct[t==c]
            if len(e)>0:
                recall[c]=e.mean()
        return recall

    recall = []
    for p,t in zip(probability,truth):
        r = compute_recall(p,t)
        r = r.mean()
        recall.append(r)

    avgerage_recall = np.average(recall, weights=[2,1,1])
    return recall, avgerage_recall



def read_pickle_from_file(pickle_file):
    with open(pickle_file,'rb') as f:
        x = pickle.load(f)
    return x

def write_pickle_to_file(pickle_file, x):
    with open(pickle_file, 'wb') as f:
        pickle.dump(x, f, pickle.HIGHEST_PROTOCOL)

def df_loc_by_list(df, key, values):
    df = df.loc[df[key].isin(values)]
    df = df.assign(sort = pd.Categorical(df[key], categories=values, ordered=True))
    df = df.sort_values('sort')
    #df = df.reset_index()
    df = df.drop('sort', axis=1)
    return  df

def write_list_to_file(list_file, strings):
    with open(list_file, 'w') as f:
        for s in strings:
            f.write('%s\n'%str(s))
    pass


def read_list_from_file(list_file, comment='#'):
    with open(list_file) as f:
        lines  = f.readlines()

    strings=[]
    for line in lines:
        if comment is not None:
            s = line.split(comment, 1)[0].strip()
        else:
            s = line.strip()

        if s != '':
            strings.append(s)
    return strings


def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)

    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)

    else:
        raise NotImplementedError
