import numpy as np


class NullScheduler():
    def __init__(self, lr=0.01 ):
        super(NullScheduler, self).__init__()
        self.lr    = lr
        self.cycle = 0

    def __call__(self, time):
        return self.lr

    def __str__(self):
        string = 'NullScheduler\n' \
                + 'lr=%0.5f '%(self.lr)
        return string


# 'Cyclical Learning Rates for Training Neural Networks'- Leslie N. Smith, arxiv 2017
#       https://arxiv.org/abs/1506.01186
#       https://github.com/bckenstler/CLR

class CyclicScheduler0():

    def __init__(self, min_lr=0.001, max_lr=0.01, period=10, ratio=1.0 ):
        super(CyclicScheduler0, self).__init__()

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.period = period
        self.ratio  = ratio

    def __call__(self, time):

        #sawtooth
        #r = (1-(time%self.period)/self.period)

        #cosine


        T = int(self.period*self.ratio)
        t = time%self.period
        if t>T:
            r=0
        else:
            r = 0.5*( np.cos(t/T*PI) + 1 )

        lr = self.min_lr + r*(self.max_lr-self.min_lr)
        return lr

    def __str__(self):
        string = 'CyclicScheduler\n' \
                + 'min_lr=%0.3f, max_lr=%0.3f, period=%0.1f, ratio=%0.2f'%(
                     self.min_lr, self.max_lr, self.period, self.ratio)
        return string



# net ------------------------------------
# https://github.com/pytorch/examples/blob/master/imagenet/main.py ###############
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]

    assert(len(lr)==1) #we support only one param_group
    lr = lr[0]

    return lr




