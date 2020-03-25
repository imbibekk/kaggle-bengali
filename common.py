import os
import time
import random
import numpy as np
import torch
from datetime import datetime
#---------------------------------------------------------------------------------
COMMON_STRING ='@%s:  \n' % os.path.basename(__file__)
IDENTIFIER   = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

if 1:
    SEED = int(time.time()) #35202   #35202  #123  #
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    COMMON_STRING += '\tset random seed\n'
    COMMON_STRING += '\t\tSEED = %d\n'%SEED

    torch.backends.cudnn.benchmark     = False  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    torch.backends.cudnn.enabled       = True
    torch.backends.cudnn.deterministic = True

    COMMON_STRING += '\tset cuda environment\n'
    COMMON_STRING += '\t\ttorch.__version__              = %s\n'%torch.__version__
    COMMON_STRING += '\t\ttorch.version.cuda             = %s\n'%torch.version.cuda
    COMMON_STRING += '\t\ttorch.backends.cudnn.version() = %s\n'%torch.backends.cudnn.version()
    COMMON_STRING += '\t\ttorch.cuda.device_count()      = %d\n'%torch.cuda.device_count()
    #print ('\t\ttorch.cuda.current_device()    =', torch.cuda.current_device())


COMMON_STRING += '\n'

#---------------------------------------------------------------------------------
## useful : http://forums.fast.ai/t/model-visualization/12365/2


if __name__ == '__main__':
    print (COMMON_STRING)
