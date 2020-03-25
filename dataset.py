import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from utils import Struct, df_loc_by_list


TASK_NAME = [ 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic' ]
NUM_TASK = len(TASK_NAME)

HEIGHT = 137
WIDTH = 236


def read_data(data_dir, files):
    tmp = []
    for f in files:
        F = os.path.join(data_dir, f)
        data = pd.read_feather(F)
        res = data.iloc[:, 1:].values
        imgs = []
        for i in tqdm(range(res.shape[0])):
            img = res[i].squeeze().reshape(HEIGHT, WIDTH)
            #img = cv2.resize(img, (224, 224))
            imgs.append(img)
        imgs = np.asarray(imgs)
        
        tmp.append(imgs)
    tmp = np.concatenate(tmp, 0)
    return tmp


class KaggleDataset(Dataset):
    def __init__(self, df, data, idx, augment=None):

        self.df = df.reset_index()
        self.data = data
        self.idx = np.asarray(idx)
        self.augment = augment

    def __len__(self):
        return self.idx.shape[0]

    def __getitem__(self, index):
        index = self.idx[index]
        img_id = self.df.iloc[index].image_id
        
        image = self.data[index]
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        
        grapheme = self.df.iloc[index].grapheme
        grapheme_root = self.df.iloc[index].grapheme_root
        vowel_diacritic = self.df.iloc[index].vowel_diacritic
        consonant_diacritic = self.df.iloc[index].consonant_diacritic
        
        label = [grapheme_root, vowel_diacritic, consonant_diacritic]
        
        infor = Struct(
            index    = index,
            image_id = img_id,
            grapheme = grapheme,
        )

        image, label, infor = self.augment(image, label, infor)
        image = image.astype(np.float32)/255

        return image, label, infor
        

def null_collate(batch):
    batch_size = len(batch)

    input = []
    label = []
    infor = []
    for b in range(batch_size):
        input.append(batch[b][0])
        label.append(batch[b][1])
        infor.append(batch[b][-1])

    input = np.stack(input)
    input = input.transpose(0,3,1,2)

    label = np.stack(label)

    input = torch.from_numpy(input).float()
    truth = torch.from_numpy(label).long()
    truth0, truth1, truth2 = truth[:,0],truth[:,1],truth[:,2]
    truth = [truth0, truth1, truth2]
    return input, truth, infor



