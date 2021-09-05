import rasterio
import numpy as np
import os
import pandas as pd
from rasterio.plot import show
from torch.utils.data import Dataset

def getData(mode ='train'):
    if mode =='train':
        # given 1981/1/1~2009/12/31 for training
        for files in os.listdir(r'./data/daily/'):
            fileYear = int(files[:4])
            fileMonth = files[5:7].strip("-")
            fileDay = files[7:].strip("-")

            lowerYear = 1981
            lowerMonth = 5
            upperYear = 2009
            upperYear = 10

            if lowerYear <= fileYear <= upperYear:
                # for training
                dataset = rasterio.open(r'./data/daily/' + files)

    else:
        # given 2010/1/1~2019/12/31 for testing
        pass

class PrecipitationDataset(Dataset):
    def __init__(self):
        pass
    def __len__(self):
        pass
    def __getitem__(self, item):
        for files in os.listdir(r'./data/daily/'):
            year = int(files[:3])
            lower = 1981
            upper = 2009
            if lower <= year <= upper:
                # for training
                dataset = rasterio.open(r'./data/daily/' + files)