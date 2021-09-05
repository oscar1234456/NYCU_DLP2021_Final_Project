import rasterio
import numpy as np
import os
import pandas as pd
from rasterio.plot import show
from torch.utils.data import Dataset

def getData(mode ='train'):
    if mode =='train':
        # given 1981/1/1~2009/12/31 for training
        # TODO: think about another sampling way
        # cut 32days in 5/1~11/8 each years
        for files in sorted(os.listdir(r'./data/daily/')):
            fileYear = int(files[:4])
            fileMonth = int(files[4:6])
            fileDay = int(files[6:-4])

            lowerYear = 1981
            lowerMonth = 5
            upperYear = 2009
            upperMonth = 11
            endDayinLastMonth = 8

            if lowerYear <= fileYear <= upperYear and lowerMonth <= fileMonth <= upperMonth:
                # for training
                # dataset = rasterio.open(r'./data/daily/' + files)
                if fileMonth==upperMonth and fileDay>endDayinLastMonth:
                    continue
                print(f"load Datafile:{files}")


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

if __name__ == "__main__":
    getData(mode = "train")