import rasterio
import numpy as np
import os
import pandas as pd
from rasterio.plot import show
from torch.utils.data import Dataset, DataLoader
import torch

def getData(mode ='train'):
    if mode =='train':
        # for training
        # given 1981/1/1~2009/12/31 for training
        # TODO: think about another sampling way
        # cut 32days in 5/1~11/8 each years
        filenameCombinedResult = list()
        teamList = list()
        index = 0
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
                if fileMonth == upperMonth and fileDay > endDayinLastMonth:
                    continue
                print(f"load Datafile:{files}")
                teamList.append(files)
                index += 1
                if index != 0 and index % 32 == 0:
                    # 32: means the default length of a sequence
                    filenameCombinedResult.append(teamList)
                    teamList = list()

    else:
        # given 2010/1/1~2019/12/31 for testing
        pass
    return filenameCombinedResult

class PrecipitationDatasetLoader(Dataset):
    def __init__(self, mode="train", root="./data/daily/"):
        self.filenameList = getData(mode)
        print(f">>There are {len(self.filenameList)} sequences.")
        self.root = root

    def __len__(self):
        return len(self.filenameList)
    def __getitem__(self, index):
        dataSequenceFilenames = self.filenameList[index]
        # dataSequence: total 32days precipitation data filenames
        result = list()
        for singleFilename in dataSequenceFilenames:
            dataset = rasterio.open(self.root + singleFilename)
            data_array = dataset.read(1)
            # data_array: 40*60 (numpy array)
            result.append(data_array[:32,19:51]) # sample scope: (0,19),(0,50),(31,19),(31,50)
        return torch.FloatTensor(result)  #output: 5*32*32*32


if __name__ == "__main__":
    # combineData = getData(mode = "train")
    # print(combineData)
    train_data = PrecipitationDatasetLoader(mode="train")
    trainLoader = DataLoader(train_data, batch_size=5, shuffle=True)
    test = next(iter(trainLoader))
    print(test)