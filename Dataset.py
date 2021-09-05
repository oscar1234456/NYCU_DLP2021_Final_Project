import rasterio
import numpy as np
import os
import pandas as pd
from rasterio.plot import show
from torch.utils.data import Dataset, DataLoader
import torch

class Date:
    def __init__(self, year, month, day):
        self.year = int(year)
        self.month = int(month)
        self.day = int(day)

    def isAfter(self, date, considerYear=False):
        if not considerYear:
            return self.month > date.month or (self.month == date.month and self.day >= date.day)
        return False

    def isBefore(self, date, considerYear = False):
        if not considerYear:
            return self.month < date.month or (self.month == date.month and self.day <= date.day)

class FileDate:
    def __init__(self, filename):
        self.END_OF_YEAR = 4
        self.END_OF_MONTH = 6
        self.END_OF_DAY = -4
        self.date = Date(self._getYear(filename), self._getMonth(filename), self._getDay(filename))

    def _getYear(self, filename):
        return filename[: self.END_OF_YEAR]

    def _getMonth(self, filename):
        return filename[self.END_OF_YEAR: self.END_OF_MONTH]

    def _getDay(self, filename):
        return filename[self.END_OF_MONTH: self.END_OF_DAY]


class DateRange:
    def __init__(self, startDate, endDate):
        self.startYear = int(startDate.year)
        self.endYear = int(endDate.year)
        self.firstDate = Date(0, startDate.month, startDate.day)
        self.lastDate = Date(0, endDate.month, endDate.day)

    def isValidFileDate(self, fileDate: FileDate):
        return self._isValidYear(fileDate) and\
               fileDate.date.isAfter(self.firstDate) and fileDate.date.isBefore(self.lastDate)

    def _isValidYear(self, fileDate: FileDate):
        return self.startYear <= fileDate.date.year <= self.endYear


class TifData:
    def __init__(self, mode='train'):
        self.FILE_ROOT = r'./data/daily/'
        self.startDateOfTrain = Date(1981, 5, 1)
        self.endDateOfTrain = Date(2010, 11, 8)
        self.trainingDateRange = DateRange(self.startDateOfTrain, self.endDateOfTrain)
        # self.startDateOfTest = ?
        # self.endDateOfTest = ?
        # self.testingDateRange = DateRange(self.startDateOfTest, self.endDateOfTest)
        self.filenameCombinedResult = self.loadData(mode)

    def loadData(self, mode='train'):
        if mode == 'train':
            return self._loadTrainData()
        else:
            return self._loadTestData()

    def _loadTrainData(self):
        result = list()
        tifGroup = list()
        groupSize = 32
        for tif in sorted(os.listdir(self.FILE_ROOT)):
            fileDate = FileDate(tif)
            if self.trainingDateRange.isValidFileDate(fileDate):
                print(f'load Datafile: {tif}')
                tifGroup.append(tif)
                if self._isGroupFilled(tifGroup, groupSize):
                    result.append(tifGroup)
                    tifGroup = list()
        return result

    def _isGroupFilled(self, group, size):
        return len(group) != 0 and len(group) % size == 0

    def _loadTestData(self):
        pass


# def getData(mode ='train'):
#     if mode =='train':
#         # for training
#         # given 1981/1/1~2009/12/31 for training
#         # TODO: think about another sampling way
#         # cut 32days in 5/1~11/8 each years
#         filenameCombinedResult = list()
#         teamList = list()
#         index = 0
#         for files in sorted(os.listdir(r'./data/daily/')):
#
#             fileYear = int(files[:4])
#             fileMonth = int(files[4:6])
#             fileDay = int(files[6:-4])
#
#             lowerYear = 1981
#             lowerMonth = 5
#             upperYear = 2009
#             upperMonth = 11
#             endDayinLastMonth = 8
#
#             if lowerYear <= fileYear <= upperYear and lowerMonth <= fileMonth <= upperMonth:
#                 if fileMonth == upperMonth and fileDay > endDayinLastMonth:
#                     continue
#                 print(f"load Datafile:{files}")
#                 teamList.append(files)
#                 index += 1
#                 if index != 0 and index % 32 == 0:
#                     # 32: means the default length of a sequence
#                     filenameCombinedResult.append(teamList)
#                     teamList = list()
#
#     else:
#         # given 2010/1/1~2019/12/31 for testing
#         pass
#     return filenameCombinedResult

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
        return torch.FloatTensor(result)  #output: 32*32*32


if __name__ == "__main__":
    # combineData = getData(mode = "train")
    tifData = TifData()
    combineData = tifData.filenameCombinedResult
    print(combineData)
