import random

import rasterio
import numpy as np
import os
import pandas as pd
from rasterio.plot import show
from torch.utils.data import Dataset, DataLoader
import torch


START_OF_TRAINING_YEAR = 1981
END_OF_TRAINING_YEAR = 2010
START_OF_TESTING_YEAR = 2011
END_OF_TESTING_YEAR = 2020
NUM_OF_BOUNDING_BOX = 16
RAND_BOUNDING_BOX_RANGE = 4
SAMPLES_PER_YEAR = 30
SAMPLE_RANGE = 150
SEQUENCE_LENGTH = 32


def calculateTotalYears(mode):
    if mode == 'train':
        return END_OF_TRAINING_YEAR - START_OF_TRAINING_YEAR + 1
    else:
        return END_OF_TESTING_YEAR - START_OF_TESTING_YEAR + 1



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
    class TifSequence:
        def __init__(self, filenamePos):
            self.startIndex = filenamePos

    def __init__(self, root, mode='train'):
        self.root = root
        self.mode = mode
        self.startDateOfTrain = Date(START_OF_TRAINING_YEAR, 5, 1)
        self.endDateOfTrain = Date(END_OF_TRAINING_YEAR, 11, 8)
        self.trainingDateRange = DateRange(self.startDateOfTrain, self.endDateOfTrain)
        # TODO: load test data
        # self.startDateOfTest = ?
        # self.endDateOfTest = ?
        # self.testingDateRange = DateRange(self.startDateOfTest, self.endDateOfTest)
        self.loadedFilenames = self.loadData(mode)
        self.sequences = self._sampleTifSequences()

    def loadData(self, mode='train'):
        if mode == 'train':
            return self._loadTrainData()
        else:
            return self._loadTestData()

    def _loadTrainData(self):
        result = list()
        for tif in self._getAllFilenames():
            fileDate = FileDate(tif)
            if self.trainingDateRange.isValidFileDate(fileDate):
                result.append(tif)
        return result

    def _sampleTifSequences(self):
        totalYears = calculateTotalYears(self.mode)
        sequences = list()
        for year in range(totalYears):
            startPos = self._sampleStartPos(year)
            sequences.extend(self._generateTifSequencesFromPos(startPos))
        return sequences

    def _sampleStartPos(self, year):
        begin = year * SAMPLE_RANGE
        startPosRange = SAMPLE_RANGE - SEQUENCE_LENGTH
        startPos = random.sample(range(startPosRange), k=SAMPLES_PER_YEAR)
        for i in range(len(startPos)):
            startPos[i] += begin
        return startPos

    def _generateTifSequencesFromPos(self, startPos):
        tifSequences = list()
        for pos in startPos:
            tifSequences.append(self.TifSequence(pos))
        return tifSequences

    def _getAllFilenames(self):
        return sorted(os.listdir(self.root))

    def _isGroupFilled(self, group, size):
        return len(group) != 0 and len(group) % size == 0

    def _loadTestData(self):
        pass


class PrecipitationDataset(Dataset):
    def __init__(self, mode="train", root="./data/daily/"):
        self.mode = mode
        self.numOfBoundingBoxes = NUM_OF_BOUNDING_BOX
        self.randBoundingBoxesRange = RAND_BOUNDING_BOX_RANGE
        self.totalYears = calculateTotalYears(mode)
        self.samplesPerYear = SAMPLES_PER_YEAR
        self.tifs = TifData(root, mode)
        # print(f">>There are {len(self.tifs.loadedFilenames)} sequences.")
        self.root = root

    def __len__(self):
        return self.numOfBoundingBoxes * self.totalYears * self.samplesPerYear

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
    tifData = TifData("./data/daily/")
    combineData = tifData.loadedFilenames
    print(combineData)
