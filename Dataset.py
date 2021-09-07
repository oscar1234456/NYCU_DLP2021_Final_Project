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
BOUNDING_BOX_START_POS = (0, 17)


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
        self.date = Date(self._extractYear(filename), self._extractMonth(filename), self._extractDay(filename))

    def _extractYear(self, filename):
        return filename[: self.END_OF_YEAR]

    def _extractMonth(self, filename):
        return filename[self.END_OF_YEAR: self.END_OF_MONTH]

    def _extractDay(self, filename):
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
        def __init__(self, filenamePos, offset):
            self.startIndex = filenamePos
            self.offset = offset

        def getPos(self):
            return BOUNDING_BOX_START_POS[0] + self.offset[0],\
                   BOUNDING_BOX_START_POS[1] + self.offset[1]

    def __init__(self, root, mode='train'):
        self.root = root
        self.mode = mode
        self.startDateOfTrain = Date(START_OF_TRAINING_YEAR, 5, 1)
        self.endDateOfTrain = Date(END_OF_TRAINING_YEAR, 11, 8)
        self.trainingDateRange = DateRange(self.startDateOfTrain, self.endDateOfTrain)
        self.startDateOfTest = Date(START_OF_TESTING_YEAR, 5, 1)
        self.endDateOfTest = Date(END_OF_TESTING_YEAR, 11, 8)
        self.testingDateRange = DateRange(self.startDateOfTest, self.endDateOfTest)
        self.loadedFilenames = self.loadFilename(mode)
        self.sequences = self._sampleTifSequences()

    def loadSequence(self, sequenceIndex: int):
        sequence = self.sequences[sequenceIndex]
        startPos = sequence.getPos()
        tifGroup = list()
        for i in range(SEQUENCE_LENGTH):
            tif = self._loadTif(sequence, i)
            tifGroup.append(self._cropTif(tif, startPos))
        return np.array(tifGroup)

    def loadSingleTif(self, fileIndex: int):
        offset = (0, 0)
        sequence = self.TifSequence(fileIndex, offset)
        tif = self._loadTif(sequence, 0)
        tif = self._cropTif(tif, sequence.getPos())
        return tif

    def loadFilename(self, mode='train'):
        result = list()
        dateRange = self.trainingDateRange if mode == 'train' else self.testingDateRange
        for tif in self._getAllFilenames():
            fileDate = FileDate(tif)
            if dateRange.isValidFileDate(fileDate):
                result.append(tif)
        return result

    def _loadTif(self, sequence, index):
        filename = self.loadedFilenames[sequence.startIndex + index]
        return rasterio.open(self.root + filename).read(1)

    def _cropTif(self, tif, pos):
        return tif[pos[0] : pos[0] + SEQUENCE_LENGTH, pos[1] : pos[1] + SEQUENCE_LENGTH]

    def _sampleTifSequences(self) -> list:
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
            for row in range(RAND_BOUNDING_BOX_RANGE):
                for col in range(RAND_BOUNDING_BOX_RANGE):
                    offset = (row, col)
                    tifSequences.append(self.TifSequence(pos, offset))
        return tifSequences

    def _isGroupFilled(self, group, size):
        return len(group) != 0 and len(group) % size == 0

    def _getAllFilenames(self):
        return sorted(os.listdir(self.root))


class PrecipitationDataset(Dataset):
    def __init__(self, mode="train", root="./data/daily/", trans=None):
        self.mode = mode
        self.numOfBoundingBoxes = NUM_OF_BOUNDING_BOX
        self.randBoundingBoxesRange = RAND_BOUNDING_BOX_RANGE
        self.totalYears = calculateTotalYears(mode)
        self.samplesPerYear = SAMPLES_PER_YEAR
        self.tifs = TifData(root, mode)
        self.root = root
        self.transform = trans
        print(f">>There are {len(self.tifs.loadedFilenames)} tif files.")

    def __len__(self):
        return len(self.tifs.sequences)

    def __getitem__(self, index):
        tifGroup = self.tifs.loadSequence(index)
        if self.transform is not None:
            tifGroup = self.transform(tifGroup)
        return torch.from_numpy(tifGroup)


if __name__ == "__main__":
    # tifData = TifData("./data/daily/", mode='test')
    # combineData = tifData.loadedFilenames
    # print(combineData)
    # dataset = PrecipitationDataset(mode='test')
    # d0 = dataset[0]
    # d1 = dataset[10]
    # d2 = dataset[483]
    # d3 = dataset[4583]
    tifData = TifData('./data/daily/', mode='test')
    group = tifData.loadSingleTif(60)
    print('aa')
