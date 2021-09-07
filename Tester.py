import random

import numpy as np

from Dataset import TifData
from VAE import VAE
from Parameters import *


class Tester:
    def __init__(self, weightPath, device):
        self.model = VAE(latentSize, device)
        self.model.load(weightPath)
        self.tifs = TifData('./data/daily/', 'test')
        self.quantiles = 500

    def qqPlot(self):
        randTifs = self._randTif(1000)
        tifMeans = self._calculatePrecipitationMean(randTifs)
        tifMeans = np.sort(tifMeans)
        lowPrecipitation, highPrecipitation = self._pickPercentileFromArr(tifMeans, 10)



    def _randTif(self, numOfTif):
        size = len(self.tifs.sequences)
        sequenceIndexes = random.sample(range(size), k=numOfTif)
        randTifs = None
        for index in sequenceIndexes:
            if randTifs is None:
                randTifs = self.tifs.loadSequence(index)
            else:
                randTifs = np.concatenate((randTifs, self.tifs.loadSequence(index)))
        return randTifs

    def _calculatePrecipitationMean(self, tifs):
        means = list()
        for tif in tifs:
            means.append(np.mean(tif))
        return np.array(means)

    def _pickPercentileFromArr(self, npArray, percent):
        size = len(npArray) // percent
        return npArray[:size], npArray[len(npArray) - size: len(npArray)]


if __name__ == "__main__":
    tester = Tester('./modelWeight/', 'cuda')
    randTifs = tester._randTif(1000)
    means = tester._calculatePrecipitationMean(randTifs)
    means = np.sort(means)
    low, high = tester._pickPercentileFromArr(means, 10)
    print(randTifs)