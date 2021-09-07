import random

import numpy as np
from matplotlib import pyplot as plt

from Dataset import TifData
from VAE import VAE
from Parameters import *


class Tester:
    def __init__(self, weightPath, device):
        self.model = VAE(latentSize, device)
        self.model.load(weightPath)
        self.tifs = TifData('./data/daily/', 'test')
        self.quantiles = 150

    def qqPlot(self):
        means = self._getRandTifGroupMean(1500)
        means = np.sort(means)
        _, highPrecipitation = self._pickPercentileFromArr(means, 10)
        originQuantiles = self._pickQuantile(means)
        highPrecipitationQuantile = self._pickQuantile(highPrecipitation)
        self._showQQPlot(originQuantiles, highPrecipitationQuantile)


    def _showQQPlot(self, x, y):
        plt.scatter(x, y)
        plt.plot([min(x), max(x)], [min(x), max(x)], color='red')
        plt.xlabel('quantile for real data')
        plt.ylabel('quantile for synthetic data')
        plt.show()

    def _pickQuantile(self, npArray: np.ndarray):
        stepSize = 1. / self.quantiles
        quantiles = list()
        for i in range(1, self.quantiles + 1):
            quantiles.append(np.quantile(npArray, stepSize * i))
        return np.array(quantiles)

    def _getRandTifGroupMean(self, numOfTif):
        size = len(self.tifs.loadedFilenames) - 32
        fileIndex = random.sample(range(size), k=numOfTif)
        means = list()
        for index in fileIndex:
            group = self.tifs.loadSingleTif(index)
            means.append(np.mean(group))
        return np.array(means)

    def _pickPercentileFromArr(self, npArray, percent):
        size = len(npArray) // percent
        return npArray[:size], npArray[len(npArray) - size: len(npArray)]


if __name__ == "__main__":
    tester = Tester('./modelWeight/', 'cuda')
    tester.qqPlot()
    # means = tester._getRandTifGroupMean(1500)
    # means = np.sort(means)
    # low, high = tester._pickPercentileFromArr(means, 10)
    # quantiles = tester._pickQuantile(means)
    # print(means)