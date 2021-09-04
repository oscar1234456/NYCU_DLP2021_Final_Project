import rasterio
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot
from rasterio.plot import show

dataset = rasterio.open("./data/chirps-v2.0.1981.08.tif")
dataset.index(21.781631,-31.384303)
pyplot.imshow(dataset.read(1), cmap='YlOrRd')
pyplot.show()
# show(dataset)

dataArray = dataset.read(1)
# pyplot.hist2d(dataArray)
# pyplot.show()
# dataArray = dataArray[1627,4035]
# dataArray *= (255.0/dataArray.max())