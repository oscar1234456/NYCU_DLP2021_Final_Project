import rasterio
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot
from rasterio.plot import show

dataset = rasterio.open("./data/daily/1981-8-15.tif")
# dataset.index(21.781631,-31.384303)
#row, col = dataset.index(20.054185,72.966381) #there is some trouble if we don't use whole world map

# pyplot way:
# pyplot.imshow(dataset.read(1), cmap='YlOrRd')
# pyplot.show()
#
# rasterio plot way:
show(dataset)

#read dataset(and specified "Band")
dataArray = dataset.read(1)

#have some trouble(reason same as above)
#dataArray = dataArray[row,col]

print(dataArray)