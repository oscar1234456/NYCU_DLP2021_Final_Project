##
import rasterio
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot
from rasterio.plot import show

##
dataset = rasterio.open("./data/daily/20100815.tif")
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
dataArray = dataArray[4: 36]
dataArray = dataArray[:, 17:49]

#have some trouble(reason same as above)
#dataArray = dataArray[row,col]

print(dataArray)

##
table = pd.DataFrame(0, index=np.arange(1, 14853), columns=['Date', 'Rainfall(mm)'])
i = 0
for files in os.listdir(r'./data/daily/'):
    if files[-4:] == '.tif':
        i = i + 1
        dataset = rasterio.open(r'./data/daily/'+files)
        data_array = dataset.read(1)

        table['Date'].loc[i] = files[:-4]

        table['Rainfall(mm)'].loc[i] = int(data_array[32, 19])
