##
import rasterio
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
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

## Find maximum
maxPrecipitation = 0
minPrecipitation = 10000
for files in os.listdir(r'./data/daily/'):
    print(f">>Running... Comparing:{files}")
    dataset = rasterio.open(r'./data/daily/'+files)
    tif = dataset.read(1)
    scope = tif[:32, 17:49]
    tempMax = np.max(scope)
    tempMin = np.min(scope)
    if tempMax > maxPrecipitation:
        maxPrecipitation = tempMax
    if tempMin < minPrecipitation:
        minPrecipitation = tempMin
print("End of Comparing")
print(f"Max: {maxPrecipitation}")
print(f"Min: {minPrecipitation}")

## QQ Plot platting way
x1 = np.random.normal(1,2,1000)
x2 = np.random.normal(1,2,1000)

x1.sort()
x2.sort()

plt.scatter(x1,x2)
plt.plot([min(x1),max(x1)],[min(x1),max(x1)],color="red")
plt.xlabel("1st dataset's quantiles")
plt.ylabel("2nd dataset's quantiles")
plt.show()


