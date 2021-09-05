import torch
from torch.utils.data import DataLoader
from Trainner import Trainner
import Parameters
from Dataset import PrecipitationDatasetLoader

##
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

##DataLoader
trainData = PrecipitationDatasetLoader(mode="train", root="./data/daily/")
trainLoader = DataLoader(trainData, Parameters.batchSize, shuffle=True, num_workers=4, pin_memory=True)

##Training
# latentSize, lr, beta1, beta2, maxEpoch, batchSize, trainDataLoader, klWeight, device
trainProcess = Trainner(Parameters.latentSize, Parameters.lr, Parameters.beta1, Parameters.beta2,
                        Parameters.maxEpoch, Parameters.batchSize, trainLoader, Parameters.klWeight,
                        device, Parameters.modelSaveRoot)
