import torch
from VAE import VAE


class Trainner():
    def __init__(self, latentSize, lr, beta1, beta2, maxEpoch, batchSize,
                 trainDataLoader, klWeight, device, modelSaveRoot):
        # Config #
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.maxEpoch = maxEpoch
        self.batchSize = batchSize
        self.device = device
        self.latentSize = latentSize
        self.klWeight = klWeight
        self.root = modelSaveRoot
        self.warmup = 10  # For KLWeight test

        # DataLoader #
        self.trainDataLoader = trainDataLoader

        # Model #
        self.vaeModel = VAE(latentSize, device, lr, beta1, beta2)


    def train(self):
        minLoss = 9999
        print(">>>Training Process Start<<<")
        for epoch in range(self.maxEpoch):
            print('Epoch [{}/{}]'.format(epoch, self.maxEpoch - 1))
            print('-' * 12)
            nowLoss = 0

            klWeight = self._setupKlWeight(epoch)

            for batch, inputs in enumerate(self.trainDataLoader):
                inputs = inputs.to(self.device)  # N*32*32*32
                self.vaeModel.reset()
                decoderOutputs, encoderMean, encoderLogVar = self.vaeModel.forward(inputs)
                loss, reconstructionLoss, klLoss = self._calculateLoss(
                    inputs, decoderOutputs, encoderLogVar, encoderMean, klWeight)
                loss.backward()
                self.vaeModel.update()

                nowLoss += loss.item() * inputs.size(0)

                if batch % 10 == 0:
                    print(f'>>Batch [{batch}] ReconstructLoss:{reconstructionLoss.item()} KLLoss:{klLoss.item()} Loss:{loss.item()}')

            size = len(self.trainDataLoader.dataset)
            epoch_loss = nowLoss / size
            print('Epoch {} Loss: {:.4f}'.format(epoch, epoch_loss))
            print()

            # TODO: discuss about when to stop the training
            if epoch > 10 and epoch_loss < minLoss:
                minLoss = epoch_loss
                self.modelWeightSaver()

        print(">>>Training Process End<<<")
        print(f"The minimal loss: {minLoss}")
        self.modelWeightSaver(final=True)

    def modelWeightSaver(self, final=False):
        self.vaeModel.save(self.root, final)

    def _setupKlWeight(self, epoch):
        # TODO: Can use more complicated one
        return 1 if epoch > 5 else 0
        # return 1

    def _KLDLoss(self, logVar, mean):
        return -0.5*(torch.sum(1+logVar-mean.pow(2)-logVar.exp()))

    def _calculateLoss(self, inputs, decoderOutputs, logVar, mean, klWeight=0):
        criterion = torch.nn.MSELoss()
        reconstructionLoss = criterion(decoderOutputs, inputs)
        klLoss = self._KLDLoss(logVar, mean)
        return reconstructionLoss + (klWeight * klLoss), reconstructionLoss, klLoss
