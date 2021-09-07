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

        # DataLoader #
        self.trainDataLoader = trainDataLoader

        # Model #
        self.vaeModel = VAE(latentSize, device, klWeight, lr, beta1, beta2)
        # # Models #
        # self._decoder = Decoder(self.latentSize).to(self.device)
        # self._encoder = Encoder(self.latentSize).to(self.device)
        # # self._decoder = Decoder(self.latentSize).to(self.device)
        #
        # # Optimizer #
        # self._encoder_optimizer = torch.optim.Adam(self._encoder.parameters(), lr=self.lr,
        #                                            betas=(self.beta1, self.beta2))
        # self._decoder_optimizer = torch.optim.Adam(self._decoder.parameters(), lr=self.lr,
        #                                            betas=(self.beta1, self.beta2))


    def train(self):
        minLoss = 9999
        print(">>>Training Process Start<<<")
        for epoch in range(self.maxEpoch):
            print('Epoch [{}/{}]'.format(epoch, self.maxEpoch - 1))
            print('-' * 12)
            nowLoss = 0
            for batch, inputs in enumerate(self.trainDataLoader):
                inputs = inputs.to(self.device)  # N*32*32*32
                self.vaeModel.reset()
                decoderOutputs, encoderMean, encoderLogVar = self.vaeModel.forward(inputs)
                loss, reconstructionLoss, klLoss = self._calculateLoss(
                    inputs, decoderOutputs, encoderLogVar, encoderMean, self.klWeight)
                loss.backward()
                self.vaeModel.update()
                # TODO: Need to add the KL annealing. Survey the method what paper used to annealing weight first

                nowLoss += loss.item() * inputs.size(0)

                if batch % 10 == 0:
                    print(f'>>Batch [{batch}] ReconstructLoss:{reconstructionLoss.item()} KLLoss:{klLoss.item()} Loss:{loss.item()}')

            size = len(self.trainDataLoader.dataset)
            epoch_loss = nowLoss / size
            print('Epoch {} Loss: {:.4f}'.format(epoch, epoch_loss))
            print()

            # TODO: discuss about when to stop the training
            if epoch_loss < minLoss:
                minLoss = epoch_loss
                self.modelWeightSaver()

        print(">>>Training Process End<<<")
        print(f"The minimal loss: {minLoss}")

    def modelWeightSaver(self):
        self.vaeModel.save(self.root)

    # def _vaeForward(self, inputs):
    #     mean, variance = self._encoder(inputs)
    #     latentCode = self._reparameterTrick(mean, variance)
    #     return self._decoder(latentCode), mean, variance
    #
    # def _updateModelParameter(self):
    #     self._encoder_optimizer.step()
    #     self._decoder_optimizer.step()
    #
    # def _resetModelAndOptimizer(self):
    #     self._encoder.train()
    #     self._decoder.train()
    #     self._encoder_optimizer.zero_grad()
    #     self._decoder_optimizer.zero_grad()
    #
    # def _reparameterTrick(self, mean, logVar):
    #     std = torch.exp(logVar * 0.5)
    #     esp = torch.randn_like(std)
    #     return mean + std * esp

    def _KLDLoss(self, logVar, mean):
        return -0.5*(torch.sum(1+logVar-mean.pow(2)-logVar.exp()))

    def _calculateLoss(self, inputs, decoderOutputs, logVar, mean, klWeight=0):
        criterion = torch.nn.MSELoss()
        reconstructionLoss = criterion(decoderOutputs, inputs)
        klLoss = self._KLDLoss(logVar, mean)
        return reconstructionLoss + (klWeight * klLoss), reconstructionLoss, klLoss
