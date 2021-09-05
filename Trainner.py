import torch
from Models import Encoder, Decoder


class Trainner():
    def __init__(self, latentSize, lr, beta1, beta2, maxEpoch, batchSize,
                 trainDataLoader, klWeight, device, modelSaveRoot):
        # Config #
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.trainDataLoader = trainDataLoader
        self.maxEpoch = maxEpoch
        self.batchSize = batchSize
        self.device = device
        self.latentSize = latentSize
        self.klWeight = klWeight
        self.root = modelSaveRoot

        # Optimizer #
        self._encoder_optimizer = torch.optim.Adam(self._encoder.parameters(), lr=self.lr,
                                                   betas=(self.beta1, self.beta2))
        self._decoder_optimizer = torch.optim.Adam(self._decoder.parameters(), lr=self.lr,
                                                   betas=(self.beta1, self.beta2))

        # DataLoader #
        self.trainDataLoader = trainDataLoader

        # Models #
        self._encoder = Encoder(self.latentSize).to(self.device)
        self._decoder = Decoder(self.latentSize).to(self.device)

    def train(self):
        minLoss = 9999
        print(">>>Training Process Start<<<")
        for epoch in self.maxEpoch:
            print('Epoch [{}/{}]'.format(epoch, self.maxEpoch - 1))
            print('-' * 12)
            nowLoss = 0
            for batch, inputs in enumerate(self.trainDataLoader):
                self._encoder.train()
                self._decoder.train()
                self._encoder_optimizer.zero_grad()
                self._decoder_optimizer.zero_grad()

                inputs = inputs.to(self.device)  #N*32*32*32

                encoderMean, encoderLogVar = self._encoder(inputs)

                latentCode = self._reparameterTrick(encoderMean, encoderLogVar)

                decoderOutputs = self._decoder(latentCode)

                loss, reconstructionLoss, klLoss = self._loss(inputs, decoderOutputs, encoderLogVar, encoderMean, self.klWeight)
                loss.backward()

                self._encoder_optimizer.step()
                self._decoder_optimizer.step()

                nowLoss += loss.item() * inputs.size(0)

                if batch % 4 == 0:
                    print(f'>>Batch [{batch}] ReconstructLoss:{reconstructionLoss.item()} KLLoss:{klLoss.item()} Loss:{loss.item()}')

            size = len(self.trainDataLoader.dataset)
            epoch_loss = nowLoss / size
            print('Epoch {} Loss: {:.4f}'.format(epoch, epoch_loss))
            print()

            if epoch_loss < minLoss:
                minLoss = epoch_loss
                self.modelWeightSaver()

        print(">>>Training Process End<<<")

    def modelWeightSaver(self):
        torch.save(self._encoder.state_dict(), self.root + '/encoder.pth')
        torch.save(self._decoder.state_dict(), self.root + '/decoder.pth')

    def _reparameterTrick(self, mean, logVar):
        std = torch.exp(logVar * 0.5)
        esp = torch.randn_like(std)
        return mean + std * esp

    def _KLDLoss(self, logVar, mean):
        return -0.5*(torch.sum(1+logVar-mean.pow(2)-logVar.exp()))

    def _loss(self, inputs, decoderOutputs, logVar, mean, klWeight=0):
        criterion = torch.nn.MSELoss()
        reconstructionLoss = criterion(decoderOutputs, inputs)
        klLoss = self._KLDLoss(logVar, mean)
        return reconstructionLoss + (klWeight * klLoss), reconstructionLoss, klLoss
