import numpy as np
import torch

import Parameters
from Models import Encoder, Decoder
from Sampler import Sampler
from Transform import ShiftNormalize
import matplotlib.pyplot as plt

class VAE:
    def __init__(self, latentSize, device, lr=0.0001, beta1=0.9, beta2=0.99):
        self.latentSize = latentSize
        self.device = device

        self._encoder = Encoder(latentSize).to(device)
        self._decoder = Decoder(latentSize).to(device)

        self._encoderOptimizer = torch.optim.Adam(self._encoder.parameters(), lr=lr,
                                                   betas=(beta1, beta2))
        self._decoderOptimizer = torch.optim.Adam(self._decoder.parameters(), lr=lr,
                                                   betas=(beta2, beta2))

    def reset(self):
        self._encoder.train()
        self._decoder.train()
        self._encoderOptimizer.zero_grad()
        self._decoderOptimizer.zero_grad()

    def forward(self, inputs):
        mean, variance = self._encoder(inputs)
        latentCode = self._reparameterTrick(mean, variance)
        return self._decoder(latentCode), mean, variance

    def update(self):
        self._encoderOptimizer.step()
        self._decoderOptimizer.step()

    def save(self, root, final=False):
        if final:
            torch.save(self._encoder.state_dict(), root + 'final_encoder.pth')
            torch.save(self._decoder.state_dict(), root + 'final_decoder.pth')
            print("Final Model Save!")
        else:
            torch.save(self._encoder.state_dict(), root + 'encoder.pth')
            torch.save(self._decoder.state_dict(), root + 'decoder.pth')
            print("Model Save!")

    def load(self, path):
        self._encoder.load_state_dict(torch.load(path + 'final_encoder.pth'))
        self._decoder.load_state_dict(torch.load(path + 'final_decoder.pth'))
        print("Model Load Complete!")
        print(f"Encoder:{self._encoder}")
        print(f"Decoder:{self._decoder}")

    def _reparameterTrick(self, mean, logVar):
        std = torch.exp(logVar * 0.5)
        esp = torch.randn_like(std)
        return mean + std * esp

    def useDecoder(self, sampleDistribution, block):
        # self.load(path)
        latentCode = self._generateLatentCode(sampleDistribution, block)
        # latentCode = torch.Tensor(np.random.normal(0, 1, 30))
        latentCode = latentCode.to(self.device)
        decoderOutput = self._decoder(latentCode) # decoderOutput: 32*32*32
        return decoderOutput


    def _generateLatentCode(self, sampleDistribution, block):
        sampler = Sampler(sampleAmount=1, sampleSize=self.latentSize,
                          sampleDistribution=sampleDistribution,
                          block=block)
        latentCode = sampler.sample()
        return latentCode



if __name__ == "__main__":
    # latentSize = 30
    # device = "cuda"
    # model = VAE(latentSize, device)
    # output = model.useDecoder("./modelWeight/", sampleDistribution="Normal",  block="Rare+")
    # normalization = ShiftNormalize(Parameters.maxPrecipitation)
    # denormalOutput = normalization.deNormalize(output).cpu().detach().numpy()
    # plt.imshow(denormalOutput[0][1], cmap='YlGnBu')
    # plt.show()
    print("")




