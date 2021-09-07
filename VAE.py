import torch
from Models import Encoder, Decoder

class VAE:
    def __init__(self, latentSize, device, lr=0.0001, beta1=0.9, beta2=0.99):
        self.latentSize = latentSize
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
        self._encoder = torch.load(path + 'encoder.pth')
        self._decoder = torch.load(path + 'decoder.pth')

    def _reparameterTrick(self, mean, logVar):
        std = torch.exp(logVar * 0.5)
        esp = torch.randn_like(std)
        return mean + std * esp