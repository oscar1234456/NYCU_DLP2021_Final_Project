import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latentSize=30):
        super(Encoder, self).__init__()
        self.latentSize = latentSize

        self.convLayer = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=128, kernel_size=(3, 3, 3), stride=(2,2,2)),
            nn.BatchNorm3d(128),
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(3, 3, 3), stride=(2,2,2)),
            nn.BatchNorm3d(128),
        )
        self.linearLayer = nn.Sequential(
            nn.Linear(in_features=128*7*7*7, out_features=500),
            nn.ReLU()
        )
        self.linear_mean = nn.Linear(500, self.latentSize)
        self.linear_logVar = nn.Linear(500, self.latentSize)

    def forward(self, inputs):
        # inputs: N*32*32*32
        out = self.convLayer(inputs.view(-1, 1, 32, 32, 32))  # out: N*128*28*28*28
        out = self.linearLayer(out.view(-1, 128*7*7*7))  # out: N*500
        linearMean = self.linear_mean(out)  # linearMean: N*30
        linearLogVar = self.linear_logVar(out)  # linearLogVar: N*30
        return linearMean, linearLogVar

class Decoder(nn.Module):
    def __init__(self, latentSize=30):
        super(Decoder, self).__init__()
        self.latentSize = latentSize

        self.linearLayer = nn.Linear(in_features=self.latentSize, out_features=256*7*7*7)
        self.convTranspose = nn.Sequential(
            nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=(3, 3, 3), stride=(2, 2, 2)),
            nn.BatchNorm3d(128),
            nn.ConvTranspose3d(in_channels=128, out_channels=128, kernel_size=(3, 3, 3), stride=(2, 2, 2)),
            nn.BatchNorm3d(128),
            nn.ReLU(True),
            nn.ConvTranspose3d(in_channels=128, out_channels=1, kernel_size=(2, 2, 2))
        )
        # self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, latent):
        # latent: N*30
        out = self.linearLayer(latent)  # out:N*131072
        result = self.convTranspose(out.view(-1, 256, 7, 7, 7))  # result: N*1*32*32*32
        # return self.relu(result.view(-1, 32, 32, 32))
        return self.tanh(result.view(-1, 32, 32, 32))


if __name__ == "__main__":
    # testTensor = torch.randn(5,32,32,32)
    # encoder = Encoder()
    # mean, log_var = encoder(testTensor)
    # print(mean)
    # print(log_var)
    testLatent = torch.randn(5,30)
    decoder = Decoder()
    # result
    Final = decoder(testLatent)
