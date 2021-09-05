import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.convLayer = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=128, kernel_size=(3, 3, 3), stride=(2, 2, 2)),
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(3, 3, 3), stride=(2, 2, 2)),
        )
        self.linearLayer = nn.Sequential(
            nn.Linear(in_features=43904, out_features=500),
            nn.ReLU()
        )
        self.linear_mean = nn.Linear(500, 30)
        self.linear_logVar = nn.Linear(500, 30)

    def forward(self, input):
        # input: N*32*32*32
        out = self.convLayer(input.view(-1, 1, 32, 32, 32))  # out: N*128*7*7*7
        out = self.linearLayer(out.view(-1, 128*7*7*7))  # out: N*500
        linearMean = self.linear_mean(out)  # linearMean: N*30
        linearLogVar = self.linear_logVar(out)  # linearLogVar: N*30
        return linearMean, linearLogVar

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.linearLayer = nn.Linear(in_features=30, out_features=256*8*8*8)
        self.convTranspose = nn.Sequential(
            nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=(3, 3, 3), stride=(2, 2, 2)),
            nn.ConvTranspose3d(in_channels=128, out_channels=128, kernel_size=(3, 3, 3), stride=(2, 2, 2)),
            # nn.ConvTranspose3d(in_channels=128, out_channels=1, kernel_size=(4, 4, 4))
        )

    def forward(self, latent):
        # latent: N*30
        out = self.linearLayer(latent)  # out:N*131072
        result = self.convTranspose(out.view(-1, 256, 8, 8, 8))  # result: N*1*32*32*32
        return result


if __name__ == "__main__":
    # testTensor = torch.randn(5,32,32,32)
    # encoder = Encoder()
    # mean, log_var = encoder(testTensor)
    # print(mean)
    # print(log_var)

    testLatent = torch.randn(5,30)
    decoder = Decoder()
    resultFinal = decoder(testLatent)
