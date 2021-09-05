import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.convLayer = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=128, kernel_size=(3,3,3), stride=(2,2,2)),
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(3,3,3), stride=(2,2,2)),
        )
        self.linearLayer = nn.Sequential(
            nn.Linear(in_features=43904, out_features=500),
            nn.ReLU()
        )
        self.linear_mean = nn.Linear(500, 30)
        self.linear_logVar = nn.Linear(500, 30)

    def forward(self, input):
        # input: N*32*32*32
        out = self.convLayer(input.view(-1, 1, 32, 32, 32)) #out: N*128*
        out = self.linearLayer(out.view(-1,128*7*7*7))
        linearMean = self.linear_mean(out)
        linearLogVar = self.linear_logVar(out)
        return linearMean, linearLogVar

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        pass

    def forward(self):
        pass



if __name__ == "__main__":
    testTensor = torch.randn(5,32,32,32)
    encoder = Encoder()
    mean, log_var = encoder(testTensor)
    print(mean)
    print(log_var)