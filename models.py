import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        self.convLayer = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=128, kernel_size=(3,3,3), stride=(2,2,2)),
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(3,3,3), stride=(2,2,2)),
            nn.Linear(128, 500),
            nn.ReLU()
        )
        self.linear_mean = nn.Linear(500, 30)
        self.linear_logVar = nn.Linear(500, 30)

    def forward(self, input):
        # input: N*32*32*32
        pass

class Decoder(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass
