# the baseblock of ssdu
import torch.nn as nn
class RB(nn.Module):
    def __init__(self, in_channel=64, out_channel=64):
        super(RB, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):

        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1

        return out + x

class Resnet(nn.Module):
    def __init__(self, in_channels=2, hidden_channel=64, out_channel=2, layers=15):
        super(Resnet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, hidden_channel, kernel_size=3, padding=1)

        self.blk = nn.Sequential()

        for i in range(1, layers+1):
            self.blk.add_module('RB{}'.format(i), RB(hidden_channel, hidden_channel))

        self.conv2 = nn.Conv2d(hidden_channel, hidden_channel, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(hidden_channel, out_channel, kernel_size=3, padding=1)

    def forward(self, x):

        out0 = self.conv1(x)

        out = self.blk(out0)

        out = self.conv2(out) + out0

        out = self.conv3(out)
    
        return out
    