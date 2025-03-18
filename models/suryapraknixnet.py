import torch.nn as nn
import torch.nn.functional as F
import torch


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        conv = lambda in_channels, out_channels, kernel_size, stride: nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1, bias=False)
        self.layers = nn.ModuleList([conv(in_channels, out_channels, 3, stride),
                                     nn.BatchNorm2d(out_channels),
                                     conv(out_channels, out_channels, 3, 1),
                                     nn.BatchNorm2d(out_channels)])

        if stride != 1 or out_channels != in_channels:
            self.layers.append(conv(in_channels, out_channels, 3, stride))
            self.layers.append(nn.BatchNorm2d(out_channels))
        else:
            self.layers.append(nn.Sequential())

        ## END YOUR CODE

    def forward(self, x):
        """
        Compute a forward pass of this batch of data on this residual block.

        x: batch of images of shape (batch_size, num_channels, width, height)
        returns: result of passing x through this block
        """
        ## YOUR CODE HERE
        idd = x.clone().detach()
        first_segment = self.layers[:2]
        second_segment = self.layers[2:4]
        shortcut = self.layers[4:]

        for layer in first_segment:
            x = layer(x)
        x = nn.functional.gelu(x)

        for layer in second_segment:
            x = layer(x)

        for layer in shortcut:
            idd = layer(idd)
        
        x = nn.functional.gelu(x + idd)

        return x

class SuryaPrakNixNet(nn.Module):
    def __init__(self, num_classes):
        """
           Deeper head, wider convolutional layers, 
           and even a deconvolution -- 
           what more could you want?
        """

        super(SuryaPrakNixNet, self).__init__()
        self.in_channels = 64   
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),  
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),  
            nn.BatchNorm2d(64),
            nn.GELU(),
            self.make_block(out_channels=64, stride=1, rep=2),
            self.make_block(out_channels=128, stride=2, rep=2),
            self.make_block(out_channels=512, stride=2, rep=2),
            self.make_block(out_channels=512, stride=2, rep=2),
            self.make_block(out_channels=1024, stride=2, rep=2),
            self.make_block(out_channels=1024, stride=2, rep=2),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifer = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )
        
        

    def make_block(self, out_channels, stride, rep):
        layers = []
        for stride in [stride, 1]:
            layers.append(ResNetBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifer(x)
        return x