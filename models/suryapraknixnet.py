import torch.nn as nn
import torch.nn.functional as F
import torch


class BRBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BRBlock, self).__init__()
        """Best practice is to just list out layers explicitly."""
        expansion = 4  # how much input/output channels differ by
        mid_channels = out_channels // expansion

        # Reduce
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        # Process
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        # Expand 
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Identity()
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.activation = nn.ReLU()
        ## END YOUR CODE

    def forward(self, x):
        """
        Compute a forward pass of this batch of data on this residual block.

        x: batch of images of shape (batch_size, num_channels, width, height)
        returns: result of passing x through this block
        """
        ## YOUR CODE HERE
        idd = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x = self.activation(x + idd)
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
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),  
            nn.BatchNorm2d(64),
            nn.GELU(),
            self.make_block(out_channels=64, stride=1, rep=2),
            self.make_block(out_channels=128, stride=2, rep=3),
            self.make_block(out_channels=512, stride=2, rep=4),
            self.make_block(out_channels=1024, stride=2, rep=6),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifer = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 52),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        

    def make_block(self, out_channels, stride, rep):
        layers = []
        for stride in [stride, 1]:
            layers.append(BRBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifer(x)
        return x