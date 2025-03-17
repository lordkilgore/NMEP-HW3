"""ResNet implementation taken from kuangliu on github
link: https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import torch.nn as nn
import torch.nn.functional as F
from torch import flatten


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Create a residual block for our ResNet18 architecture.

        Here is the expected network structure:
        - conv layer with
            out_channels=out_channels, 3x3 kernel, stride=stride
        - batchnorm layer (Batchnorm2D)
        - conv layer with
            out_channels=out_channels, 3x3 kernel, stride=1
        - batchnorm layer (Batchnorm2D)
        - shortcut layer:
            if either the stride is not 1 or the out_channels is not equal to in_channels:
                the shortcut layer is composed of two steps:
                - conv layer with
                    in_channels=in_channels, out_channels=out_channels, 1x1 kernel, stride=stride
                - batchnorm layer (Batchnorm2D)
            else:
                the shortcut layer should be an no-op

        All conv layers will have a padding of 1 and no bias term. To facilitate this, consider using
        the provided conv() helper function.
        When performing a forward pass, the ReLU activation should be applied after the first batchnorm layer
        and after the second batchnorm gets added to the shortcut.
        """
        ## YOUR CODE HERE

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
        x = nn.functional.relu(x)

        for layer in second_segment:
            x = layer(x)

        for layer in shortcut:
            idd = layer(idd)
        
        x = nn.functional.relu(x + idd)

        return x



class ResNet18(nn.Module):
    def __init__(self, num_classes=200):
        num_classes = num_classes
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=64,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_block(out_channels=64, stride=1)
        self.layer2 = self.make_block(out_channels=128, stride=2)
        self.layer3 = self.make_block(out_channels=256, stride=2)
        self.layer4 = self.make_block(out_channels=512, stride=2)
        self.linear = nn.Linear(512, num_classes)

    def make_block(self, out_channels, stride):
        layers = []
        for stride in [stride, 1]:
            layers.append(ResNetBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = flatten(x, 1)
        x = self.linear(x)
        return x
