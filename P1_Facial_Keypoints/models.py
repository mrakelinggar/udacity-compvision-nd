## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 3)  # 32x111x111
        self.conv2 = nn.Conv2d(32, 128, 3)  # 128x54x54
        self.conv3 = nn.Conv2d(128, 256, 3)  # 256x26x26
        self.conv4 = nn.Conv2d(256, 512, 3)  # 512x12x12
        self.conv5 = nn.Conv2d(512, 256, 5)  # 256x4x4
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.pool = nn.MaxPool2d(2, 2)
        # self.dp1 = nn.Dropout(p=0.2)
        # self.dp2 = nn.Dropout(p=0.3)
        # self.dp3 = nn.Dropout(p=0.4)
        # self.dp4 = nn.Dropout(p=0.3)
        self.c_bn1 = nn.BatchNorm2d(32)
        self.c_bn2 = nn.BatchNorm2d(128)
        self.c_bn3 = nn.BatchNorm2d(256)
        self.c_bn4 = nn.BatchNorm2d(512)
        self.c_bn5 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256*4*4, 2560)
        # self.dp4 = nn.Dropout(p=0.4)
        self.fc_bn1 = nn.BatchNorm1d(2560)

        self.fc2 = nn.Linear(2560, 1024)
        # self.dp5 = nn.Dropout(p=0.3)
        self.fc_bn2 = nn.BatchNorm1d(1024)

        self.fc3 = nn.Linear(1024, 136)

    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        # x = self.dp1(self.pool(F.leaky_relu(self.conv1(x))))
        # x = self.dp2(self.pool(F.leaky_relu(self.conv2(x))))
        # x = self.dp3(self.pool(F.leaky_relu(self.conv3(x))))
        # x = self.dp4(self.pool(F.leaky_relu(self.conv4(x))))
        x = self.c_bn1(self.pool(F.leaky_relu(self.conv1(x))))
        x = self.c_bn2(self.pool(F.leaky_relu(self.conv2(x))))
        x = self.c_bn3(self.pool(F.leaky_relu(self.conv3(x))))
        x = self.c_bn4(self.pool(F.leaky_relu(self.conv4(x))))
        x = self.c_bn5(self.pool(F.leaky_relu(self.conv5(x))))
        x = x.view(x.size(0), -1)
        # x = self.dp4(F.leaky_relu(self.fc1(x)))
        # x = self.dp5(F.leaky_relu(self.fc2(x)))
        x = self.fc_bn1(F.leaky_relu(self.fc1(x)))
        x = self.fc_bn2(F.leaky_relu(self.fc2(x)))
        x = self.fc3(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x
