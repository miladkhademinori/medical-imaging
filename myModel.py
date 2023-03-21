import torch.nn as nn
import torch

# Model architecture: U-Net is the deep structure that you will use for the segmentation
# task. You can develop your own version, or modify different variations that already
# provided in the challenge website as baselines:
# https://github.com/DIAGNijmegen/picai_baseline
# Whichever model you choose to use, provide details about the structure in your report
# and include the model summary. Consider at least four levels of contraction and
# expansion for your model. Note that if you take code from public repositories, the input
# size needs to be adjusted to match the size of your images. Also, ensure you have
# properly cited/commented the source of your architecture.

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Contraction path
        self.conv1_1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)

        # Expansion path
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv3_3 = nn.Conv2d(512, 256, 3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, 3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2_3 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv2_4 = nn.Conv2d(128, 128, 3, padding=1)

        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv1_3 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv1_4 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv_out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        # Contraction path
        x1 = nn.functional.relu(self.conv1_1(x))
        x1 = nn.functional.relu(self.conv1_2(x1))
        x1p = self.pool1(x1)

        x2 = nn.functional.relu(self.conv2_1(x1p))
        x2 = nn.functional.relu(self.conv2_2(x2))
        x2p = self.pool2(x2)

        x3 = nn.functional.relu(self.conv3_1(x2p))
        x3 = nn.functional.relu(self.conv3_2(x3))
        x3p = self.pool3(x3)

        x4 = nn.functional.relu(self.conv4_1(x3p))
        x4 = nn.functional.relu(self.conv4_2(x4))

        # Expansion path
        x3u = self.upconv3(x4)
        x3c = torch.cat((x3, x3u), dim=1)
        x3 = nn.functional.relu(self.conv3_3(x3c))
        x3 = nn.functional.relu(self.conv3_4(x3))

        x2u = self.upconv2(x3)
        x2c = torch.cat((x2, x2u), dim=1)
        x2 = nn.functional.relu(self.conv2_3(x2c))
        x2 = nn.functional.relu(self.conv2_4(x2))

        x1u = self.upconv1(x2)
        x1c = torch.cat((x1, x1u), dim=1)
        x1 = nn.functional.relu(self.conv1_3(x1c))
        x1 = nn.functional.relu(self.conv1_4(x1))

        # Output
        output = self.conv_out(x1)
        return output

