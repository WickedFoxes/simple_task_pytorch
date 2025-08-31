import torch
import torch.nn as nn


class CustomNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomNet, self).__init__()
        self.inplanes = 64  # base channel width

        self.features = nn.Sequential(
            nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.inplanes, self.inplanes*2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.inplanes*2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(self.inplanes*2, self.inplanes*4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.inplanes*4),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.inplanes*4, self.inplanes*4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.inplanes*4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.inplanes*4, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
