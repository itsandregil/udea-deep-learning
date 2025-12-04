import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_in, x_skip):
        x_in = self.up(x_in)

        diffY = x_skip.size()[2] - x_in.size()[2]
        diffX = x_skip.size()[3] - x_in.size()[3]

        x_in = F.pad(
            x_in, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
        )

        x = torch.cat([x_skip, x_in], dim=1)
        return self.conv(x)


class MultiTaskModel(nn.Module):
    def __init__(
        self, num_classification_classes=4, use_pretrained=True, freeze_internals=False
    ):
        super().__init__()

        if use_pretrained:
            backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).features
        else:
            backbone = mobilenet_v2().features

        self.conv1_new = nn.Conv2d(
            1, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        backbone[0] = self.conv1_new

        self.layer1 = backbone[:2]
        self.layer2 = backbone[2:4]
        self.layer3 = backbone[4:7]
        self.layer4 = backbone[7:11]
        self.layer5 = backbone[11:15]
        self.bottleneck = backbone[15:19]

        # Freeze params
        if freeze_internals:
            for param in (
                self.layer1,
                self.layer2,
                self.layer3,
                self.layer4,
                self.layer5,
                self.bottleneck,
            ):
                param.requires_grad_(False)

        classifier_in_features = 1280
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(classifier_in_features, num_classification_classes),
        )

        self.up_conv1 = UpConv(1280 + 96, 128)
        self.up_conv2 = UpConv(128 + 64, 64)
        self.up_conv3 = UpConv(64 + 32, 32)
        self.up_conv4 = UpConv(32 + 24, 16)

        self.seg_out_conv = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x_bottleneck = self.bottleneck(x5)

        classification_output = self.classification_head(x_bottleneck)

        d1 = self.up_conv1(x_bottleneck, x5)
        d2 = self.up_conv2(d1, x4)
        d3 = self.up_conv3(d2, x3)
        d4 = self.up_conv4(d3, x2)

        segmentation_output = self.seg_out_conv(d4)

        return classification_output, segmentation_output


if __name__ == "__main__":
    model = MultiTaskModel()
    num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"Multi-Task Model Params: {num_params}")
