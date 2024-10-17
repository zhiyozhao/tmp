import torch
import torch.nn as nn
from torchvision import models


class PortraitNet(nn.Module):
    def __init__(self, backbone_type="resnet50", num_classes=1):
        super(PortraitNet, self).__init__()

        # Load ResNet backbone
        if backbone_type == "resnet34":
            self.backbone = models.resnet34()
            filters = [64, 64, 128, 256, 512]
        elif backbone_type == "resnet50":
            self.backbone = models.resnet50()
            filters = [64, 256, 512, 1024, 2048]
        elif backbone_type == "resnet101":
            self.backbone = models.resnet101()
            filters = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(
                "Unsupported backbone: choose from resnet34, resnet50, resnet101"
            )

        # Encoder from ResNet backbone
        self.encoder1 = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
        )
        self.encoder2 = nn.Sequential(
            self.backbone.maxpool,
            self.backbone.layer1,
        )
        self.encoder3 = self.backbone.layer2
        self.encoder4 = self.backbone.layer3
        self.encoder5 = self.backbone.layer4

        # Decoder part with bilinear upsampling and U-Net skip connections
        self.upsample4 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.decoder4 = self._decoder_block(filters[4] + filters[3], filters[3])

        self.upsample3 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.decoder3 = self._decoder_block(filters[3] + filters[2], filters[2])

        self.upsample2 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.decoder2 = self._decoder_block(filters[2] + filters[1], filters[1])

        self.upsample1 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.decoder1 = self._decoder_block(filters[1] + filters[0], filters[0])

        # Final segmentation output
        self.final_upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.final_conv = nn.Conv2d(filters[0], num_classes, kernel_size=1)

    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        d4 = self.upsample4(e5)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.decoder4(d4)

        d3 = self.upsample3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upsample2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upsample1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.decoder1(d1)

        d0 = self.final_upsample(d1)
        d0 = self.final_conv(d0)

        return d0


if __name__ == "__main__":
    model = PortraitNet(backbone_type="resnet34", num_classes=1)
    dummy_input = torch.randn(3, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
