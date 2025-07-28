import torch
import torch.nn as nn
import torch.nn.functional as F



# --- U-Net ---
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[32, 64]):
        super(UNet, self).__init__()

        # Encoder blocks
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for feat in features:
            self.downs.append(self._double_conv(in_channels, feat))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = feat

        # Bottleneck
        self.bottleneck = self._double_conv(features[-1], features[-1]*2)

        # Decoder blocks
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for feat in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feat*2, feat, kernel_size=2, stride=2))
            self.up_convs.append(self._double_conv(feat*2, feat))

        # Final output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def _double_conv(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        skip_connections = []

        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skip_connections.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for up, conv, skip in zip(self.ups, self.up_convs, reversed(skip_connections)):
            x = up(x)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])  # fix mismatch due to odd dims
            x = torch.cat((skip, x), dim=1)
            x = conv(x)

        return self.final_conv(x)


# --- DnCNN ---
class DnCNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, depth=17, features=64):
        super(DnCNN, self).__init__()
        layers = []

        # First layer (no batch norm)
        layers.append(nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))

        # Middle layers
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))

        # Last layer (no activation)
        layers.append(nn.Conv2d(features, out_channels, kernel_size=3, padding=1, bias=True))

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        residual = self.dncnn(x)
        return x - residual  # subtract the predicted noise/blur


# --- Model Selector ---
def get_model(name: str, **kwargs):
    name = name.lower()
    if name == "unet":
        return UNet(**kwargs)
    elif name == "dncnn":
        return DnCNN(**kwargs)
    else:
        raise ValueError(f"Unsupported model: {name}")