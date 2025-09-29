import torch
import torch.nn as nn
import torch.nn.functional as F

# A compact W-shaped segmentation network (encoder-decoder-encoder-decoder)
# This is a lightweight, clean implementation suitable for camera-ready reproduction.
# It is not tied to nnU-Net; you can extend blocks to include Mamba etc.

def conv_block(in_ch, out_ch, k=3, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, k, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = conv_block(in_ch, out_ch)
    def forward(self, x):
        return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)
        self.conv = conv_block(in_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        # pad if needed
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class WNet2D(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, base_ch=32):
        super().__init__()
        # First U
        self.inc = conv_block(in_channels, base_ch)
        self.down1 = Down(base_ch, base_ch*2)
        self.down2 = Down(base_ch*2, base_ch*4)
        self.down3 = Down(base_ch*4, base_ch*8)
        self.bot = conv_block(base_ch*8, base_ch*16)

        self.up3 = Up(base_ch*16, base_ch*8)
        self.up2 = Up(base_ch*8, base_ch*4)
        self.up1 = Up(base_ch*4, base_ch*2)
        self.up0 = Up(base_ch*2, base_ch)

        # Second U (refinement)
        self.down1b = Down(base_ch, base_ch*2)
        self.down2b = Down(base_ch*2, base_ch*4)
        self.down3b = Down(base_ch*4, base_ch*8)
        self.botb = conv_block(base_ch*8, base_ch*16)

        self.up3b = Up(base_ch*16, base_ch*8)
        self.up2b = Up(base_ch*8, base_ch*4)
        self.up1b = Up(base_ch*4, base_ch*2)
        self.up0b = Up(base_ch*2, base_ch)

        self.head = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x):
        # U1
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        xb = self.bot(x4)

        u3 = self.up3(xb, x4)
        u2 = self.up2(u3, x3)
        u1 = self.up1(u2, x2)
        u0 = self.up0(u1, x1)

        # U2 (refine)
        y2 = self.down1b(u0)
        y3 = self.down2b(y2)
        y4 = self.down3b(y3)
        yb = self.botb(y4)

        v3 = self.up3b(yb, y4)
        v2 = self.up2b(v3, y3)
        v1 = self.up1b(v2, y2)
        v0 = self.up0b(v1, u0)

        out = self.head(v0)
        return out

def build_model(name="wnet2d", in_channels=3, num_classes=2):
    name = name.lower()
    if name == "wnet2d":
        return WNet2D(in_channels=in_channels, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")
