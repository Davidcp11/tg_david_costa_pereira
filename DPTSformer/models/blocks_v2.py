import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class Read_ignore(nn.Module):
    def __init__(self, start_index=1):
        super(Read_ignore, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        return x[:, self.start_index :]


class Read_add(nn.Module):
    def __init__(self, start_index=1):
        super(Read_add, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        if self.start_index == 2:
            readout = (x[:, 0] + x[:, 1]) / 2
        else:
            readout = x[:, 0]
        return x[:, self.start_index :] + readout.unsqueeze(1)


class Read_projection(nn.Module):
    def __init__(self, in_features, start_index=1):
        super(Read_projection, self).__init__()
        self.start_index = start_index
        self.project = nn.Sequential(nn.Linear(2 * in_features, in_features), nn.GELU())

    def forward(self, x):
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index :])
        features = torch.cat((x[:, self.start_index :], readout), -1)
        return self.project(features)


class MyConvTranspose2d(nn.Module):
    def __init__(self, conv, output_size):
        super(MyConvTranspose2d, self).__init__()
        self.output_size = output_size
        self.conv = conv

    def forward(self, x):
        x = self.conv(x, output_size=self.output_size)
        return x


class Resample(nn.Module):
    def __init__(self, emb_dim, resample_dim, scale_factor):
        super(Resample, self).__init__()
        assert scale_factor in [
            4,
            8,
            16,
            32,
        ], "scale_factor must be in [0.5, 4, 8, 16, 32]"
        self.conv1 = nn.Conv2d(
            emb_dim, resample_dim, kernel_size=1, stride=1, padding=0
        )
        if scale_factor == 4:
            self.conv2 = nn.ConvTranspose2d(
                resample_dim,
                resample_dim,
                kernel_size=4,
                stride=4,
                padding=0,
                bias=True,
                dilation=1,
                groups=1,
            )
        elif scale_factor == 8:
            self.conv2 = nn.ConvTranspose2d(
                resample_dim,
                resample_dim,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True,
                dilation=1,
                groups=1,
            )
        elif scale_factor == 16:
            self.conv2 = nn.Identity()
        else:
            self.conv2 = nn.Conv2d(
                resample_dim,
                resample_dim,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True,
            )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ReassembleBlock(nn.Module):
    def __init__(
        self,
        emb_dim,
        resample_dim,
        read,
        scale_factor,
    ):
        """
        p = patch size
        s = coefficient resample
        emb_dim <=> D (in the paper)
        resample_dim <=> ^D (in the paper)
        read : {"ignore", "add", "proj"}
        """
        super(ReassembleBlock, self).__init__()

        # Read
        self.read = Read_ignore()
        if read == "add":
            self.read = Read_add()
        elif read == "proj":
            self.read = Read_projection(emb_dim)

        # Projection + Resample
        self.resample = Resample(emb_dim, resample_dim, scale_factor)

    def forward(self, x, W, T):
        B, N, E = x.shape
        H = int((N - 1) / (W * T))

        x = self.read(x)

        # concatenation
        x = rearrange(x, "b (h w) c -> b c h w", b=B, h=H, w=W)

        x = self.resample(x)
        return x


class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x


class FusionBlock(nn.Module):
    def __init__(self, resample_dim):
        super(FusionBlock, self).__init__()
        self.res_conv1 = ResidualConvUnit(resample_dim)
        self.res_conv2 = ResidualConvUnit(resample_dim)
        # self.resample = nn.ConvTranspose2d(resample_dim, resample_dim, kernel_size=2, stride=2, padding=0, bias=True, dilation=1, groups=1)

    def forward(self, x, previous_stage=None):
        if previous_stage == None:
            previous_stage = torch.zeros_like(x)
        output_stage1 = self.res_conv1(x)
        output_stage1 += previous_stage
        output_stage2 = self.res_conv2(output_stage1)
        output_stage2 = nn.functional.interpolate(
            output_stage2, scale_factor=2, mode="bilinear", align_corners=True
        )
        return output_stage2
