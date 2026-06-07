import torch
import torch.nn as nn
import torch.nn.functional as F


class ReassembleBlock(nn.Module):
    def __init__(self, embed_dim, output_dim, scale_factor=2, read_type="proj"):
        """
        Args:
            embed_dim: Dimension of input tokens (E_{dim}).
            output_dim: Output feature dimension (D).
            scale_factor: Upsampling factor.
            read_type: One of ["ignore", "add", "proj"].
        """
        super().__init__()
        self.read_type = read_type
        self.output_dim = output_dim
        self.scale_factor = scale_factor

        if read_type == "proj":
            self.read_proj = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim), nn.GELU()
            )

        self.conv1x1 = nn.Conv3d(embed_dim, output_dim, kernel_size=1)
        self.conv3x3 = nn.Conv3d(
            output_dim, output_dim, kernel_size=3, padding=1, stride=1
        )

    def forward(self, x, W, T):
        """
        Args:
            x: Tensor of shape [B, H*W*T+1, E_{dim}]
            W: Spatial dimensions (number of patches in width).
            T: Temporal dimension (number of frames).

        Returns:
            Tensor of shape [B, D, T, H, W]
        """
        B, N, E = x.shape
        H = int((N - 1) / (W * T))

        # Readout token and actual tokens
        readout, tokens = x[:, 0], x[:, 1:]

        if self.read_type == "ignore":
            pass
        elif self.read_type == "add":
            tokens = tokens + readout.unsqueeze(1)
        elif self.read_type == "proj":
            readout_expanded = readout.unsqueeze(1).expand(-1, tokens.shape[1], -1)
            tokens = self.read_proj(torch.cat([tokens, readout_expanded], dim=-1))

        # Reshape to [B, E, T, H, W]
        tokens = tokens.view(B, H, W, T, E).permute(0, 4, 3, 1, 2)  # [B, E, T, H, W]

        # Convolutional processing
        tokens = self.conv1x1(tokens)  # [B, D, T, H, W]
        tokens = self.conv3x3(tokens)  # [B, D, T, H, W]

        # Upsample (Only in H,W dim, not in C)
        tokens = F.interpolate(
            tokens,
            scale_factor=(1, self.scale_factor, self.scale_factor),
            mode="trilinear",
            align_corners=True,
        )

        return tokens


class ResidualConvUnit(nn.Module):
    """Residual convolution module using 3D convolutions."""

    def __init__(self, features):
        super().__init__()
        self.conv1 = nn.Conv3d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.conv2 = nn.Conv3d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x


class FusionBlock(nn.Module):
    def __init__(self, feature_dim, output_dim, scale_factor=2):
        """
        Args:
            feature_dim: Input feature dimension (D).
            output_dim: Output feature dimension (D').
            scale_factor: Upsampling factor.
        """
        super().__init__()
        self.scale_factor = scale_factor

        self.residual_conv1 = ResidualConvUnit(feature_dim)  # Apply residual conv unit
        self.residual_conv2 = ResidualConvUnit(output_dim)  # Apply residual conv unit

        self.upsample = nn.ConvTranspose3d(
            output_dim,
            output_dim,
            kernel_size=3,
            stride=scale_factor,
            padding=1,
            output_padding=scale_factor - 1,
        )

    def forward(self, x, prev_fusion=None):
        x = self.residual_conv1(x)

        if prev_fusion is not None:
            x = x + prev_fusion

        x = self.residual_conv2(x)

        # Upsample (Only in H,W dim, not in C)
        x = F.interpolate(
            x,
            scale_factor=(1, self.scale_factor, self.scale_factor),
            mode="trilinear",
            align_corners=True,
        )

        return x


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode="trilinear", align_corners=True):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )
