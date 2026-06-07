# This code is based on the original implementation by Facebook Research.
# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
# Original source: https://github.com/facebookresearch/TimeSformer/blob/main/timesformer/models/vit.py
# Modifications made by André Françani, 2025.

import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from models.vit_utils import DropPath, to_2tuple, trunc_normal_
from models.blocks import ReassembleBlock, FusionBlock, Interpolate


class MLP(nn.Module):
    """
    A simple feedforward neural network (MLP) block.

    This class implements a two-layer MLP with a specified activation function and dropout.

    Args:
        in_features (int): Number of input features.
        hidden_features (int, optional): Number of hidden features. Defaults to in_features.
        out_features (int, optional): Number of output features. Defaults to in_features.
        act_layer (callable, optional): Activation layer to use. Defaults to nn.GELU.
        drop (float, optional): Dropout rate. Defaults to 0.0.
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        Forward pass through the MLP block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (B, out_features).
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """
    Multi-head attention mechanism.

    This class implements a multi-head attention mechanism, allowing the model to focus on different parts of the input.

    Args:
        dim (int): Input dimension of the attention mechanism.
        num_heads (int, optional): Number of attention heads. Defaults to 8.
        qkv_bias (bool, optional): Whether to use bias in Q, K, V projections. Defaults to False.
        qk_scale (float, optional): Scaling factor for QK dot products. Defaults to None.
        attn_drop (float, optional): Dropout rate for attention. Defaults to 0.0.
        proj_drop (float, optional): Dropout rate for output projection. Defaults to 0.0.
        with_qkv (bool, optional): Whether to use combined QKV projection. Defaults to True.
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        with_qkv=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        """
        Forward pass through the multi-head attention mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C).

        Returns:
            torch.Tensor: Output tensor of shape (B, N, C).
        """
        B, N, C = x.shape
        if self.with_qkv:
            qkv = (
                self.qkv(x)
                .reshape(B, N, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(
                0, 2, 1, 3
            )
            q, k, v = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
            x = self.proj(x)
            x = self.proj_drop(x)
        return x


class SPM(nn.Module):
    """
    Structure Perception Module
    Source: https://github.com/kamiLight/CADepth-master/blob/main/networks/spm.py
    """

    def __init__(self):
        super(SPM, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : input feature maps(B X C X H X W)
        returns :
            out : attention value + input feature
            attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = out + x

        return out


class Block(nn.Module):
    """
    Transformer block that includes both attention and feedforward MLP.

    This class represents a single transformer block, consisting of attention and MLP layers with normalization and residual connections.

    Args:
        dim (int): Input dimension of the block.
        num_heads (int): Number of attention heads.
        mlp_ratio (float, optional): Ratio of the hidden dimension in the MLP to the input dimension. Defaults to 4.0.
        qkv_bias (bool, optional): Whether to use bias in Q, K, V projections. Defaults to False.
        qk_scale (float, optional): Scaling factor for QK dot products. Defaults to None.
        drop (float, optional): Dropout rate for the attention and MLP layers. Defaults to 0.0.
        attn_drop (float, optional): Dropout rate for attention. Defaults to 0.0.
        drop_path (float, optional): Drop path rate for stochastic depth. Defaults to 0.1.
        act_layer (callable, optional): Activation layer to use. Defaults to nn.GELU.
        norm_layer (callable, optional): Normalization layer to use. Defaults to nn.LayerNorm.
        attention_type (str, optional): Type of attention to use. Defaults to "divided_space_time".
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.1,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attention_type="divided_space_time",
    ):
        super().__init__()
        self.attention_type = attention_type
        assert attention_type in [
            "divided_space_time",
            "space_only",
            "joint_space_time",
        ]

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        # Temporal Attention Parameters
        if self.attention_type == "divided_space_time":
            self.temporal_norm1 = norm_layer(dim)
            self.temporal_attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
            self.temporal_fc = nn.Linear(dim, dim)

        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        # Structure perception module
        self.spm = SPM()

    def forward(self, x, B, T, W):
        num_spatial_tokens = (x.size(1) - 1) // T
        H = num_spatial_tokens // W

        if self.attention_type in ["space_only", "joint_space_time"]:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        elif self.attention_type == "divided_space_time":
            # Temporal
            xt = x[:, 1:, :]
            xt = rearrange(xt, "b (h w t) m -> (b h w) t m", b=B, h=H, w=W, t=T)
            res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(xt)))
            res_temporal = rearrange(
                res_temporal, "(b h w) t m -> b (h w t) m", b=B, h=H, w=W, t=T
            )
            res_temporal = self.temporal_fc(res_temporal)
            xt = x[:, 1:, :] + res_temporal

            # Spatial
            init_cls_token = x[:, 0, :].unsqueeze(1)
            cls_token = init_cls_token.repeat(1, T, 1)
            cls_token = rearrange(cls_token, "b t m -> (b t) m", b=B, t=T).unsqueeze(1)
            xs = xt
            xs = rearrange(xs, "b (h w t) m -> (b t) (h w) m", b=B, h=H, w=W, t=T)
            xs = torch.cat((cls_token, xs), 1)
            res_spatial = self.drop_path(self.attn(self.norm1(xs)))

            # Taking care of CLS token
            cls_token = res_spatial[:, 0, :]
            cls_token = rearrange(cls_token, "(b t) m -> b t m", b=B, t=T)
            cls_token = torch.mean(cls_token, 1, True)  ## averaging for every frame
            res_spatial = res_spatial[:, 1:, :]
            res_spatial = rearrange(
                res_spatial, "(b t) (h w) m -> b (h w t) m", b=B, h=H, w=W, t=T
            )
            res = res_spatial
            x = xt

            # Structure perpection module
            res_spatial = rearrange(
                res_spatial, "b (h w t) m -> b (t m) h w", b=B, h=H, w=W, t=T
            )
            res_spatial = self.spm(res_spatial)
            res_spatial = rearrange(
                res_spatial, "b (t m) h w -> b (h w t) m", b=B, h=H, w=W, t=T
            )

            # Mlp
            x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding
    This class splits an image into patches and embeds them using a convolutional layer.
    """

    def __init__(self, img_size=(224, 224), patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        # Convert patch_size to a tuple if it's not already
        patch_size = to_2tuple(patch_size)

        # Calculate the total number of patches based on image and patch sizes
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])

        # Store parameters as class attributes
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        # Convolutional layer to convert image patches to embedding vectors
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # B: Batch size, C: Channels, T: Time frames, H: Height, W: Width
        B, C, T, H, W = x.shape

        # Merge batch and time dimensions for processing
        x = rearrange(x, "b c t h w -> (b t) c h w")

        # Apply convolution to embed patches
        x = self.proj(x)

        # Update width after patch embedding
        W = x.size(-1)

        # Flatten the spatial dimensions (H, W) and transpose to prepare for the transformer
        x = x.flatten(2).transpose(1, 2)

        return x, T, W


class PoseHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of hidden layer features.
            output_dim (int): Number of output features.
        """
        super(PoseHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass for the pose head.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, output_dim).
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class DepthHead(nn.Module):
    """
    Depth estimation head for 3D feature maps.

    Args:
        features (int): The number of input feature channels.
        non_negative (bool): If True, applies ReLU activation to ensure non-negative depth values.
    """

    def __init__(self, features, non_negative=True):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv3d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=(1, 2, 2), mode="trilinear", align_corners=True),
            nn.GELU(),
            nn.Conv3d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv3d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
        )

    def forward(self, x):
        """
        Forward pass of the depth head.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T, H, W)

        Returns:
            torch.Tensor: Output depth map tensor of shape (B, 1, T, H, W).
        """
        return self.head(x)


class DPTSformer(nn.Module):
    """
    DPTSformer model based on ViT
    """

    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        num_frames=5,
        attention_type="divided_space_time",
        dropout=0.0,
        dpt_features=256,
        vit_layers=[0, 1, 2, 3],
        dpt_readout="proj",
    ):
        super().__init__()
        self.in_chans = in_chans
        self.vit_layers = vit_layers

        # Store attention type and model depth
        self.attention_type = attention_type
        self.depth = depth

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

        # Set number of output classes
        self.num_classes = num_classes
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )

        # Patch embedding module to split image and embed patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        # Get the number of patches generated from the input image
        num_patches = self.patch_embed.num_patches

        # Positional Embeddings
        # Learnable class token for classification tasks
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embeddings for patches and class token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Time embeddings for video input (if applicable)
        if self.attention_type != "space_only":
            self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
            self.time_drop = nn.Dropout(p=drop_rate)

        # Attention Blocks
        # Stochastic depth schedule for each transformer block
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]

        # Create a sequence of transformer blocks
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    attention_type=self.attention_type,
                )
                for i in range(self.depth)
            ]
        )

        # Reassemble blocks
        self.reassemble_1 = ReassembleBlock(
            embed_dim=embed_dim,
            output_dim=dpt_features,
            read_type=dpt_readout,
            scale_factor=4,
        )
        self.reassemble_2 = ReassembleBlock(
            embed_dim=embed_dim,
            output_dim=dpt_features,
            read_type=dpt_readout,
            scale_factor=2,
        )
        self.reassemble_3 = ReassembleBlock(
            embed_dim=embed_dim,
            output_dim=dpt_features,
            read_type=dpt_readout,
            scale_factor=1,
        )
        self.reassemble_4 = ReassembleBlock(
            embed_dim=embed_dim,
            output_dim=dpt_features,
            read_type=dpt_readout,
            scale_factor=0.5,
        )

        # Fusion blocks
        self.fusion_1 = FusionBlock(
            feature_dim=dpt_features, output_dim=dpt_features, scale_factor=2
        )
        self.fusion_2 = FusionBlock(
            feature_dim=dpt_features, output_dim=dpt_features, scale_factor=2
        )
        self.fusion_3 = FusionBlock(
            feature_dim=dpt_features, output_dim=dpt_features, scale_factor=2
        )
        self.fusion_4 = FusionBlock(
            feature_dim=dpt_features, output_dim=dpt_features, scale_factor=2
        )

        # Layer normalization at the end of transformer blocks
        self.norm = norm_layer(embed_dim)

        # Depth Estimation Head
        self.depth_head = DepthHead(dpt_features, non_negative=True)

        # Classification head (linear layer for classification)
        self.translation_head = PoseHead(embed_dim, 256, num_classes // 2)
        self.rotation_head = PoseHead(embed_dim, 256, num_classes // 2)

        # Initialize positional and class token embeddings
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Weight initialization for linear and normalization layers"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        """Specify which parameters should not be decayed by the optimizer"""
        return {"pos_embed", "cls_token", "time_embed"}

    def get_classifier(self):
        """Return the classification head (linear layer)"""
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        """Reset the classifier with a new number of output classes"""
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_features(self, x):
        """Forward pass to compute features before classification"""
        B = x.shape[0]  # Batch size

        # Apply patch embedding
        x, T, W = self.patch_embed(x)

        # Expand class token for the current batch
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)

        # Concatenate class token to the patch embeddings
        x = torch.cat((cls_tokens, x), dim=1)

        ## Resizing positional embeddings if they don't match the input at inference
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = self.pos_embed
            cls_pos_embed = pos_embed[0, 0, :].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[0, 1:, :].unsqueeze(0).transpose(1, 2)
            P = int(other_pos_embed.size(2) ** 0.5)
            H = x.size(1) // W
            other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P)
            new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode="nearest")
            new_pos_embed = new_pos_embed.flatten(2).transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            x = x + new_pos_embed
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        # Time Embeddings (for video inputs)
        if self.attention_type != "space_only":
            cls_tokens = x[:B, 0, :].unsqueeze(1)  # Extract class token
            x = x[:, 1:]  # Remove class token for processing
            x = rearrange(x, "(b t) n m -> (b n) t m", b=B, t=T)

            # Resize time embeddings if necessary
            if T != self.time_embed.size(1):
                time_embed = self.time_embed.transpose(1, 2)
                new_time_embed = F.interpolate(time_embed, size=(T), mode="nearest")
                new_time_embed = new_time_embed.transpose(1, 2)
                x = x + new_time_embed
            else:
                x = x + self.time_embed
            x = self.time_drop(x)
            x = rearrange(x, "(b n) t m -> b (n t) m", b=B, t=T)
            x = torch.cat((cls_tokens, x), dim=1)

        # Apply attention blocks before DPT
        features = []
        for i in range(self.depth):
            x = self.blocks[i](x, B, T, W)
            if i in self.vit_layers:
                features.append(x)

        # Apply reassemble blocks
        features_1 = self.reassemble_1(features[0], W, T)
        features_2 = self.reassemble_2(features[1], W, T)
        features_3 = self.reassemble_3(features[2], W, T)
        features_4 = self.reassemble_4(features[3], W, T)

        # Apply fusion blocks
        path_4 = self.fusion_4(features_4)  # No previous fusion block
        path_3 = self.fusion_3(path_4, features_3)
        path_2 = self.fusion_2(path_3, features_2)
        path_1 = self.fusion_1(path_2, features_1)

        # For 'space_only' attention, average predictions over all frames
        if self.attention_type == "space_only":
            x = rearrange(x, "(b t) n m -> b t n m", b=B, t=T)
            x = torch.mean(x, 1)  # Average predictions for each frame

        # Final normalization
        x = self.norm(x)
        return x[:, 0], path_1

    def forward(self, x):
        """Forward pass to compute class predictions"""
        x, x_depth = self.forward_features(x)
        x_t = self.translation_head(x)
        x_rot = self.rotation_head(x)
        x_depth = self.depth_head(x_depth).squeeze(1)
        return [x_rot, x_t], x_depth


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    img_size = (512, 384)
    num_classes = 12
    num_frames = 3
    in_chans = 3
    model = DPTSformer(
        img_size=img_size,
        patch_size=16,
        in_chans=in_chans,
        num_classes=num_classes,
        embed_dim=384,
        depth=9,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        drop_path_rate=0.1,
        num_frames=num_frames,
        attention_type="divided_space_time",
        dropout=0.1,
        dpt_features=256,
        vit_layers=[0, 1, 2, 3],
        dpt_readout="proj",
    )

    # Random input tensor (B x C x T x H x W)
    batch_size = 8
    x = torch.randn(batch_size, in_chans, num_frames, img_size[0], img_size[1])

    model = model.to(device)
    model = model.eval()
    with torch.no_grad():
        output = model(x.to(device))

    print(f"Output shape: {output.shape}")

    # # Forward pass through the model
    # import torch.autograd.profiler as profiler

    # with profiler.profile(use_cuda=True) as prof:
    #     model(x)  # Forward pass with your TSformerVO

    # print(prof.key_averages().table(sort_by="cuda_time_total"))
