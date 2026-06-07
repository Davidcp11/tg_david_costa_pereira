from typing import Any, Dict, Union

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, zeros_
from functools import partial

from models.dptsformer import DPTSformer

# from models.tsformer_spm import TSformerSPM
from models.helpers import load_pretrained


default_cfgs = {
    "vit_patch16_edim768": {
        "url": "https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",  #'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        "first_conv": "patch_embed.proj",
        "classifier": "head",
    },
    "vit_patch16_edim192": {
        "url": "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
        "first_conv": "patch_embed.proj",
        "classifier": "head",
    },
    "vit_patch32_edim1024": {
        "url": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth",
        "first_conv": "patch_embed.proj",
        "classifier": "head",
    },
    "vit_patch16_edim384": {
        "url": "https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
        "first_conv": "patch_embed.proj",
        "classifier": "head",
    },
}


def build_model(
    model_params: Dict[str, Any], device: Union[torch.device, str]
) -> torch.nn.Module:
    """
    Builds and initializes a Vision Transformer (ViT) model based on the provided parameters.

    Args:
        model_params (Dict[str, Any]): A dictionary containing the parameters needed to build the model.
        device (Union[torch.device, str]): Either a CPU or a CUDA device.

    Returns:
        torch.nn.Module: The initialized Vision Transformer model.
    """
    # Build model
    if model_params.get("spm", False):
        model = DPTSformer(
            img_size=model_params["image_size"],
            num_classes=model_params["num_classes"],
            patch_size=model_params["patch_size"],
            in_chans=model_params["num_channels"],
            embed_dim=model_params["embed_dim"],
            depth=model_params["depth"],
            num_heads=model_params["num_heads"],
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            drop_rate=0.0,
            attn_drop_rate=model_params["attn_dropout"],
            drop_path_rate=model_params["ff_dropout"],
            num_frames=model_params["num_frames"],
            attention_type=model_params["attention_type"],
            vit_layers=model_params["vit_layers"],
        )
        print("--- Building TSformerVO with SPM")
    else:
        model = DPTSformer(
            img_size=model_params["image_size"],
            num_classes=model_params["num_classes"],
            patch_size=model_params["patch_size"],
            in_chans=model_params["num_channels"],
            embed_dim=model_params["embed_dim"],
            depth=model_params["depth"],
            num_heads=model_params["num_heads"],
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            drop_rate=0.0,
            attn_drop_rate=model_params["attn_dropout"],
            drop_path_rate=model_params["ff_dropout"],
            num_frames=model_params["num_frames"],
            attention_type=model_params["attention_type"],
            vit_layers=model_params["vit_layers"],
        )

    # Load ViT pretrained on ImageNet
    if model_params.get("pretrained_ViT", False):
        model = load_pretrained_ViT(model)

    # Send model to device
    model = model.to(device)

    return model


def load_pretrained_ViT(
    model: torch.nn.Module, model_params: Dict[str, Any]
) -> torch.nn.Module:
    """
    Loads pretrained Vision Transformer (ViT) weights into the model. If no checkpoints are available,
    the model is initialized with pretrained ViT weights.

    Args:
        model (torch.nn.Module): The model to load the pretrained weights into.
        model_params (Dict[str, Any]): Dictionary containing the following model parameters:
            - "image_size" (Tuple[int, int]): The input image size.
            - "patch_size" (int): The size of the patches to split the image into.
            - "embed_dim" (int): The embedding dimension for the model.
            - "num_classes" (int): The number of output classes.
            - "num_channels" (int): The number of input channels.
            - "num_frames" (int): Number of frames for input if working with videos.
            - "attention_type" (str): The type of attention mechanism to use.

    Returns:
        torch.nn.Module: The model with the loaded pretrained state dict.
    """
    print(
        " -- No checkpoint_last.pth and no model initialization file found. Initializing with pretrained ViT weights --"
    )

    # Model parameters for computing patch and model info
    img_size = model_params["image_size"]
    num_patches = (img_size[0] // model_params["patch_size"]) * (
        img_size[1] // model_params["patch_size"]
    )

    # Construct model name for loading default configuration
    model_name = (
        f"vit_patch{model_params['patch_size']}_edim{model_params['embed_dim']}"
    )
    model.default_cfg = default_cfgs[model_name]

    print(
        f"--- Loading pretrained ViT model for training ---\n{model.default_cfg['url']}\n"
    )

    # Load pretrained weights with custom patch embedding filter function (_conv_filter)
    load_pretrained(
        model,
        num_classes=model_params["num_classes"],
        in_chans=model_params["num_channels"],
        filter_fn=_conv_filter,  # Custom function for converting weights
        img_size=img_size,
        num_frames=model_params["num_frames"],
        num_patches=num_patches,
        attention_type=model_params["attention_type"],
        pretrained_model="",  # Empty for no specific model
    )

    def _conv_filter(
        state_dict: Dict[str, torch.Tensor], patch_size: int = 16
    ) -> Dict[str, torch.Tensor]:
        """
        Converts patch embedding weights from a linear projection to convolutional.

        Args:
            state_dict (Dict[str, torch.Tensor]): The state dictionary of the pretrained model.
            patch_size (int, optional): The size of the patch for the projection. Defaults to 16.

        Returns:
            Dict[str, torch.Tensor]: Modified state dictionary with convolutional patch embedding weights.
        """
        out_dict = {}
        for k, v in state_dict.items():
            # Modify the patch embedding weights for convolutional layers
            if "patch_embed.proj.weight" in k:
                # Adjust the patch size if necessary
                if v.shape[-1] != patch_size:
                    patch_size = v.shape[-1]
                # Reshape the patch embedding weight for convolution
                v = v.reshape((v.shape[0], 3, patch_size, patch_size))
            out_dict[k] = v
        return out_dict

    return model


def load_model_state(
    model: torch.nn.Module,
    checkpoint: Union[Dict[str, Any], None],
) -> torch.nn.Module:
    """
    Loads the state dict of the model from a given checkpoint.

    Args:
        model (torch.nn.Module): The model to load the state into.
        checkpoint (Union[Dict[str, Any], None]): The checkpoint dictionary containing the model's state dict.
                                                   If None, the model remains in its initialized state.

    Returns:
        torch.nn.Module: The model with the loaded state dict (if checkpoint is provided),
                         or the model initialized with default parameters.
    """
    if checkpoint:
        try:
            # Load the model state dictionary from the checkpoint
            model.load_state_dict(checkpoint.get("model_state_dict"))
            print("--- Model state loaded successfully.")
        except KeyError:
            print("--- No model state dict found in checkpoint!")
    else:
        print("--- No checkpoint provided. Model initialized with default parameters.")
    return model


def initialize_heads_with_xavier(model: nn.Module) -> None:
    """
    Reinitializes all layers in the model with "head" in their name using Xavier initialization.

    Args:
        model (nn.Module): The model containing the layers to be reinitialized.

    Raises:
        TypeError: If a "head" layer is not an instance of nn.Linear.
    """
    for name, module in model.named_modules():
        if name in ["translation_head", "rotation_head"]:
            # Initialize fc1
            xavier_uniform_(module.fc1.weight)
            if module.fc1.bias is not None:
                zeros_(module.fc1.bias)
            # Initialize fc2
            xavier_uniform_(module.fc2.weight)
            if module.fc2.bias is not None:
                zeros_(module.fc2.bias)


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in the given model.

    Args:
        model (torch.nn.Module): The PyTorch model whose parameters are to be counted.

    Returns:
        int: The total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    img_size = (192, 640)
    num_classes = 10
    num_frames = 4
    in_chans = 3
    model = TSformerSPM(
        img_size=img_size,
        patch_size=16,
        in_chans=in_chans,
        num_classes=num_classes,
        embed_dim=768,
        depth=12,
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
    )

    # Random input tensor (B x C x T x H x W)
    batch_size = 8
    x = torch.randn(batch_size, in_chans, num_frames, img_size[0], img_size[1])

    model = model.to(device)
    with torch.no_grad():
        output = model(x.to(device))

    print(f"Output shape: {output.shape}")

    # # Forward pass through the model
    # import torch.autograd.profiler as profiler

    # with profiler.profile(use_cuda=True) as prof:
    #     model(x)  # Forward pass with your TSformerVO

    # print(prof.key_averages().table(sort_by="cuda_time_total"))
