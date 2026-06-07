# Code adapted from https://github.com/antocad/FocusOnDepth/blob/master/FOD/Loss.py
# Modifications made by André Françani, 2025.

from typing import Tuple, Optional, Sequence
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class ScaleAndShiftInvariantLoss(nn.Module):
    """
    Scale-and-Shift Invariant Loss for depth estimation, inspired by MegaDepth.
    This loss removes scale and shift ambiguities by optimizing for the best
    scale and shift parameters and penalizing structural differences.

    Args:
        alpha (float): Weight for the gradient-based loss term.
        bbox_norm (Optional[Sequence[float]]): Bbox espacial em coords normalizadas
            (y_min, y_max, x_min, x_max) em [0, 1]. Quando fornecido, pixels fora
            dessa bbox sao ignorados na loss (tanto no MSE quanto no gradiente).
            Util pra restringir a supervisao a regiao com cobertura LiDAR confiavel.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        bbox_norm: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.bbox_norm = tuple(bbox_norm) if bbox_norm is not None else None
        # Cache da bbox_mask por (H, W) para evitar reconstruir a cada batch
        self._bbox_mask_cache = {}

    def compute_scale_and_shift(
        self, depth_pred: torch.Tensor, depth_gt: torch.Tensor, valid_mask: torch.Tensor
    ):
        """
        Computes the optimal scale and shift values to align the predicted depth with the ground truth.

        Args:
            depth_pred (torch.Tensor): Predicted depth map (B, C, H, W)
            depth_gt (torch.Tensor): Ground truth depth map (B, C, H, W)
            valid_mask (torch.Tensor): Mask indicating valid depth values.

        Returns:
            torch.Tensor, torch.Tensor: Optimized scale and shift values.
        """
        depth_pred_valid = depth_pred * valid_mask
        depth_gt_valid = depth_gt * valid_mask

        mean_pred = torch.sum(depth_pred_valid) / torch.sum(valid_mask)
        mean_gt = torch.sum(depth_gt_valid) / torch.sum(valid_mask)

        scale = torch.sum(
            (depth_gt_valid - mean_gt) * (depth_pred_valid - mean_pred)
        ) / (torch.sum((depth_pred_valid - mean_pred) ** 2) + 1e-6)

        shift = mean_gt - scale * mean_pred

        return scale, shift

    def _get_bbox_mask(
        self, shape: Tuple[int, int], device: torch.device, dtype: torch.dtype
    ) -> Optional[torch.Tensor]:
        """
        Constroi (e cacheia) a mascara espacial 2D (H, W) a partir de self.bbox_norm.
        Retorna None quando nenhum bbox foi configurado.

        A mascara e 2D pra fazer broadcast natural com tensores (B, T, H, W) ou
        (B, H, W) do dataloader.
        """
        if self.bbox_norm is None:
            return None
        h, w = shape
        key = (h, w, device, dtype)
        mask = self._bbox_mask_cache.get(key)
        if mask is not None:
            return mask
        y_min, y_max, x_min, x_max = self.bbox_norm
        y0 = int(round(y_min * h))
        y1 = int(round(y_max * h))
        x0 = int(round(x_min * w))
        x1 = int(round(x_max * w))
        # Garante bbox nao-vazia
        y0 = max(0, min(h - 1, y0))
        y1 = max(y0 + 1, min(h, y1 + 1))
        x0 = max(0, min(w - 1, x0))
        x1 = max(x0 + 1, min(w, x1 + 1))
        mask = torch.zeros((h, w), device=device, dtype=dtype)
        mask[y0:y1, x0:x1] = 1.0
        self._bbox_mask_cache[key] = mask
        return mask

    def compute_gradient_loss(
        self,
        depth_pred: torch.Tensor,
        depth_gt: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes the gradient-matching loss between predicted and ground truth depth.

        Args:
            depth_pred (torch.Tensor): Predicted depth (B, T, H, W).
            depth_gt (torch.Tensor): Ground truth depth (B, T, H, W).
            mask (Optional[torch.Tensor]): Mascara espacial 2D (H, W); quando
                fornecida, restringe os gradientes a pares onde AMBOS os lados
                estao dentro da mascara (evita borda da bbox virar sinal espurio).
                Como a mascara nao varia no tempo, o gradiente temporal so requer
                que o pixel esteja dentro da bbox.

        Returns:
            torch.Tensor: Gradient loss value (escalar).
        """
        # Tensores de entrada sao (B, T, H, W). Indexacao explicita pra evitar
        # confusao com .squeeze degenerada quando T=1.
        # dx = gradiente em H (linhas consecutivas), shape (B, T, H-1, W)
        # dy = gradiente em T (frames consecutivos), shape (B, T-1, H, W)
        pred_dx = depth_pred[:, :, :-1, :] - depth_pred[:, :, 1:, :]
        pred_dy = depth_pred[:, :-1, :, :] - depth_pred[:, 1:, :, :]
        gt_dx = depth_gt[:, :, :-1, :] - depth_gt[:, :, 1:, :]
        gt_dy = depth_gt[:, :-1, :, :] - depth_gt[:, 1:, :, :]

        if mask is None:
            return torch.mean(torch.abs(pred_dx - gt_dx)) + torch.mean(
                torch.abs(pred_dy - gt_dy)
            )

        # Mascaras 2D apropriadas pra cada gradiente:
        #   - dx (H consecutivos): pares de linhas validos quando ambos na bbox
        #   - dy (T consecutivos): bbox nao muda no tempo -> mascara espacial completa
        dx_mask_2d = mask[:-1, :] * mask[1:, :]   # (H-1, W)
        dy_mask_2d = mask                          # (H,   W)

        eps = 1e-6
        # Broadcasting de mascara 2D contra (B, T, H', W) ou (B, T-1, H, W)
        loss_x = (torch.abs(pred_dx - gt_dx) * dx_mask_2d).sum() / (
            dx_mask_2d.sum() * pred_dx.shape[0] * pred_dx.shape[1] + eps
        )
        loss_y = (torch.abs(pred_dy - gt_dy) * dy_mask_2d).sum() / (
            dy_mask_2d.sum() * pred_dy.shape[0] * pred_dy.shape[1] + eps
        )
        return loss_x + loss_y

    def forward(
        self,
        depth_pred: torch.Tensor,
        depth_gt: torch.Tensor,
        tensorboard_writer: SummaryWriter = None,
        iter: int = None,
    ) -> torch.Tensor:
        """
        Computes the Scale-and-Shift Invariant Loss.

        Args:
            depth_pred (torch.Tensor): Predicted depth map (B, C, H, W)
            depth_gt (torch.Tensor): Ground truth depth map (B, C, H, W)
            tensorboard_writer (SummaryWriter, optional): TensorBoard writer to log losses.
            iter (int, optional): Iteration counter for TensorBoard.

        Returns:
            torch.Tensor: Scale-invariant loss value.
        """
        valid_mask = (depth_gt > 0).float()

        # Aplica a bbox espacial (quando configurada): valid_mask passa a ser
        # interseccao de pixels com GT valido E dentro da bbox de cobertura LiDAR.
        bbox_mask = self._get_bbox_mask(
            shape=tuple(depth_gt.shape[-2:]),
            device=depth_gt.device,
            dtype=depth_gt.dtype,
        )
        if bbox_mask is not None:
            valid_mask = valid_mask * bbox_mask

        # Compute optimal scale and shift
        scale, shift = self.compute_scale_and_shift(depth_pred, depth_gt, valid_mask)

        # Apply scale and shift correction
        depth_pred_aligned = scale * depth_pred + shift

        # Compute scale-invariant loss
        loss_mse = torch.mean((depth_pred_aligned - depth_gt) ** 2 * valid_mask)

        # Compute gradient loss (mascarada pela bbox quando configurada, pra nao
        # criar gradientes espurios na borda do retangulo).
        # depth_pred_aligned / depth_gt sao (B, T, H, W) -- nao squeeza.
        loss_grad = self.compute_gradient_loss(
            depth_pred_aligned, depth_gt, mask=bbox_mask
        )

        # Combine losses
        total_loss = loss_mse + self.alpha * loss_grad

        # Log losses in TensorBoard
        if tensorboard_writer and iter is not None:
            tensorboard_writer.add_scalar("train/ssi_mse_loss", loss_mse.item(), iter)
            tensorboard_writer.add_scalar("train/ssi_grad_loss", loss_grad.item(), iter)
            tensorboard_writer.add_scalar(
                "train/ssi_total_loss", total_loss.item(), iter
            )

        return total_loss


def get_depth_criterion(
    loss_name: str,
    alpha: float = 0.5,
    bbox_norm: Optional[Sequence[float]] = None,
) -> nn.Module:
    """
    Args:
        loss_name (str): The name of the loss function.
        alpha (float, optional): Weight for the gradient-based term in SSI loss.
        bbox_norm (Optional[Sequence[float]]): Bbox espacial normalizada
            (y_min, y_max, x_min, x_max) pra mascarar a loss.

    Returns:
        nn.Module: An instance of the ScaleAndShiftInvariantLoss.
    """
    if loss_name.lower() == "ssi_loss":
        return ScaleAndShiftInvariantLoss(alpha=alpha, bbox_norm=bbox_norm)

    else:
        raise ValueError(
            f"Unsupported loss name: {loss_name}. Available option: 'ssi_loss'."
        )


if __name__ == "__main__":
    # Define dummy inputs for testing
    batch_size, T, H, W = 4, 3, 128, 128
    depth_pred = torch.rand(batch_size, T, H, W)
    depth_gt = torch.rand(batch_size, T, H, W)

    # Test ScaleAndShiftInvariantLoss
    ssi_loss_fn = get_depth_criterion("ssi_loss", alpha=0.5)
    loss = ssi_loss_fn(depth_pred, depth_gt)
    print("Loss:", loss.item())
