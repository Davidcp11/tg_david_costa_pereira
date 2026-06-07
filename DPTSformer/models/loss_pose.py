import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


class WeightedMSELoss(nn.Module):
    """
    Computes a weighted MSE loss for angle and translation components.

    Args:
        window_size (int): Used to reshape tensors for weighted loss calculation.
        lambda_rot (float, optional): Weight for the angle loss.
        lambda_t (float, optional): Weight for the translation loss.
    """

    def __init__(
        self,
        window_size: int = 3,
        lambda_t: float = 1.0,
        lambda_rot: float = 1.0,
    ):
        super(WeightedMSELoss, self).__init__()
        self.window_size = window_size
        self.lambda_rot = float(lambda_rot)
        self.lambda_t = float(lambda_t)
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        tensorboard_writer: SummaryWriter = None,
        iter: int = None,
    ) -> torch.Tensor:
        """
        Forward pass to calculate the weighted MSE loss.

        Args:
            y_pred (torch.Tensor): Tensor with dimention [batch_size x (window_size - 1) * 6]
            y_true (torch.Tensor): Tensor with dimention [batch_size x (window_size - 1) * 6]
            tensorboard_writer (SummaryWriter): Tensorboard writer to log intermediate losses
            iter (int): Iteration counter to save in Tensorboard
        """

        # Separate angles and translation for ground truth
        y_true = torch.reshape(y_true, (y_true.shape[0], self.window_size - 1, 6))
        gt_angles = y_true[:, :, :3].flatten()
        gt_translation = y_true[:, :, 3:].flatten()

        # Separate angles and translation for predicted
        y_pred_angles, y_pred_translation = y_pred
        y_pred_angles = torch.reshape(
            y_pred_angles, (y_pred_angles.shape[0], self.window_size - 1, 3)
        )
        y_pred_translation = torch.reshape(
            y_pred_translation, (y_pred_translation.shape[0], self.window_size - 1, 3)
        )
        estimated_angles = y_pred_angles.flatten()
        estimated_translation = y_pred_translation.flatten()

        # Calculate weighted losses for angles and translation
        rot_loss = self.mse_loss(estimated_angles, gt_angles.float())
        t_loss = self.mse_loss(estimated_translation, gt_translation.float())

        # Compute final weighted loss
        loss = self.lambda_t * t_loss + self.lambda_rot * rot_loss

        # Log intermediate losses in TensorBoard (only training)
        if tensorboard_writer:
            tensorboard_writer.add_scalar(
                "train/E_tx",
                torch.norm(y_true[:, :, 3] - y_pred_translation[:, :, 0], p=2).item(),
                iter,
            )
            tensorboard_writer.add_scalar(
                "train/E_ty",
                torch.norm(y_true[:, :, 4] - y_pred_translation[:, :, 1], p=2).item(),
                iter,
            )
            tensorboard_writer.add_scalar(
                "train/E_tz",
                torch.norm(y_true[:, :, 5] - y_pred_translation[:, :, 2], p=2).item(),
                iter,
            )
            tensorboard_writer.add_scalar(
                "train/E_rx",
                torch.norm(y_true[:, :, 0] - y_pred_angles[:, :, 0], p=2).item(),
                iter,
            )
            tensorboard_writer.add_scalar(
                "train/E_ry",
                torch.norm(y_true[:, :, 1] - y_pred_angles[:, :, 1], p=2).item(),
                iter,
            )
            tensorboard_writer.add_scalar(
                "train/E_rz",
                torch.norm(y_true[:, :, 2] - y_pred_angles[:, :, 2], p=2).item(),
                iter,
            )
            tensorboard_writer.add_scalar("train/T_loss", t_loss.item(), iter)
            tensorboard_writer.add_scalar("train/R_loss", rot_loss.item(), iter)
            tensorboard_writer.add_scalar("train/total_loss", loss.item(), iter)

        return loss


class NormalizedDistanceLoss(nn.Module):
    """
    Calculates the normalized distance loss for monocular visual odometry.

    Args:
        window_size (int): Used to reshape tensors for weighted loss calculation.
        lambda_rot (float, optional): Weight for the angle loss.
        lambda_t (float, optional): Weight for the translation loss.
    """

    def __init__(
        self, lambda_t: float = 1.0, lambda_rot: float = 1.0, window_size: int = 3
    ):

        super(NormalizedDistanceLoss, self).__init__()
        self.window_size = window_size
        self.lambda_rot = float(lambda_rot)
        self.lambda_t = float(lambda_t)

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        tensorboard_writer: SummaryWriter = None,
        iter: int = None,
    ) -> torch.Tensor:
        """
        Forward pass to calculate the normalized distance loss.

        Args:
            y_pred (torch.Tensor): Tensor with dimention [batch_size x (window_size - 1) * 6]
            y_true (torch.Tensor): Tensor with dimention [batch_size x (window_size - 1) * 6]
            tensorboard_writer (SummaryWriter): Tensorboard writer to log intermediate losses
            iter (int): Iteration counter to save in Tensorboard
        """
        # Separate angles and translation for ground truth
        y_true = torch.reshape(y_true, (y_true.shape[0], self.window_size - 1, 6))
        target_rot = y_true[:, :, :3].flatten()
        target_t = y_true[:, :, 3:].flatten()

        # Separate angles and translation for predicted
        y_pred_angles, y_pred_translation = y_pred
        y_pred_angles = torch.reshape(
            y_pred_angles, (y_pred_angles.shape[0], self.window_size - 1, 3)
        )
        y_pred_translation = torch.reshape(
            y_pred_translation, (y_pred_translation.shape[0], self.window_size - 1, 3)
        )
        pred_rot = y_pred_angles.flatten()
        pred_t = y_pred_translation.flatten()

        # Calculate translation loss with normalization
        eps = 1e-6
        pred_t_loss = pred_t / max(torch.linalg.vector_norm(pred_t), eps)
        target_t_loss = target_t / max(torch.linalg.vector_norm(target_t), eps)
        t_loss = torch.linalg.vector_norm(pred_t_loss - target_t_loss)

        # Calculate rotation loss
        rot_loss = torch.linalg.vector_norm(pred_rot - target_rot)
        # rot_loss = self.lambda_rot * torch.mean(torch.norm(rot_diff, dim=1))

        # Combined loss
        loss = self.lambda_t * t_loss + self.lambda_rot * rot_loss

        # Log intermediate losses in TensorBoard (only training)
        if tensorboard_writer:
            tensorboard_writer.add_scalar(
                "train/E_tx",
                torch.norm(y_true[:, :, 3] - y_pred_translation[:, :, 0], p=2).item(),
                iter,
            )
            tensorboard_writer.add_scalar(
                "train/E_ty",
                torch.norm(y_true[:, :, 4] - y_pred_translation[:, :, 1], p=2).item(),
                iter,
            )
            tensorboard_writer.add_scalar(
                "train/E_tz",
                torch.norm(y_true[:, :, 5] - y_pred_translation[:, :, 2], p=2).item(),
                iter,
            )
            tensorboard_writer.add_scalar(
                "train/E_rx",
                torch.norm(y_true[:, :, 0] - y_pred_angles[:, :, 0], p=2).item(),
                iter,
            )
            tensorboard_writer.add_scalar(
                "train/E_ry",
                torch.norm(y_true[:, :, 1] - y_pred_angles[:, :, 1], p=2).item(),
                iter,
            )
            tensorboard_writer.add_scalar(
                "train/E_rz",
                torch.norm(y_true[:, :, 2] - y_pred_angles[:, :, 2], p=2).item(),
                iter,
            )
            tensorboard_writer.add_scalar("train/T_loss", t_loss.item(), iter)
            tensorboard_writer.add_scalar("train/R_loss", rot_loss.item(), iter)
            tensorboard_writer.add_scalar("train/total_loss", loss.item(), iter)

        return loss


class ALoss(nn.Module):
    def __init__(self, epsilon: float = 1.5, c: float = 0.1, window_size: int = 3):
        """
        A-Loss implementation.
        Args:
            epsilon (float): Robustness parameter (controls sensitivity to outliers).
            c (float): Scale parameter.
        """
        super(ALoss, self).__init__()
        self.epsilon = epsilon
        self.c = c
        self.window_size = window_size

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        tensorboard_writer: SummaryWriter = None,
        iter: int = None,
    ) -> torch.Tensor:
        """
        Compute A-Loss.

        Args:
            y_pred (torch.Tensor): Tensor with dimention [batch_size x (window_size - 1) * 6]
            y_true (torch.Tensor): Tensor with dimention [batch_size x (window_size - 1) * 6]
            tensorboard_writer (SummaryWriter): Tensorboard writer to log intermediate losses
            iter (int): Iteration counter to save in Tensorboard

        Returns:
            torch.Tensor: A-Loss value.
        """
        epsilon = self.epsilon
        c = self.c
        b_size = y_true.shape[0]

        # |ε - 2| / ε
        robustness_term = abs(epsilon - 2) / epsilon

        # Process angles and translation to y_true format
        y_pred_angles, y_pred_translation = y_pred
        y_pred_angles = torch.reshape(y_pred_angles, (b_size, self.window_size - 1, 3))
        y_pred_translation = torch.reshape(
            y_pred_translation, (b_size, self.window_size - 1, 3)
        )
        y_pred = torch.cat((y_pred_angles, y_pred_translation), dim=1)
        y_pred = torch.reshape(y_pred, (b_size, (self.window_size - 1) * 6))

        # Loss computation
        x = y_true - y_pred
        loss = robustness_term * (
            ((x / c) ** 2 / abs(epsilon - 2) + 1) ** (epsilon / 2) - 1
        )
        # Mean loss across the batch
        loss = loss.mean()

        # Log intermediate losses in TensorBoard (only training)
        if tensorboard_writer:
            tensorboard_writer.add_scalar("train/norm_x", abs(x.mean().item()), iter)
            tensorboard_writer.add_scalar("train/total_loss", loss.item(), iter)

        return loss


class WeightedALoss(nn.Module):
    """
    Computes a weighted A-loss for angle and translation components.

    Args:
        window_size (int): Used to reshape tensors for weighted loss calculation.
        epsilon (float): Robustness parameter (controls sensitivity to outliers).
        c (float): Scale parameter.
        lambda_rot (float, optional): Weight for the angle loss.
        lambda_t (float, optional): Weight for the translation loss.
    """

    def __init__(
        self,
        window_size: int = 3,
        epsilon: float = 1.5,
        c: float = 0.1,
        lambda_t: float = 1.0,
        lambda_rot: float = 1.0,
    ):
        super(WeightedALoss, self).__init__()
        self.window_size = window_size
        self.lambda_rot = float(lambda_rot)
        self.lambda_t = float(lambda_t)
        self.mse_loss = nn.MSELoss()
        self.epsilon = epsilon
        self.c = c

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        tensorboard_writer: SummaryWriter = None,
        iter: int = None,
    ) -> torch.Tensor:
        """
        Forward pass to calculate the weighted MSE loss.

        Args:
            y_pred (torch.Tensor): Tensor with dimention [batch_size x (window_size - 1) * 6]
            y_true (torch.Tensor): Tensor with dimention [batch_size x (window_size - 1) * 6]
            tensorboard_writer (SummaryWriter): Tensorboard writer to log intermediate losses
            iter (int): Iteration counter to save in Tensorboard
        """
        epsilon = self.epsilon
        c = self.c

        # |ε - 2| / ε
        robustness_term = abs(epsilon - 2) / epsilon

        # Separate angles and translation for ground truth
        y_true = torch.reshape(y_true, (y_true.shape[0], self.window_size - 1, 6))
        gt_angles = y_true[:, :, :3].flatten()
        gt_translation = y_true[:, :, 3:].flatten()

        # Separate angles and translation for predicted
        y_pred_angles, y_pred_translation = y_pred
        y_pred_angles = torch.reshape(
            y_pred_angles, (y_pred_angles.shape[0], self.window_size - 1, 3)
        )
        y_pred_translation = torch.reshape(
            y_pred_translation, (y_pred_translation.shape[0], self.window_size - 1, 3)
        )
        estimated_angles = y_pred_angles.flatten()
        estimated_translation = y_pred_translation.flatten()

        # Calculate weighted losses for angles and translation
        x1 = estimated_angles - gt_angles.float()
        rot_loss = robustness_term * (
            ((x1 / c) ** 2 / abs(epsilon - 2) + 1) ** (epsilon / 2) - 1
        )
        rot_loss = rot_loss.mean()

        x2 = estimated_translation - gt_translation.float()
        t_loss = robustness_term * (
            ((x2 / c) ** 2 / abs(epsilon - 2) + 1) ** (epsilon / 2) - 1
        )
        t_loss = t_loss.mean()

        # Compute final weighted loss
        loss = self.lambda_t * t_loss + self.lambda_rot * rot_loss

        # Log intermediate losses in TensorBoard (only training)
        if tensorboard_writer:
            tensorboard_writer.add_scalar(
                "train/E_tx",
                torch.norm(y_true[:, :, 3] - y_pred_translation[:, :, 0], p=2).item(),
                iter,
            )
            tensorboard_writer.add_scalar(
                "train/E_ty",
                torch.norm(y_true[:, :, 4] - y_pred_translation[:, :, 1], p=2).item(),
                iter,
            )
            tensorboard_writer.add_scalar(
                "train/E_tz",
                torch.norm(y_true[:, :, 5] - y_pred_translation[:, :, 2], p=2).item(),
                iter,
            )
            tensorboard_writer.add_scalar(
                "train/E_rx",
                torch.norm(y_true[:, :, 0] - y_pred_angles[:, :, 0], p=2).item(),
                iter,
            )
            tensorboard_writer.add_scalar(
                "train/E_ry",
                torch.norm(y_true[:, :, 1] - y_pred_angles[:, :, 1], p=2).item(),
                iter,
            )
            tensorboard_writer.add_scalar(
                "train/E_rz",
                torch.norm(y_true[:, :, 2] - y_pred_angles[:, :, 2], p=2).item(),
                iter,
            )
            tensorboard_writer.add_scalar("train/T_loss", t_loss.item(), iter)
            tensorboard_writer.add_scalar("train/R_loss", rot_loss.item(), iter)
            tensorboard_writer.add_scalar("train/total_loss", loss.item(), iter)

        return loss


class CameraPoseLoss(nn.Module):
    """
    A class to represent camera pose loss
    [Source]: https://github.com/yolish/multi-scene-pose-transformer
    """

    def __init__(
        self,
        learnable: bool = True,
        s_x: float = 0.0,
        s_q: float = -3.0,
        s_d: float = 0.0,
        norm: int = 2,
        window_size: int = 3,
        device: torch.device = "cuda",
    ):
        """
        Homoscedastic uncertainty weighting (Kendall et al., 2018) for the
        multi-task objective. Three learnable log-variances balance the tasks:
        translation (s_x), rotation (s_q) and monocular depth (s_d). The depth
        term is applied by ``weight_depth`` so the pose and depth losses can be
        computed by separate criteria and still share the same Kendall scheme.

        Args:
            learnable (bool): if True, s_x/s_q/s_d are trainable parameters.
            s_x (float): initial log-variance for translation.
            s_q (float): initial log-variance for rotation.
            s_d (float): initial log-variance for depth.
            norm (int):
        """
        super(CameraPoseLoss, self).__init__()
        self.learnable = learnable
        self.s_x = nn.Parameter(
            torch.tensor(s_x, device=device), requires_grad=self.learnable
        )
        self.s_q = nn.Parameter(
            torch.tensor(s_q, device=device), requires_grad=self.learnable
        )
        self.s_d = nn.Parameter(
            torch.tensor(s_d, device=device), requires_grad=self.learnable
        )
        self.norm = norm
        self.window_size = window_size

    def weight_depth(
        self,
        depth_loss: torch.Tensor,
        tensorboard_writer: SummaryWriter = None,
        iter: int = None,
    ) -> torch.Tensor:
        """
        Applies the homoscedastic uncertainty weighting to the depth loss,
        completing the 3-task Kendall objective:
            L_depth * exp(-s_d) + s_d.

        Args:
            depth_loss (torch.Tensor): the raw depth loss from the depth criterion.
            tensorboard_writer (SummaryWriter): optional logger.
            iter (int): iteration counter for logging.
        """
        if self.learnable:
            weighted = depth_loss * torch.exp(-self.s_d) + self.s_d
        else:
            weighted = self.s_d * depth_loss

        if tensorboard_writer:
            tensorboard_writer.add_scalar("train/s_d", self.s_d, iter)
            tensorboard_writer.add_scalar("train/D_loss", depth_loss.item(), iter)

        return weighted

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        tensorboard_writer: SummaryWriter = None,
        iter: int = None,
    ) -> torch.Tensor:
        """
        Forward pass to calculate the Camera Pose loss loss.

        Args:
            y_pred (torch.Tensor): Tensor with dimention [batch_size x (window_size - 1) * 6]
            y_true (torch.Tensor): Tensor with dimention [batch_size x (window_size - 1) * 6]
            tensorboard_writer (SummaryWriter): Tensorboard writer to log intermediate losses
            iter (int): Iteration counter to save in Tensorboard
        """
        # Separate angles and translation for ground truth
        y_true = torch.reshape(y_true, (y_true.shape[0], self.window_size - 1, 6))
        y_true_angles = y_true[:, :, :3]
        y_true_t = y_true[:, :, 3:]

        # Separate angles and translation for predicted
        y_pred_angles, y_pred_t = y_pred
        y_pred_angles = torch.reshape(
            y_pred_angles, (y_pred_angles.shape[0], self.window_size - 1, 3)
        )
        y_pred_t = torch.reshape(y_pred_t, (y_pred_t.shape[0], self.window_size - 1, 3))

        # Position loss and orientation loss
        # t_loss = torch.norm(y_true_t - y_pred_t, p=self.norm)
        # rot_loss = torch.norm(
        #     F.normalize(y_true_angles, p=2) - F.normalize(y_pred_angles, p=2),
        #     p=self.norm,
        # )

        # Position loss and orientation loss
        t_loss = F.mse_loss(y_true_t, y_pred_t)
        rot_loss = F.mse_loss(y_true_angles, y_pred_angles)

        # Compute final loss
        if self.learnable:
            loss = (
                t_loss * torch.exp(-self.s_x)
                + self.s_x
                + rot_loss * torch.exp(-self.s_q)
                + self.s_q
            )
        else:
            loss = self.s_x * t_loss + self.s_q * rot_loss

        # Log intermediate losses in TensorBoard (only training)
        if tensorboard_writer:
            tensorboard_writer.add_scalar(
                "train/E_tx",
                torch.norm(y_true[:, :, 3] - y_pred_t[:, :, 0], p=2).item(),
                iter,
            )
            tensorboard_writer.add_scalar(
                "train/E_ty",
                torch.norm(y_true[:, :, 4] - y_pred_t[:, :, 1], p=2).item(),
                iter,
            )
            tensorboard_writer.add_scalar(
                "train/E_tz",
                torch.norm(y_true[:, :, 5] - y_pred_t[:, :, 2], p=2).item(),
                iter,
            )
            tensorboard_writer.add_scalar(
                "train/E_rx",
                torch.norm(y_true[:, :, 0] - y_pred_angles[:, :, 0], p=2).item(),
                iter,
            )
            tensorboard_writer.add_scalar(
                "train/E_ry",
                torch.norm(y_true[:, :, 1] - y_pred_angles[:, :, 1], p=2).item(),
                iter,
            )
            tensorboard_writer.add_scalar(
                "train/E_rz",
                torch.norm(y_true[:, :, 2] - y_pred_angles[:, :, 2], p=2).item(),
                iter,
            )
            tensorboard_writer.add_scalar("train/s_x", self.s_x, iter)
            tensorboard_writer.add_scalar("train/s_q", self.s_q, iter)
            tensorboard_writer.add_scalar("train/T_loss", t_loss.item(), iter)
            tensorboard_writer.add_scalar("train/R_loss", rot_loss.item(), iter)
            tensorboard_writer.add_scalar("train/total_loss", loss.item(), iter)

        return loss


def get_pose_criterion(
    loss_name: str,
    window_size: int = 3,
    lambda_t: float = 1.0,
    lambda_rot: float = 1.0,
    device: torch.device = "cuda",
) -> nn.Module:
    """
    Instantiates and returns a loss function based on the provided loss name.

    Args:
        loss_name (str): The name of the loss function. Options are 'mse' for standard MSE loss
                         and 'weighted_mse' for the custom weighted MSE loss.
        window_size (int, optional): Used to reshape tensors in the 'weighted_mse' loss.
        lambda_rot (float, optional): Weight for the angle loss.
        lambda_t (float, optional): Weight for the translation loss.

    Returns:
        nn.Module: An instance of the specified loss function.
    """
    if loss_name.lower() == "mse":
        return nn.MSELoss()  # Unweighted MSE loss

    elif loss_name.lower() == "w_mse":
        return WeightedMSELoss(
            window_size=window_size, lambda_t=lambda_t, lambda_rot=lambda_rot
        )

    elif loss_name.lower() == "norm_dist":
        return NormalizedDistanceLoss(
            window_size=window_size, lambda_t=lambda_t, lambda_rot=lambda_rot
        )

    elif loss_name.lower() == "a_loss":
        return ALoss(window_size=window_size, epsilon=1.5, c=0.1)

    elif loss_name.lower() == "w_aloss":
        return WeightedALoss(
            window_size=window_size,
            epsilon=1.5,
            c=0.1,
            lambda_t=lambda_t,
            lambda_rot=lambda_rot,
        )

    elif loss_name.lower() == "camera_pose_loss":
        return CameraPoseLoss(
            learnable=True,
            s_x=lambda_t,
            s_q=lambda_rot,
            norm=2,
            window_size=window_size,
            device=device,
        )

    else:
        raise ValueError(
            f"Unsupported loss name: {loss_name}. Available options: 'mse', 'weighted_mse'."
        )


if __name__ == "__main__":
    # Define dummy inputs for testing
    batch_size = 4
    window_size = 3
    lambda_t = 1.0
    lambda_rot = 2.0

    # Dummy ground truth and predictions
    y_true = torch.rand(batch_size, (window_size - 1) * 6)
    y_pred = torch.rand(batch_size, (window_size - 1) * 6)

    # Test MSELoss
    mse_loss_fn = get_criterion("mse")
    mse_loss = mse_loss_fn(y_pred, y_true)
    print("MSE Loss:", mse_loss.item())

    # Test WeightedMSELoss with a weight for angle loss
    weighted_loss_fn = get_criterion("w_mse", window_size=window_size, lambda_t=2)
    weighted_loss = weighted_loss_fn(y_pred, y_true)
    print("Weighted MSE Loss:", weighted_loss.item())

    # Test NormalizedDistanceLoss with a weight for angle loss
    norm_dist_loss_fn = get_criterion(
        "norm_dist", window_size=window_size, lambda_rot=2
    )
    weighted_loss = norm_dist_loss_fn(y_pred, y_true)
    print("Weighted MSE Loss:", weighted_loss.item())
