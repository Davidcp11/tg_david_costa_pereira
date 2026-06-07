from typing import Callable, Dict, Optional, Any, Union

from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

from utils.train_utils import save_checkpoint


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        training_params: Dict[str, Any],
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        checkpoint_params: Dict[str, Any],
        device: Union[torch.device, str],
    ):
        self.training_params = training_params
        self.checkpoint_params = checkpoint_params
        self.device = device
        self.model = model.to(self.device)
        self.pose_criterion, self.depth_criterion = criterion
        # When False, the depth branch is not supervised: the encoder is shaped
        # purely by the pose loss (pose-only TSformer-VO baseline). The model still
        # produces depth_pred in the forward pass, but it is ignored.
        self.use_depth = training_params.get("use_depth", True)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.tensorboard_writer = SummaryWriter(
            log_dir=self.checkpoint_params.get("checkpoint_dpath", "checkpoints/test")
        )

    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        epoch: int,
    ) -> float:
        self.model.train()
        epoch_loss = 0
        iter = (epoch - 1) * len(train_loader) + 1

        with tqdm(train_loader, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                if self.use_depth:
                    images, y_true, depth_true = batch
                    depth_true = depth_true.to(self.device)
                else:
                    images, y_true = batch
                images = images.to(self.device).float()
                y_true = y_true.to(self.device).float()

                # Predict pose (and depth)
                y_pred, depth_pred = self.model(images)

                # Compute loss
                pose_loss = self.pose_criterion(
                    y_pred,
                    y_true,
                    tensorboard_writer=self.tensorboard_writer,
                    iter=iter,
                )
                if self.use_depth:
                    depth_loss = self.depth_criterion(
                        depth_pred,
                        depth_true,
                        tensorboard_writer=self.tensorboard_writer,
                        iter=iter,
                    )
                    # 3-task homoscedastic uncertainty weighting (Kendall 2018):
                    # the depth loss is weighted by the learnable s_d when the pose
                    # criterion supports it; otherwise it is summed with weight 1.
                    if hasattr(self.pose_criterion, "weight_depth"):
                        depth_loss = self.pose_criterion.weight_depth(
                            depth_loss,
                            tensorboard_writer=self.tensorboard_writer,
                            iter=iter,
                        )
                    loss = pose_loss + depth_loss
                else:
                    loss = pose_loss

                # Compute gradient and do optimizer step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())

                iter += 1

        return epoch_loss / len(train_loader)

    def val_epoch(
        self,
        val_loader: Optional[torch.utils.data.DataLoader],
    ) -> float:
        if val_loader is None:
            return float("inf")

        self.model.eval()
        epoch_loss = 0

        with torch.no_grad():
            with tqdm(val_loader, unit="batch") as tepoch:
                for batch in tepoch:
                    tepoch.set_description("Validating")

                    if self.use_depth:
                        images, y_true, depth_true = batch
                        depth_true = depth_true.to(self.device)
                    else:
                        images, y_true = batch
                    images = images.to(self.device)
                    y_true = y_true.to(self.device)

                    # Predict pose (and depth)
                    y_pred, depth_pred = self.model(images.float())

                    # Compute loss
                    pose_loss = self.pose_criterion(y_pred, y_true)
                    if self.use_depth:
                        depth_loss = self.depth_criterion(depth_pred, depth_true)
                        if hasattr(self.pose_criterion, "weight_depth"):
                            depth_loss = self.pose_criterion.weight_depth(depth_loss)
                        loss = pose_loss + depth_loss
                    else:
                        loss = pose_loss

                    epoch_loss += loss.item()
                    tepoch.set_postfix(val_loss=loss.item())

        self.model.train()
        return epoch_loss / len(val_loader)

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader],
    ) -> None:
        best_val = self.training_params.get("best_val", float("inf"))
        save_best = False

        for epoch in range(
            self.training_params.get("epoch_init", 1),
            self.training_params.get("epoch", 100),
        ):
            # Training for one epoch
            train_loss = self.train_epoch(train_loader, epoch)

            # Validate model
            val_loss = self.val_epoch(val_loader)
            print(
                f"Epoch: {epoch} -- loss: {train_loss:.4f} -- val_loss: {val_loss:.4f} -- lr: {self.optimizer.param_groups[-1]['lr'] * 1e4:.6f} e-4"
            )

            if val_loss < best_val:
                print(
                    f"Saving new best model -- loss decreased from {best_val:.6f} to {val_loss:.6f}"
                )
                best_val = val_loss
                save_best = True

            # Log train and validation loss in TensorBoard
            self.tensorboard_writer.add_scalar("loss/train", train_loss, epoch)
            self.tensorboard_writer.add_scalar("loss/val", val_loss, epoch)

            # Create checkpoint dict and save
            checkpoint_dict = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": (
                    self.scheduler.state_dict() if self.scheduler else {}
                ),
                "best_val": best_val,
                "criterion_pose_state_dict": self.pose_criterion.state_dict(),
                "criterion_depth_state_dict": self.depth_criterion.state_dict(),
            }

            save_checkpoint(
                checkpoint_dict,
                self.checkpoint_params.get("checkpoint_dpath", "checkpoints/test"),
                epoch,
                save_best,
                save_interval=10,
            )
            save_best = False

            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()
