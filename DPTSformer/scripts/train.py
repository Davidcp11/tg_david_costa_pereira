import torch

from models.build_model import build_model
from datasets.dataloader import build_dataloader
from scripts.trainer import Trainer
from models.loss_pose import get_pose_criterion
from models.loss_depth import get_depth_criterion
from utils.train_utils import get_optimizer, get_scheduler
from utils.checkpoint_utils import update_state
from utils.config_utils import load_config

torch.cuda.empty_cache()


def main(config_dpath: str):
    # Load configuration from the JSON file
    config = load_config(config_dpath)

    # Extract configuration parameters
    training_params = config.get("training", {})
    checkpoint_params = config.get("checkpoint", {})

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # Load data
    print("Loading data...")
    train_loader, val_loader = build_dataloader(
        data_params=config.get("data", {}), split="train"
    )

    # Build model
    print("Building model...")
    model = build_model(config.get("model", {}), device)

    # Loss and optimizer
    pose_criterion = get_pose_criterion(
        loss_name=training_params.get("pose_loss", "w_mse"),
        window_size=config["data"]["window_size"],
        lambda_t=training_params.get("lambda_t", 1.0),
        lambda_rot=training_params.get("lambda_rot", 1.0),
        device=device,
    )
    depth_criterion = get_depth_criterion(
        loss_name=training_params.get("depth_loss", "ssi_loss"),
        alpha=training_params.get("ssi_alpha", 0.5),
        bbox_norm=training_params.get("depth_bbox_norm", None),
    )
    optimizer = get_optimizer(
        list(model.parameters()) + list(pose_criterion.parameters()),
        config.get("optimizer", {}),
    )
    scheduler = get_scheduler(
        optimizer,
        config.get("scheduler", {}),
        epochs=training_params.get("epoch", 100),
        steps_per_epoch=len(train_loader),
    )

    # Update state dict
    model, optimizer, scheduler, training_params = update_state(
        model,
        optimizer,
        scheduler,
        [pose_criterion, depth_criterion],
        checkpoint_params,
        training_params,
    )

    # Initialize Trainer and start training
    trainer = Trainer(
        model=model,
        training_params=training_params,
        criterion=[pose_criterion, depth_criterion],
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_params=checkpoint_params,
        device=device,
    )

    print(20 * "--" + " Training " + 20 * "--")
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="This script trains a model based on the provided configuration path. "
        "Ensure that the configuration file includes model parameters, dataset paths, and training settings."
        "Usage: python -m training.main --config <config_dpath>"
    )

    # Add a required --config argument
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file (e.g., --config configs/exp1.json)",
    )

    # Parse the arguments and get config file path
    args = parser.parse_args()
    config_dpath = args.config
    if not config_dpath:
        print("Please provide a configuration file using --config")
        sys.exit(1)

    main(config_dpath)
