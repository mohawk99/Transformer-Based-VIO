import argparse
import os
import json
import logging
from sklearn.model_selection import train_test_split
from util import utils
from models.IMUTransformerEncoder import IMUTransformerEncoder
from models.IMUCLSBaseline import IMUCLSBaseline
from util.IMU_NU import SynchronizedDataset
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
import random


def load_and_pair_files(folder_path):
    """
    Load and pair IMU and pose JSON files from a folder based on their naming convention.

    Args:
        folder_path (str): Path to the folder containing IMU and pose JSON files.

    Returns:
        list: A list of tuples where each tuple contains (IMU data, Pose data).
    """
    imu_files = [f for f in os.listdir(folder_path) if "_ms_imu.json" in f]
    pose_files = [f for f in os.listdir(folder_path) if "_pose.json" in f]

    paired_files = []
    for imu_file in imu_files:
        # Extract the scene identifier
        scene_id = imu_file.replace("_ms_imu.json", "")
        # Find the corresponding pose file
        pose_file = f"{scene_id}_pose.json"
        if pose_file in pose_files:
            # Load the JSON data
            with open(os.path.join(folder_path, imu_file), 'r') as imu_f:
                imu_data = json.load(imu_f)
            with open(os.path.join(folder_path, pose_file), 'r') as pose_f:
                pose_data = json.load(pose_f)
            paired_files.append((imu_data, pose_data))
        else:
            logging.warning(f"No matching pose file for IMU file: {imu_file}")

    if not paired_files:
        raise ValueError("No paired IMU and pose files found.")
    logging.info(f"Found {len(paired_files)} paired files.")
    return paired_files


def create_dataset(paired_data, window_size, window_shift):
    """
    Prepares the dataset from paired IMU and pose data.

    Args:
        paired_data (list): List of paired IMU and pose data.
        window_size (int): Size of the data window.
        window_shift (int): Shift between consecutive windows.

    Returns:
        SynchronizedDataset: Prepared dataset object.
    """
    imu_samples = [sample[0] for sample in paired_data]  # Extract IMU data
    pose_samples = [sample[1] for sample in paired_data]  # Extract pose data

    # Flatten the data if it's wrapped in dictionaries
    imu_samples = [item for sublist in imu_samples for item in sublist]
    pose_samples = [item for sublist in pose_samples for item in sublist]

    return SynchronizedDataset(imu_samples, pose_samples, window_size, window_shift)



def evaluate_model(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for minibatch in dataloader:
            imu_window = minibatch["imu"]["data"].to(device)
            pose_window = minibatch["pose"]["pose_and_orientation"].to(device)
            predictions = model({"data": imu_window})
            loss = loss_fn(predictions, pose_window)
            total_loss += loss.item()
    return total_loss / len(dataloader)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("mode", help="train or test")
    arg_parser.add_argument("dataset_folder", help="path to the folder containing IMU and pose JSON files")
    arg_parser.add_argument("--checkpoint_path", help="path to a pre-trained model")
    arg_parser.add_argument("--experiment", help="a short string to describe the experiment/commit used")
    args = arg_parser.parse_args()

    # Configure logging
    utils.init_logger()
    logging.info("Starting {} mode".format(args.mode))

    # Load configuration
    with open('config.json', "r") as read_file:
        config = json.load(read_file)

    # Set random seed for reproducibility
    torch.manual_seed(config.get('torch_seed', 0))
    np.random.seed(config.get('numpy_seed', 0))

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and pair files
    paired_files = load_and_pair_files(args.dataset_folder)
    #paired_files = random.sample(paired_files, int(len(paired_files) * 0.1))

    # Split data into train, val, and test sets
    train_val, test = train_test_split(paired_files, test_size=0.2, random_state=42)
    train, val = train_test_split(train_val, test_size=0.2, random_state=42)
    logging.info(f"Dataset split into {len(train)} train, {len(val)} val, {len(test)} test samples")

    # Dataset parameters
    window_size = config.get("window_size", 10)
    window_shift = config.get("window_shift", 5)

    if args.mode == "train":
        train_dataset = create_dataset(train, window_size, window_shift)
        val_dataset = create_dataset(val, window_size, window_shift)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.get("batch_size", 32),
            shuffle=True,
            num_workers=config.get("n_workers", 4),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.get("batch_size", 32),
            shuffle=False,
            num_workers=config.get("n_workers", 4),
        )

        # Initialize model
        model = IMUTransformerEncoder(config).to(device)

        # Load checkpoint if provided
        if args.checkpoint_path:
            model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
            logging.info(f"Model loaded from checkpoint: {args.checkpoint_path}")

        # Define loss function, optimizer, and scheduler
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config.get("lr"),
                                     eps=config.get("eps"),
                                     weight_decay=config.get("weight_decay"))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=config.get("lr_scheduler_step_size"),
                                                    gamma=config.get("lr_scheduler_gamma"))

        # Training loop
        n_epochs = config.get("n_epochs", 10)
        checkpoint_prefix = utils.create_output_dir("out") + "/" + utils.get_stamp_from_log()
        logging.info("Starting training...")
        for epoch in range(n_epochs):
            model.train()
            epoch_loss = 0.0
            for minibatch in train_loader:
                imu_window = minibatch["imu"]["data"].to(device)
                pose_window = minibatch["pose"]["pose_and_orientation"].to(device)

                # Forward pass
                optimizer.zero_grad()
                res = model({"data": imu_window})
                loss = loss_fn(res, pose_window)

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # Calculate average training loss
            avg_train_loss = epoch_loss / len(train_loader)

            # Calculate validation loss
            avg_val_loss = evaluate_model(model, val_loader, loss_fn, device)

            # Log losses
            logging.info(f"Epoch {epoch+1}/{n_epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

            # Save checkpoint
            if (epoch + 1) % config.get("n_freq_checkpoint", 5) == 0:
                checkpoint_path = f"{checkpoint_prefix}_epoch{epoch+1}.pth"
                torch.save(model.state_dict(), checkpoint_path)
                logging.info(f"Checkpoint saved at {checkpoint_path}")

            scheduler.step()

        # Save final model
        final_checkpoint_path = f"{checkpoint_prefix}_final.pth"
        torch.save(model.state_dict(), final_checkpoint_path)
        logging.info(f"Final model saved at {final_checkpoint_path}")

    elif args.mode == "test":
        test_dataset = create_dataset(test, window_size, window_shift)
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=config.get("n_workers", 4),
        )

        # Initialize model
        model = IMUTransformerEncoder(config).to(device)
        if args.checkpoint_path:
            model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
            logging.info(f"Model loaded from checkpoint: {args.checkpoint_path}")

        # Test the model
        test_loss = evaluate_model(model, test_loader, torch.nn.MSELoss(), device)
        logging.info(f"Test Loss: {test_loss:.4f}")
