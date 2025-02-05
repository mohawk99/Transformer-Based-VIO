import argparse
import os
import json
import logging
from sklearn.model_selection import train_test_split
from util import utils
from models.IMUModel2 import IMUTransformer, IMUTransformerLoss
import torch
import numpy as np
from torch.utils.data import DataLoader
from util.IMU_NU import *
from sklearn.metrics import mean_squared_error
import random
from sklearn.preprocessing import StandardScaler
import itertools

def train_epoch(model, train_loader, loss_fn, optimizer, device, grad_clip):
    model.train()
    accumulated_loss = 0.0
    accumulated_pos_loss = 0.0
    accumulated_ori_loss = 0.0

    for minibatch in train_loader:
        acc_data = minibatch["imu"]["acc"].to(device)
        gyro_data = minibatch["imu"]["gyro"].to(device)
        pose_window = minibatch["pose"]["pose_and_orientation"].to(device)

        optimizer.zero_grad()
        output = model(acc_data, gyro_data)
        output = output.squeeze(1)

        batch_loss, pos_loss, ori_loss = loss_fn(output, pose_window)
        batch_loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        accumulated_loss += batch_loss.item()
        accumulated_pos_loss += pos_loss.item()
        accumulated_ori_loss += ori_loss.item()

    return accumulated_loss/len(train_loader), accumulated_pos_loss/len(train_loader), accumulated_ori_loss/len(train_loader)

def evaluate_model(model, val_loader, loss_fn, device):
   model.eval()
   total_loss = 0.0
   total_pos_loss = 0.0
   total_ori_loss = 0.0

   with torch.no_grad():
       for minibatch in val_loader:
           acc_data = minibatch["imu"]["acc"].to(device)  
           gyro_data = minibatch["imu"]["gyro"].to(device)
           pose_window = minibatch["pose"]["pose_and_orientation"].to(device)

           output = model(acc_data, gyro_data)
           output = output.squeeze(1)

           loss, pos_loss, ori_loss = loss_fn(output, pose_window)
           
           total_loss += loss.item()
           total_pos_loss += pos_loss.item()
           total_ori_loss += ori_loss.item()

   num_batches = len(val_loader)
   return total_loss/num_batches, total_pos_loss/num_batches, total_ori_loss/num_batches





def grid_search(args, base_config):
    param_grid = {
        'hidden_dim': [128, 256, 512],
        'num_encoder_layers': [2, 4],
        'lr': [1e-3, 1e-4],
        'batch_size': [16, 32, 64]
    }
    
    configs = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    
    for i, exp_config in enumerate(configs):
        try:
            config = base_config.copy()
            config.update(exp_config)
            
            # Create experiment directory
            exp_dir = f"exp_{i}"
            os.makedirs(exp_dir, exist_ok=True)
            
            # Log experiment details
            logging.info(f"\n{'='*50}")
            logging.info(f"Starting experiment {i+1}/{len(configs)}")
            logging.info(f"Configuration parameters:")
            for key, value in exp_config.items():
                logging.info(f"{key}: {value}")
                
            # Update paths for this experiment
            args.checkpoint_path = f"{exp_dir}/best_model.pth"
            
            # Create new data loaders with updated batch size
            train_loader = DataLoader(train, batch_size=config["batch_size"], shuffle=True)
            val_loader = DataLoader(val, batch_size=config["batch_size"], shuffle=False)
            
            # Train using existing training code
            model = IMUTransformer(config).to(device)
            loss_fn = IMUTransformerLoss().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                      step_size=config["lr_scheduler_step_size"],
                                                      gamma=config["lr_scheduler_gamma"])
            grad_clip = config["grad_clip"]

            for epoch in range(config["n_epochs"]):
                train_loss, train_pos_loss, train_ori_loss = train_epoch(
                    model, train_loader, loss_fn, optimizer, device, grad_clip)
                val_loss, val_pos_loss, val_ori_loss = evaluate_model(
                    model, val_loader, loss_fn, device)
                
                # Log using existing format
                logging.info(
                    f"Epoch {epoch + 1}/{n_epochs}: "
                    f"Train(Loss={float(train_loss):.4f}, P={float(train_pos_loss):.4f}, "
                    f"O={float(train_ori_loss):.4f}, "
                    f"Val(Loss={float(val_loss):.4f}, P={float(val_pos_loss):.4f}, "
                    f"O={float(val_ori_loss):.4f})"
                )
                
                scheduler.step()
                
        except Exception as e:
            logging.error(f"Error in experiment {i+1}: {str(e)}")
            continue



if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("mode", help="train or test")
    arg_parser.add_argument("dataset_folder", help="path to the folder containing IMU and pose JSON files")
    arg_parser.add_argument("--checkpoint_path", help="path to a pre-trained model")
    arg_parser.add_argument("--experiment", help="a short string to describe the experiment/commit used")
    arg_parser.add_argument("--grid_search", action="store_true", help="run grid search")
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
    random.seed(config.get('torch_seed', 0))

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load and prepare data
    window_size = config.get("window_size", 200)
    window_shift = config.get("window_shift", 10)
    paired_files = load_and_pair_files(args.dataset_folder)
    all_data = create_scene_datasets(paired_files, window_size, window_shift)
    
    # Split dataset
    train_val, test = train_test_split(all_data, test_size=0.2, random_state=42)
    train, val = train_test_split(train_val, test_size=0.2, random_state=42)
    logging.info(f"Dataset split into {len(train)} train, {len(val)} val, {len(test)} test samples")

    if args.mode == "train":
        if args.grid_search:
            grid_search(args, config)
        else:
            # Create data loaders
            train_loader = DataLoader(train, batch_size=config.get("batch_size", 32), shuffle=True)
            val_loader = DataLoader(val, batch_size=config.get("batch_size", 32), shuffle=False)

            logging.info("Training Configuration:")
            for key, value in config.items():
                logging.info(f"{key}: {value}")

            # Initialize model
            model = IMUTransformer(config).to(device)
            
            # Load checkpoint if provided
            if args.checkpoint_path:
                model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
                logging.info(f"Model loaded from checkpoint: {args.checkpoint_path}")

            # Initialize loss function, optimizer and scheduler
            loss_fn = IMUTransformerLoss()

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.get("lr", 5e-4),
                eps=config.get("eps", 1e-8),
                weight_decay=config.get("weight_decay", 1e-2)
            )
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            #     optimizer,
            #     T_0=config.get("scheduler", {}).get("T_0", 10),
            #     T_mult=config.get("scheduler", {}).get("T_mult", 2),
            #     eta_min=config.get("scheduler", {}).get("eta_min", 1e-6)
            # )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=config.get("scheduler", {}).get("factor", 0.5),
                patience=config.get("scheduler", {}).get("patience", 7),
                threshold=config.get("scheduler", {}).get("threshold", 1e-4),
                cooldown=config.get("scheduler", {}).get("cooldown", 3),
                min_lr=config.get("scheduler", {}).get("min_lr", 1e-6),
                verbose=config.get("scheduler", {}).get("verbose", False)
            )

            grad_clip = config.get("grad_clip")

            # Training loop
            n_epochs = config.get("n_epochs", 50)
            checkpoint_prefix = utils.create_output_dir("out") + "/" + utils.get_stamp_from_log()
            best_val_loss = float('inf')
            
            logging.info("Starting training...")
            for epoch in range(n_epochs):
                # Train
                train_loss, train_pos_loss, train_ori_loss= train_epoch(
                    model, train_loader, loss_fn, optimizer, device, grad_clip
                )
                
                # Validate
                val_loss, val_pos_loss, val_ori_loss= evaluate_model(
                    model, val_loader, loss_fn, device
                )
                
                # Log progress
                logging.info(
                    f"Epoch {epoch + 1}/{n_epochs}: "
                    f"Train(Loss={float(train_loss):.4f}, P={float(train_pos_loss):.4f}, "
                    f"O={float(train_ori_loss):.4f}, "
                    f"Val(Loss={float(val_loss):.4f}, P={float(val_pos_loss):.4f}, "
                    f"O={float(val_ori_loss):.4f})"
                )

                scheduler.step(val_loss)

                current_lr = optimizer.param_groups[0]['lr']
                logging.info(f"Current learning rate: {current_lr}")

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_path = f"{checkpoint_prefix}_best.pth"
                    torch.save(model.state_dict(), best_model_path)
                    logging.info(f"New best model saved at {best_model_path}")

                # Regular checkpoint saving
                if (epoch + 1) % config.get("n_freq_checkpoint", 10) == 0:
                    checkpoint_path = f"{checkpoint_prefix}_epoch{epoch + 1}.pth"
                    torch.save(model.state_dict(), checkpoint_path)
                    logging.info(f"Checkpoint saved at {checkpoint_path}")

                

            # Save final model
            final_checkpoint_path = f"{checkpoint_prefix}_final.pth"
            torch.save(model.state_dict(), final_checkpoint_path)
            logging.info(f"Final model saved at {final_checkpoint_path}")

    elif args.mode == "test":
        test_loader = DataLoader(test, batch_size=config.get("batch_size", 32), shuffle=False)
        
        # Initialize model
        model = IMUTransformer(config).to(device)
        if not args.checkpoint_path:
            raise ValueError("Checkpoint path must be provided for testing")
        
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
        model.eval()
        
        # Test the model
        loss_fn = IMUTransformerLoss()
        test_loss, test_pos_loss, test_ori_loss= evaluate_model(
            model, test_loader, loss_fn, device
        )
        
        logging.info(
            f"Test Results: Loss={test_loss:.4f}, "
            f"Position Loss={test_pos_loss:.4f}, "
            f"Orientation Loss={test_ori_loss:.4f}"

        )