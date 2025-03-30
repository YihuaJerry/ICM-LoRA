"""
Multi-GPU Training Script for VAE using Accelerate Library

This script trains the provided VAE model on multiple GPUs using the Hugging Face Accelerate library.
It includes mixed-precision training, learning rate scheduling, and logging.

"""
import os
import argparse
import logging
from glob import glob
from tkinter.tix import Form
from turtle import st
from venv import logger

from accelerate.utils import set_seed
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from accelerate import Accelerator
from accelerate.logging import get_logger
import logging
# Import the VAE model from the provided code
from model.VAEdesign import OneDimVAE as VAE
from model.CVAE_design import CVAE
import matplotlib.pyplot as plt
import os
import numpy as np
from transformers import get_linear_schedule_with_warmup

import matplotlib.pyplot as plt

# =============================================================================
# Dataset Definition
# =============================================================================
import os
os.environ['TMPDIR'] = '/tmp'

class VAEDataset(Dataset):
    """
    Custom Dataset for loading normalized VAE training and evaluation data.
    Each data file is a dictionary containing:
      - "flattened": the normalized data as a 1D tensor.
      - "mean": the mean value of the original data.
      - "std": the standard deviation of the original data.
    """

    def __init__(self, data_dir,category_vector):
        """
        Args:
            data_dir (str): Directory containing the normalized data files.
        """
        self.data_files = glob(os.path.join(data_dir, 'normalized_*.pth'))
        if not self.data_files:
            raise FileNotFoundError(f"No data files found in {data_dir}")
        self.category_vector = category_vector

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        """
        Load and return a data sample.
        """
        try:
            data_dict = torch.load(self.data_files[idx])
            if "data" not in data_dict:
                raise ValueError(f"Missing 'data' key in {self.data_files[idx]}")

            # Load the flattened normalized data
            flattened_data = data_dict["data"]

            # Ensure data is in the expected shape for training
            if flattened_data.dim() != 1:
                raise ValueError(f"Flattened data in {self.data_files[idx]} is not 1D")
            return flattened_data.unsqueeze(0), self.category_vector
        except Exception as e:
            raise ValueError(f"Error loading {self.data_files[idx]}: {e}")


# =============================================================================
# Training Function
# =============================================================================
def print_grad(name):
    def hook(grad):
        print(f"Layer: {name}, Gradient mean: {grad.mean()}, Gradient std: {grad.std()}")
    return hook
def train(args):
    """
    Main training loop.
    """
    task_vector = args.task_vector_path
    C_loaded = np.load(task_vector)
    print("Loaded category vector shape:", C_loaded.shape)
    category_vector = torch.tensor(C_loaded, dtype=torch.float32)

    accelerator = Accelerator(mixed_precision='fp16' if args.fp16 else 'no', log_with="all", project_dir=args.output_dir )
    device = accelerator.device
    if args.seed is not None:
        set_seed(args.seed)
    else:
        set_seed(2024)

    # =============================================================================
    # Setup Logging
    # =============================================================================
    logger = get_logger(__name__, log_level="INFO")
    log_file_path = args.log_dir
    file_handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.logger.addHandler(file_handler)

    logger.info("This is an info message.")


    accelerator.print(f"Using {accelerator.device.type} device")
    accelerator.print("Random seed set to", args.seed)
    logger.info("Starting training...")

    train_loss_list = []
    train_recon_loss_list = []  
    train_kld_loss_list = []  
    # Initialize the CVAE model
    model = CVAE(
        latent_dim=args.latent_dim,
        input_length=args.input_length,
        condition_dim = args.condition_dim,
        kld_weight=args.kld_weight,
    ).to(device)

    # Prepare datasets and dataloaders
    train_dataset = VAEDataset(args.train_data_dir,category_vector)

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    total_steps = len(train_dataloader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = CosineAnnealingLR(
        optimizer=optimizer,
        T_max=args.num_epochs * 2,
    )

    accelerator.print(f"Total training steps: {total_steps}")
    logger.info(f"Total training steps: {total_steps}")

    model, optimizer, train_dataloader, scheduler, logger = accelerator.prepare(
        model, optimizer, train_dataloader , scheduler, logger
    )

    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume_from_checkpoint and args.checkpoint_dir:
        checkpoints = glob(os.path.join(args.checkpoint_dir, 'checkpoint_epoch_*'))
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split('_')[-1]))
            latest_checkpoint = checkpoints[-1]
            accelerator.load_state(latest_checkpoint)
            training_state = torch.load(os.path.join(latest_checkpoint, 'training_state.pt'))
            start_epoch = training_state['epoch']
            logger.info(f"Resumed training from checkpoint {latest_checkpoint} at epoch {start_epoch}")
        else:
            logger.info(f"No checkpoints found in {args.checkpoint_dir}, starting from scratch.")

    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"Warning: Parameter {name} does not require gradient.")

    # Training loop
    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        total_loss = 0
        total_recon_loss = 0  
        total_kld_loss = 0 

        # Use progress bar only in the main process
        if accelerator.is_main_process:
            train_iter = tqdm(train_dataloader, desc=f"Epoch [{epoch+1}/{args.num_epochs}]")
        else:
            train_iter = train_dataloader

        for batch_idx, (data,category_vector) in enumerate(train_iter):
            optimizer.zero_grad()

            # Move data to the appropriate device
            data = data.to(device, non_blocking=True)

            # Forward pass
            loss, recon_loss, kld_loss = model(data,category_vector)
            # Backward pass
            accelerator.backward(loss)
            optimizer.step()


            total_loss += loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            recon_loss = recon_loss.mean() 
            kld_loss = kld_loss.mean() 
            total_recon_loss += recon_loss.item() 
            total_kld_loss += kld_loss.item() 

            if accelerator.is_main_process and batch_idx % args.log_interval == 0:
                accelerator.print(
                    f"Epoch [{epoch + 1}/{args.num_epochs}], "
                    f"Batch [{batch_idx + 1}/{len(train_dataloader)}], "
                    f"Loss: {loss.item():.4f}, "
                    f"Recon Loss: {recon_loss.item():.4f}, "
                    f"KLD Loss: {kld_loss.item():.4f}, "
                    f"Learning Rate: {current_lr:.10f}"
                )
                logger.info(
                    f"Epoch [{epoch+1}/{args.num_epochs}], "
                    f"Batch [{batch_idx+1}/{len(train_dataloader)}], "
                    f"Loss: {loss.item():.4f}, "
                    f"Recon Loss: {recon_loss.item():.4f}, "
                    f"KLD Loss: {kld_loss.item():.4f}"
                    f"Learning Rate: {current_lr:.10f}"
                )
            metrics = {
                'train_loss': loss.item(),
                'train_recon_loss': recon_loss.item(),
                'train_kld_loss': kld_loss.item(),
                'epoch': epoch + 1,
                'batch': batch_idx + 1,
                'Learning Rate': current_lr
            }
            accelerator.log(metrics, step=epoch * len(train_dataloader) + batch_idx)


        if (epoch + 1) % args.save_checkpoint_epochs == 0 and args.checkpoint_dir and accelerator.is_main_process:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch + 1}')
            accelerator.save_state(checkpoint_path)
            training_state = {'epoch': epoch + 1}
            torch.save(training_state, os.path.join(checkpoint_path, 'training_state.pt'))
            logger.info(f"Saved checkpoint at {checkpoint_path}")
            accelerator.print(f"Saved checkpoint at {checkpoint_path}")

        avg_train_loss = total_loss / len(train_dataloader)
        avg_recon_loss = total_recon_loss / len(train_dataloader) 
        avg_kld_loss = total_kld_loss / len(train_dataloader) 


        train_loss_list.append(avg_train_loss)
        train_recon_loss_list.append(avg_recon_loss)  
        train_kld_loss_list.append(avg_kld_loss) 

        scheduler.step() 

    # Save the final model
    if args.output_dir and accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

        # model_path = os.path.join(args.output_dir, 'vae_final.pth')
        checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_End')

        accelerator.save_state(checkpoint_path)
        accelerator.print(f"Model saved to {checkpoint_path}")
        logger.info(f"Model saved to {checkpoint_path}")
    os.makedirs(args.output_dir, exist_ok=True)
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.num_epochs + 1), train_loss_list, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(args.output_dir, 'loss_curve1.png'))
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.num_epochs + 1), train_loss_list, label='Total Loss')
    plt.plot(range(1, args.num_epochs + 1), train_recon_loss_list, label='Reconstruction Loss')
    plt.plot(range(1, args.num_epochs + 1), train_kld_loss_list, label='KLD Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Components over Epochs')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(args.output_dir, 'loss_components_curve.png'))  # 保存新的图像
    plt.show()

# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Multi-GPU Training Script for VAE")

    # Data parameters
    parser.add_argument('--train_data_dir', type=str, required=True,
                        help="Directory containing training data files.")

    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=2000,
                        help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=2,
                        help="Batch size for training.")
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        help="Learning rate for the optimizer.")
    parser.add_argument('--latent_dim', type=int, default=256,
                        help="Dimension of the latent space.")
    parser.add_argument('--kld_weight', type=float, default=0.005,
                        help="Weight for the KL divergence loss.")
    # encoder channels，输入一个列表
    parser.add_argument('--encoder_channels', type=int, nargs='+',default=None,
                        help="Number of channels in the encoder layers.")
    # decoder channels，输入一个列表
    parser.add_argument('--decoder_channels', type=int, nargs='+',default=None,
                        help="Number of channels in the decoder layers.")
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help="Warm-up steps ratio")

    # Mixed precision
    parser.add_argument('--fp16', action='store_true',
                        help="Use mixed precision training.")
    
    #condition dim, the length of task vector(get from last_time)
    parser.add_argument('--condition_dim',type=int, nargs='+',default=768,help="condition dim, the length of task vector(get from last_time)")

    #input_dim, the dim of lora param
    parser.add_argument('--input_dim', type=int, nargs='+',default=1929928,help="input_dim, the dim of lora param")

    #task_vector path
    parser.add_argument('--task_vetcor_path',type=str,required=True,default="../ICL/hidden_state/e.g dog-r=8/hidden_state.pth")


    # Logging and output
    parser.add_argument('--log_interval', type=int, default=1,
                        help="How often to log training progress (in batches).")
    # 日志输出目录
    parser.add_argument('--log_dir', type=str, default='./logs/trainlog_cvae.log')
    parser.add_argument('--output_dir', type=str, default='./output2_cvae/new',
                        help="Directory to save the final model.")
    # Checkpoint parameters
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/lora_cvae_checkpoints/',
                        help="Directory to save/load checkpoints.")
    parser.add_argument('--save_checkpoint_epochs', type=int, default=1000,
                        help="Save checkpoints every N epochs.")
    parser.add_argument('--resume_from_checkpoint', action='store_true',
                        help="Resume training from the latest checkpoint.")

    # Seed for reproducibility (optional)
    parser.add_argument('--seed', type=int, default=2024,
                        help="Random seed for reproducibility.")
    args = parser.parse_args()
    print(args)
    logger.info(args)
    return args

# =============================================================================
# Main Execution
# =============================================================================

if __name__ == '__main__':
    try:
        args = parse_args()
        train(args)
    except Exception as e:
        raise e