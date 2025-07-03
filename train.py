# Code for training

import logging
import datetime
from dataset.dataset import CryoEMDataset
from models.model import create_regularized_model
from sklearn.model_selection import train_test_split
import training_config
import torch
from torch.utils.data import DataLoader
from utils.loss import WeightedMultiTaskLoss
import glob
from tqdm import tqdm
import wandb
from torch.nn.parallel import DataParallel
import torch.nn.utils as nn_utils
from collections import deque
import os
import warnings
warnings.filterwarnings("ignore")


# Setup logging
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_directory = "logs/training_logs"
os.makedirs(log_directory, exist_ok=True)
log_filename = f'training_log_BS_{training_config.batch_size}_{timestamp}.log'
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(f"{log_directory}/{log_filename}"),
                            logging.StreamHandler()])

data_path = list(glob.glob(training_config.train_dataset_path + '*/*.npz'))
train_path, val_path = train_test_split(data_path, test_size=0.2, random_state=42)
train_ds = CryoEMDataset(data_dir=train_path, use_augmentation=True)
val_ds = CryoEMDataset(data_dir=val_path, use_augmentation=False)

train_loader = DataLoader(train_ds, shuffle=True, batch_size=training_config.batch_size, pin_memory=training_config.pin_memory, num_workers=training_config.num_workers)
val_loader = DataLoader(val_ds, shuffle=False, batch_size=training_config.batch_size, pin_memory=training_config.pin_memory, num_workers=training_config.num_workers)

model_name = training_config.architecture_name
logging.info(f"Model Name: {model_name}")

def load_pretrained_weights(model, checkpoint_path):
    """Load pretrained weights into model"""
    logging.info(f"Loading pretrained weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['model_state_dict']
    
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    try:
        model.load_state_dict(new_state_dict)
    except:
        model.load_state_dict(state_dict)
    
    return model, checkpoint

def init_model():
    """Initialize the MICA model with specified parameters"""
    model = create_regularized_model()

    return model

def init_wandb():
    """Initialize wandb run with enhanced configuration"""
    wandb.login()
    run = wandb.init(
        project="MICA",
        config={
            "learning_rate": training_config.learning_rate,
            "batch_size": training_config.batch_size,
            "epochs": training_config.num_epochs,
            "model_type": "MICA",
            "optimizer": "Adam",
            "scheduler": "ReduceLROnPlateau",
        }
    )
    run.name = f"{model_name}_{timestamp}"
    
    # Create custom panels in WandB
    wandb.define_metric("batch_idx", hidden=True)
    wandb.define_metric("batch/*", step_metric="batch_idx")
    wandb.define_metric("epoch/*", step_metric="epoch")
    
    return run

def log_batch_metrics(losses_dict, phase, global_step, batch_idx, epoch, learning_rate):
    """Helper function to log batch-level metrics"""
    metrics = {
        f"batch/{phase}/{k}": v for k, v in losses_dict.items()
    }
    metrics.update({
        "batch_idx": global_step,
        "learning_rate": learning_rate,
        f"batch/{phase}/batch": batch_idx,
        f"batch/{phase}/epoch": epoch
    })
    wandb.log(metrics)

def log_epoch_metrics(avg_losses, phase, epoch, learning_rate):
    """Helper function to log epoch-level metrics"""
    metrics = {
        f"epoch/{phase}/{k}": v for k, v in avg_losses.items()
    }
    metrics.update({
        "epoch": epoch,
        "learning_rate": learning_rate
    })
    wandb.log(metrics)

def train_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    epoch_losses = {'total_loss': 0, 'backbone_loss': 0, 'carbon_alpha_loss': 0, 'amino_acid_loss': 0}
    
    # Initialize gradient history (persistent across calls)
    if not hasattr(train_epoch, 'grad_history'):
        train_epoch.grad_history = deque(maxlen=10)
    
    pbar = tqdm(loader, desc=f'Training Epoch {epoch}')
    
    for batch_idx, (input_map, af3_features, backbone_mask, carbon_alpha_mask, amino_acid_mask) in enumerate(pbar):
        input_map, af3_features = input_map.to(training_config.device), af3_features.to(training_config.device)
        backbone_mask, carbon_alpha_mask, amino_acid_mask = backbone_mask.to(training_config.device), carbon_alpha_mask.to(training_config.device), amino_acid_mask.to(training_config.device)
        
        optimizer.zero_grad()
        outputs = model(input_map, af3_features)
        loss, losses_dict = criterion(outputs, (backbone_mask, carbon_alpha_mask, amino_acid_mask), epoch, training_config.num_epochs)
        
        loss.backward()
        
        # Adaptive gradient clipping
        total_norm = sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
        train_epoch.grad_history.append(total_norm)
        
        # Track clipping info
        clipped = False
        clip_value = 0.0
        
        # Apply clipping if gradient is abnormally high
        if len(train_epoch.grad_history) >= 5:
            avg_norm = sum(train_epoch.grad_history) / len(train_epoch.grad_history)
            if total_norm > avg_norm * 2.0:
                clip_value = avg_norm * 1.5
                nn_utils.clip_grad_norm_(model.parameters(), clip_value)
                clipped = True
        
        # Add clipping info to losses_dict
        losses_dict.update({
            'gradient_norm': total_norm,
            'gradient_clipped': clipped,
            'clip_value': clip_value
        })
        
        optimizer.step()
        
        if training_config.logging:
            global_step = epoch * len(loader) + batch_idx
            current_lr = optimizer.param_groups[0]['lr']
            log_batch_metrics(
                losses_dict,
                'train',
                global_step,
                batch_idx,
                epoch,
                current_lr
            )
        
        for key in epoch_losses:
            epoch_losses[key] += losses_dict[key]
        
        # Enhanced progress bar with clipping info
        postfix = losses_dict.copy()
        if clipped:
            postfix['clip'] = f"✂️{clip_value:.2f}"
        pbar.set_postfix(postfix)
    
    avg_losses = {k: v/len(loader) for k, v in epoch_losses.items()}
    
    if training_config.logging:
        current_lr = optimizer.param_groups[0]['lr']
        log_epoch_metrics(
            avg_losses,
            'train',
            epoch,
            current_lr
        )
    
    return avg_losses

def validate(model, loader, criterion, epoch, optimizer):
    model.eval()
    val_losses = {'total_loss': 0, 'backbone_loss': 0, 'carbon_alpha_loss': 0, 'amino_acid_loss': 0}
    with torch.no_grad():
        pbar = tqdm(loader, desc=f'Validating Epoch {epoch}')
        for batch_idx, (input_map, af3_features, backbone_mask, carbon_alpha_mask, amino_acid_mask) in enumerate(pbar):
            input_map, af3_features = input_map.to(training_config.device), af3_features.to(training_config.device)
            backbone_mask, carbon_alpha_mask, amino_acid_mask = backbone_mask.to(training_config.device), carbon_alpha_mask.to(training_config.device), amino_acid_mask.to(training_config.device)

            outputs = model(input_map, af3_features)
            loss, losses_dict = criterion(outputs, (backbone_mask, carbon_alpha_mask, amino_acid_mask), epoch, training_config.num_epochs)
            
            if training_config.logging:
                global_step = epoch * len(loader) + batch_idx
                current_lr = optimizer.param_groups[0]['lr']
                log_batch_metrics(
                    losses_dict, 
                    'val', 
                    global_step, 
                    batch_idx, 
                    epoch,
                    current_lr
                )
            
            for key in val_losses:
                val_losses[key] += losses_dict[key]
            pbar.set_postfix(losses_dict)
    
    avg_losses = {k: v/len(loader) for k, v in val_losses.items()}
    
    if training_config.logging:
        current_lr = optimizer.param_groups[0]['lr']
        log_epoch_metrics(
            avg_losses, 
            'val', 
            epoch, 
            current_lr
        )
    
    return avg_losses

def main():    
    model = init_model().to(training_config.device)
    model = DataParallel(model)
    if training_config.resume_train:
        model, checkpoint = load_pretrained_weights(model, training_config.model_checkpoint)
    
    if training_config.logging:
        run = init_wandb()
    
    criterion = WeightedMultiTaskLoss().to(training_config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)
    
    if training_config.resume_train:
        if 'criterion_state_dict' in checkpoint:
            criterion.load_state_dict(checkpoint['criterion_state_dict'])
            logging.info("Loaded criterion state")
        
        if 'optimizer_state_dict' in checkpoint:
            optimizer_state = checkpoint['optimizer_state_dict']
            new_optimizer_state = {}
            for k, v in optimizer_state.items():
                if 'state' in k:
                    new_k = k.replace('module.', '')
                    new_optimizer_state[new_k] = v
                else:
                    new_optimizer_state[k] = v
                    
            try:
                optimizer.load_state_dict(new_optimizer_state)
            except:
                optimizer.load_state_dict(optimizer_state)
            logging.info("Loaded optimizer state")
        
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
    # Training loop
    if training_config.resume_train:
        start_epoch = checkpoint.get('epoch', -1) + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
    else:
        start_epoch = 0
        best_val_loss = float('inf')
    logging.info(f"Best Validation loss: {best_val_loss}")
    logging.info(f"Starting training from epoch {start_epoch}")
    for epoch in range(start_epoch, training_config.num_epochs):
        logging.info(f'Epoch {epoch+1}/{training_config.num_epochs}')
        model.module.update_dropout_rate(epoch)
        train_losses = train_epoch(model, train_loader, criterion, optimizer, epoch)
        val_losses = validate(model, val_loader, criterion, epoch, optimizer)
        print(f"Epoch {epoch+1}: Current dropout rate = {model.module.current_dropout:.3f}")
        # Log metrics
        for phase in ['train', 'val']:
            losses = train_losses if phase == 'train' else val_losses
            for k, v in losses.items():
                logging.info(f'{phase.capitalize()} {k}: {v:.4f}')
        
        # Save checkpoint if best model
        if val_losses['total_loss'] < best_val_loss:
            best_val_loss = val_losses['total_loss']
            checkpoint_path = f"{training_config.output_path}/{model_name}_epoch_{epoch}_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'criterion_state_dict': criterion.state_dict(),  # Save loss weights
            }, checkpoint_path)
        else:
            best_val_loss = val_losses['total_loss']
            checkpoint_path = f"{training_config.output_path}/{model_name}_epoch_{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'criterion_state_dict': criterion.state_dict(),  # Save loss weights
            }, checkpoint_path)
            
            
        scheduler.step(val_losses['total_loss'])
        
    if training_config.logging:
        wandb.finish()

if __name__ == '__main__':
    main()
