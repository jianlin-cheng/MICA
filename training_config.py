# Configuration file for training
import argparse
import torch

# Create an argument parser
parser = argparse.ArgumentParser(description="MICA Training")

# Add arguments
parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
parser.add_argument("--num_epochs", type=int, default=60, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size")

# Logging-related arguments
parser.add_argument("--logging", type=bool, default=False, help="Enable logging")

# Data related arguments
parser.add_argument("--train_dataset_path", type=str, default=f"Training_Dataset/Grids/normalized_maps/", help="Path to the training dataset")
parser.add_argument("--output_path", type=str, default="trained_models", help="Output directory")
parser.add_argument("--model_checkpoint", type=str, default="trained_models/MICA_best_model.pth", help="Path to MICA checkpoint")
parser.add_argument("--resume_train", type=bool, default=False, help="Resume Training")


# Device-related arguments
parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device (cuda:0, cuda:1 or cpu)")
parser.add_argument("--pin_memory", action="store_true", help="Enable pin_memory for data loading if using CUDA")
parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")

# Additional info in architecture name
architecture_name = "MICA_BS_{}".format(
    parser.parse_args().batch_size
)
parser.add_argument("--architecture_name", type=str, default=architecture_name, help="Model architecture name")

# Parse the command-line arguments
args = parser.parse_args()

learning_rate = args.learning_rate
num_epochs = args.num_epochs
batch_size = args.batch_size
logging = args.logging
train_dataset_path = args.train_dataset_path
output_path = args.output_path
model_checkpoint = args.model_checkpoint
resume_train = args.resume_train
device = args.device
pin_memory = args.pin_memory
num_workers = args.num_workers
architecture_name = args.architecture_name
