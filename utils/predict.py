#!/usr/bin/env python3
"""
CryoEM Structure Predictor

This module provides automated deep learning prediction for CryoEM structures.
It predicts backbone atoms, carbon alpha atoms, and amino acid positions with probabilities.

The predictor automatically adapts its processing strategy based on dataset size:
- Small datasets (≤120 samples): Single sample processing
- Large datasets (>120 samples): Optimized batch processing

Requirements:
- Path to deep learning model
- Path to directory containing grids
- Base path to save predicted results

Author: Rajan Gyawali
Date: June 01, 2025
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import glob
import shutil
import argparse
import sys
import logging
import psutil
from tqdm import tqdm
from dataset.dataset import CryoEMTestDataset
from models.model import MICA
import time
import warnings
import gc
warnings.filterwarnings("ignore")


class CryoEMPredictor:
    """
    CryoEM structure predictor with adaptive processing strategies.
    
    Automatically selects the most appropriate processing method based on
    dataset size and available system resources.
    """
    
    def __init__(self, model_path, grids_path, output_path, save_output=True, device="cuda", quiet=False):
        """
        Initialize the CryoEM predictor.
        
        Args:
            model_path (str): Path to trained model checkpoint
            grids_path (str): Path to directory containing grid files (.npz)
            output_path (str): Output directory for saving prediction results
            save_output (bool): Whether to save reconstructed volumes to disk
            device (str): Device to run inference on ('cuda', 'cpu', etc.)
            quiet (bool): Whether to suppress detailed output
        """
        self.model_path = model_path
        self.grids_path = grids_path
        self.output_path = output_path
        self.temp_output_path = os.path.join(output_path, "results", "predicted_grids")
        self.reconstruction_path = os.path.join(output_path, "results", self.grids_path.split("/")[-2])
        self.save_output = self._parse_save_output(save_output)
        self.device = device
        self.quiet = quiet
        
        # Processing configuration
        self.num_workers = 0
        self.pin_memory = False
        self.batch_threshold = 200
        
        # Model components
        self.model = None
        self.softmax = torch.nn.Softmax(dim=1)
        
        # Processing state
        self.use_optimized_batching = False
        self.sample_count = 0
        self.optimal_batch_size = 1
        
        # Performance tracking
        self.timing_stats = {
            'strategy_selection': 0,
            'model_loading': 0,
            'data_loading': 0,
            'inference': 0,
            'reconstruction': 0,
            'saving': 0,
            'total': 0
        }
        
        self._setup_logging()

    def _parse_save_output(self, save_output):
        """Parse save_output parameter."""
        if isinstance(save_output, str):
            return save_output.lower() == 'true'
        return bool(save_output)

    def _setup_logging(self):
        """Initialize logging configuration."""
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)
        self.logger.propagate = False

    def _print_status(self, message):
        """Print status message if not in quiet mode."""
        if not self.quiet:
            print(message)

    def _get_gpu_memory_gb(self):
        """Get available GPU memory in GB."""
        if self.device != 'cpu' and torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                return gpu_memory / (1024**3)
            except Exception:
                return 0
        return 0

    def _get_system_memory_gb(self):
        """Get total system memory in GB."""
        memory = psutil.virtual_memory()
        return memory.total / (1024**3)

    def _estimate_model_memory_gb(self):
        """Estimate model memory usage in GB."""
        try:
            temp_model = MICA()
            total_params = sum(p.numel() for p in temp_model.parameters())
            model_memory_gb = (total_params * 4) / (1024**3)
            estimated_total = model_memory_gb * 3  # Account for activations
            del temp_model
            return estimated_total
        except Exception:
            return 2.0

    def _estimate_batch_memory_gb(self):
        """Estimate memory usage per batch sample in GB."""
        grid_size = 64 * 64 * 64
        af3_features_size = 64 * 64 * 64 * 24
        total_elements = (grid_size + af3_features_size) * 5  # Input + intermediate calculations
        return total_elements / (1024**3)

    def _calculate_optimal_batch_size(self):
        """Calculate optimal batch size based on available memory."""
        gpu_memory = self._get_gpu_memory_gb()
        system_memory = self._get_system_memory_gb()
        model_memory = self._estimate_model_memory_gb()
        batch_memory = self._estimate_batch_memory_gb()
        
        if self.device != 'cpu' and gpu_memory > 0:
            available_memory = gpu_memory * 0.7  # Conservative GPU usage
            memory_for_batches = available_memory - model_memory
        else:
            available_memory = system_memory * 0.5  # Conservative system usage
            memory_for_batches = available_memory - model_memory
        
        if memory_for_batches <= 0:
            return 1
        
        optimal_size = max(1, int(memory_for_batches / batch_memory))
        return min(optimal_size, 8)  # Cap at 8 for stability

    def select_processing_strategy(self):
        """Analyze dataset and select optimal processing strategy."""
        start_time = time.time()
        
        self._print_status("")
        self._print_status("=" * 80)
        self._print_status("🧭 PROCESSING STRATEGY SELECTION")
        self._print_status("=" * 80)
        
        try:
            grid_files = glob.glob(f"{self.grids_path}/normalized_map_grids/*.npz")
            self.sample_count = len(grid_files)
            
            if self.sample_count == 0:
                self.logger.error(f"No grid files found in: {self.grids_path}/normalized_map_grids/")
                return False
            
            self.use_optimized_batching = self.sample_count > self.batch_threshold
            
            self._print_status(f"📊 Dataset Analysis:")
            self._print_status(f"   🔢 Total samples: {self.sample_count}")
            self._print_status(f"   🎯 Batch threshold: {self.batch_threshold}")
            
            if self.use_optimized_batching:
                self.optimal_batch_size = self._calculate_optimal_batch_size()
                self._print_status(f"   ✅ Strategy: OPTIMIZED BATCH PROCESSING")
                self._print_status(f"   📦 Batch size: {self.optimal_batch_size}")
            else:
                self._print_status(f"   ✅ Strategy: SINGLE SAMPLE PROCESSING")
                self._print_status(f"   📦 Batch size: 1")
            
            self.timing_stats['strategy_selection'] = time.time() - start_time
            self._print_status("")
            return True
            
        except Exception as e:
            self.logger.error(f"Strategy selection failed: {str(e)}")
            self.timing_stats['strategy_selection'] = time.time() - start_time
            return False

    def load_model(self):
        """Load the trained model."""
        self._print_status("=" * 80)
        self._print_status("🤖 MODEL LOADING")
        self._print_status("=" * 80)
        
        start_time = time.time()
        
        try:
            if not os.path.exists(self.model_path):
                self.logger.error(f"Model file not found: {self.model_path}")
                return False
            
            self.logger.info(f"Loading model from: {os.path.basename(self.model_path)}")
            
            # Initialize and load model
            self.model = MICA().to(self.device)
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Clean state dict keys
            state_dict = {k.replace("module.", ""): v.to(self.device) 
                         for k, v in checkpoint['model_state_dict'].items()}
            
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
            
            self.timing_stats['model_loading'] = time.time() - start_time
            self.logger.info("Model loaded successfully")
            
            self._print_status(f"🎯 Results:")
            self._print_status(f"   ✅ Model loaded successfully")
            self._print_status(f"   ⏱️  Loading time: {self.timing_stats['model_loading']:.2f}s")
            self._print_status("")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Model loading failed: {str(e)}")
            self.timing_stats['model_loading'] = time.time() - start_time
            self._print_status(f"   ❌ Model loading failed")
            self._print_status("")
            return False

    def prepare_data(self):
        """Prepare dataset and data loader."""
        self._print_status("=" * 80)
        self._print_status("📊 DATA PREPARATION")
        self._print_status("=" * 80)
        
        start_time = time.time()
        
        try:
            grid_files = glob.glob(f"{self.grids_path}/normalized_map_grids/*.npz")
            
            if not grid_files:
                self.logger.error(f"No grid files found in: {self.grids_path}/normalized_map_grids/")
                return False, None
            
            # Create dataset
            dataset = CryoEMTestDataset(data_dir=grid_files, transform=None)
            
            # Configure data loader
            batch_size = self.optimal_batch_size if self.use_optimized_batching else 1
            data_loader = DataLoader(
                dataset,
                shuffle=False,
                batch_size=batch_size,
                pin_memory=self.pin_memory,
                num_workers=self.num_workers
            )
            
            self.timing_stats['data_loading'] = time.time() - start_time
            self.logger.info(f"Dataset prepared with {len(dataset)} samples")
            
            self._print_status(f"🎯 Results:")
            self._print_status(f"   ✅ Dataset prepared successfully")
            self._print_status(f"   🔢 Total samples: {len(dataset)}")
            self._print_status(f"   📦 Batch size: {batch_size}")
            self._print_status(f"   ⏱️  Preparation time: {self.timing_stats['data_loading']:.2f}s")
            self._print_status("")
            
            return True, data_loader
            
        except Exception as e:
            self.logger.error(f"Data preparation failed: {str(e)}")
            self.timing_stats['data_loading'] = time.time() - start_time
            self._print_status(f"   ❌ Data preparation failed")
            self._print_status("")
            return False, None

    def run_inference(self, data_loader):
        """Execute inference on all samples."""
        self._print_status("=" * 80)
        strategy_name = "OPTIMIZED BATCH" if self.use_optimized_batching else "SINGLE SAMPLE"
        self._print_status(f"🚀 RUNNING {strategy_name} INFERENCE")
        self._print_status("=" * 80)
        
        start_time = time.time()
        
        try:
            # Prepare output directory
            if os.path.exists(self.temp_output_path):
                shutil.rmtree(self.temp_output_path)
            os.makedirs(self.temp_output_path, exist_ok=True)
            
            total_batches = len(data_loader)
            samples_processed = 0
            
            self._print_status(f"📊 Configuration:")
            self._print_status(f"   🔢 Total batches: {total_batches}")
            self._print_status(f"   📦 Batch size: {self.optimal_batch_size if self.use_optimized_batching else 1}")
            self._print_status("")
            
            self.model.eval()
            with torch.no_grad():
                progress_bar = tqdm(data_loader, desc="Processing samples")
                
                for batch_idx, (x, af3_features, metadata_batch) in enumerate(progress_bar):
                    # Process batch
                    x, af3_features = x.to(self.device), af3_features.to(self.device)
                    
                    # Forward pass
                    bb_output, ca_output, aa_output = self.model(x, af3_features)
                    
                    # Apply softmax and process outputs
                    bb_output = torch.cat((bb_output[:,:1], bb_output[:,2:]), dim=1)
                    bb_scores = self.softmax(bb_output)
                    
                    ca_output = torch.cat((ca_output[:,:1], ca_output[:,2:]), dim=1)
                    ca_scores = self.softmax(ca_output)
                    
                    aa_scores = self.softmax(aa_output[:,1:,:,:,:])
                    aa_predictions = torch.max(aa_scores, 1)[1]
                    
                    # Save results for each sample in batch
                    batch_size = x.size(0)
                    for i in range(batch_size):
                        # Extract metadata for this sample
                        sample_metadata = self._extract_sample_metadata(metadata_batch, i)
                        
                        # Prepare results
                        results = {
                            'backbone_probability': bb_scores[i,2,:,:,:].cpu().numpy(),
                            'carbon_alpha_probability': ca_scores[i,2,:,:,:].cpu().numpy(),
                            'amino_acid_probability': aa_scores[i,:,:,:,:].cpu().numpy(),
                            'amino_acid_prediction': aa_predictions[i,:,:,:].cpu().numpy()
                        }
                        
                        # Save results
                        filename = sample_metadata['filename']
                        for key, data in results.items():
                            output_file = f"{self.temp_output_path}/{filename}_{key}.npz"
                            np.savez(output_file, data=data, metadata=sample_metadata)
                        
                        samples_processed += 1
                    
                    # Memory cleanup
                    del x, af3_features, bb_output, ca_output, aa_output
                    if self.device != 'cpu':
                        torch.cuda.empty_cache()
            
            self.timing_stats['inference'] = time.time() - start_time
            self.logger.info("Inference completed successfully")
            
            self._print_status(f"🎯 Results:")
            self._print_status(f"   ✅ Inference completed successfully")
            self._print_status(f"   🔢 Samples processed: {samples_processed}")
            self._print_status(f"   ⏱️  Inference time: {self.timing_stats['inference']:.2f}s")
            
            if self.timing_stats['inference'] > 0:
                throughput = samples_processed / self.timing_stats['inference']
                self._print_status(f"   ⚡ Throughput: {throughput:.2f} samples/sec")
            
            self._print_status("")
            return True
            
        except Exception as e:
            self.logger.error(f"Inference failed: {str(e)}")
            self.timing_stats['inference'] = time.time() - start_time
            self._print_status(f"   ❌ Inference failed")
            self._print_status("")
            return False

    def _extract_sample_metadata(self, metadata_batch, sample_idx):
        """Extract metadata for a specific sample from batch metadata."""
        sample_metadata = {}
        
        for key, value in metadata_batch.items():
            try:
                if hasattr(value, 'detach'):
                    # Handle tensor metadata
                    if value.dim() > 0 and len(value) > sample_idx:
                        tensor_val = value[sample_idx].detach()
                        if tensor_val.dim() == 0:
                            sample_metadata[key] = tensor_val.item()
                        else:
                            sample_metadata[key] = tensor_val.numpy()
                    else:
                        sample_metadata[key] = value.item() if value.dim() == 0 else value.detach().numpy()
                elif isinstance(value, (list, tuple)):
                    # Handle list/tuple metadata
                    if len(value) > sample_idx:
                        sample_metadata[key] = value[sample_idx]
                    else:
                        sample_metadata[key] = value[0] if len(value) > 0 else value
                elif isinstance(value, np.ndarray):
                    # Handle numpy array metadata
                    if value.ndim > 0 and len(value) > sample_idx:
                        sample_metadata[key] = value[sample_idx]
                    elif value.ndim == 0:
                        sample_metadata[key] = value.item()
                    else:
                        sample_metadata[key] = value[0] if len(value) > 0 else value
                else:
                    # Handle scalar metadata
                    sample_metadata[key] = value
            except Exception as e:
                # Fallback: use the original value
                sample_metadata[key] = value
                
        return sample_metadata

    def reconstruct_volume(self, grid_files, map_type, padding=8):
        """Reconstruct full volume from grid predictions."""
        start_time = time.time()
        
        try:
            # Load first grid to get metadata
            first_grid = np.load(grid_files[0], allow_pickle=True)
            metadata = first_grid['metadata'].item()
            
            # Handle different orig_shape formats
            if isinstance(metadata['orig_shape'], np.ndarray):
                if metadata['orig_shape'].ndim > 0:
                    original_shape = metadata['orig_shape'].flatten()
                else:
                    # Handle 0-dimensional array
                    original_shape = [int(metadata['orig_shape'].item())]
            else:
                original_shape = list(metadata['orig_shape'])
            
            # Initialize reconstruction volume
            if map_type == 'amino_acid_probability':
                volume = np.zeros((20, *original_shape), dtype=np.float32)
            else:
                volume = np.zeros(original_shape, dtype=np.float32)
            
            # Reconstruct from all grids
            for grid_file in grid_files:
                data = np.load(grid_file, allow_pickle=True)
                grid = data['data']
                grid_metadata = data['metadata'].item()
                
                # Extract position information with robust handling
                def extract_value(val):
                    """Extract scalar value from various formats."""
                    if isinstance(val, np.ndarray):
                        if val.ndim == 0:
                            return int(val.item())
                        else:
                            return int(val.flatten()[0])
                    elif isinstance(val, (list, tuple)):
                        return int(val[0])
                    else:
                        return int(val)
                
                try:
                    i = extract_value(grid_metadata['i'])
                    j = extract_value(grid_metadata['j'])
                    k = extract_value(grid_metadata['k'])
                    di = extract_value(grid_metadata['di'])
                    dj = extract_value(grid_metadata['dj'])
                    dk = extract_value(grid_metadata['dk'])
                except Exception as meta_error:
                    self.logger.error(f"Metadata extraction failed for {grid_file}: {str(meta_error)}")
                    continue
                
                # Extract central region (remove padding)
                try:
                    if map_type == 'amino_acid_probability':
                        central_grid = grid[:, padding:padding+di, padding:padding+dj, padding:padding+dk]
                        volume[:, i:i+di, j:j+dj, k:k+dk] = central_grid
                    else:
                        central_grid = grid[padding:padding+di, padding:padding+dj, padding:padding+dk]
                        volume[i:i+di, j:j+dj, k:k+dk] = central_grid
                except Exception as grid_error:
                    self.logger.error(f"Grid processing failed for {grid_file}: {str(grid_error)}")
                    continue
            
            reconstruction_time = time.time() - start_time
            return volume, reconstruction_time
            
        except Exception as e:
            reconstruction_time = time.time() - start_time
            self.logger.error(f"Volume reconstruction failed for {map_type}: {str(e)}")
            return None, reconstruction_time

    def reconstruct_and_save_volumes(self):
        """Reconstruct all prediction types and save volumes."""
        self._print_status("=" * 80)
        self._print_status("🔧 VOLUME RECONSTRUCTION")
        self._print_status("=" * 80)
        
        start_time = time.time()
        
        try:
            os.makedirs(self.reconstruction_path, exist_ok=True)
            
            # Define map types to reconstruct
            map_types = {
                'backbone_probability': 'backbone_probability',
                'carbon_alpha_probability': 'carbon_alpha_probability',
                'amino_acid_prediction': 'amino_acid_prediction',
                'amino_acid_probability': 'amino_acid_probability'
            }
            
            reconstructed_volumes = {}
            volumes_created = 0
            total_reconstruction_time = 0
            
            save_start = time.time()
            
            for map_name, map_type in map_types.items():
                # Find grid files for this map type
                grid_pattern = f"{self.temp_output_path}/*{map_name}.npz"
                grid_files = glob.glob(grid_pattern)
                
                if grid_files:
                    self.logger.info(f"Reconstructing {map_name} from {len(grid_files)} grids")
                    
                    volume, recon_time = self.reconstruct_volume(grid_files, map_type)
                    
                    if volume is not None:
                        reconstructed_volumes[map_type] = volume
                        total_reconstruction_time += recon_time
                        volumes_created += 1
                        
                        # Save volume if requested
                        if self.save_output:
                            output_file = f'{self.reconstruction_path}/{map_name}.npy'
                            np.save(output_file, volume)
                            self.logger.info(f"Saved {map_name}.npy with shape {volume.shape}")
                    else:
                        self.logger.error(f"Failed to reconstruct {map_name}")
                else:
                    self.logger.warning(f"No grid files found for {map_name}")
            
            save_time = time.time() - save_start - total_reconstruction_time
            self.timing_stats['reconstruction'] = total_reconstruction_time
            self.timing_stats['saving'] = save_time
            
            success = volumes_created > 0
            
            self._print_status(f"🎯 Results:")
            if success:
                self._print_status(f"   ✅ Reconstruction completed successfully")
                self._print_status(f"   🗺️  Volumes created: {volumes_created}/4")
                self._print_status(f"   ⏱️  Reconstruction time: {total_reconstruction_time:.2f}s")
                if self.save_output:
                    self._print_status(f"   💾 Saving time: {save_time:.2f}s")
            else:
                self._print_status(f"   ❌ Reconstruction failed")
            
            self._print_status("")
            return success, reconstructed_volumes
            
        except Exception as e:
            self.logger.error(f"Volume reconstruction failed: {str(e)}")
            self._print_status(f"   ❌ Reconstruction failed")
            self._print_status("")
            return False, {}

    def run_prediction(self):
        """Execute the complete prediction pipeline."""
        total_start_time = time.time()
        self.logger.info(f"Starting prediction pipeline at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Step 1: Select processing strategy
            if not self.select_processing_strategy():
                self._print_status("❌ Strategy selection failed")
                return False, {}
            
            # Step 2: Load model
            if not self.load_model():
                self._print_status("❌ Model loading failed")
                return False, {}
            
            # Step 3: Prepare data
            data_success, data_loader = self.prepare_data()
            if not data_success:
                self._print_status("❌ Data preparation failed")
                return False, {}
            
            # Step 4: Run inference
            if not self.run_inference(data_loader):
                self._print_status("❌ Inference failed")
                return False, {}
            
            # Step 5: Reconstruct volumes
            reconstruction_success, volumes = self.reconstruct_and_save_volumes()
            if not reconstruction_success:
                self._print_status("❌ Volume reconstruction failed")
                return False, {}
            
            # Calculate total time and print summary
            self.timing_stats['total'] = time.time() - total_start_time
            self._print_summary()
            
            # Cleanup temporary files
            if os.path.exists(self.temp_output_path):
                shutil.rmtree(self.temp_output_path)
            
            return True, volumes
            
        except Exception as e:
            self.logger.error(f"Prediction pipeline failed: {str(e)}")
            return False, {}

    def _print_summary(self):
        """Print comprehensive pipeline summary."""
        self._print_status("=" * 80)
        self._print_status("📊 DEEP LEARNING PREDICTION SUMMARY")
        self._print_status("=" * 80)
        
        strategy_name = "Optimized batching" if self.use_optimized_batching else "Single sample"
        
        self._print_status(f"🎯 Overall Results:")
        self._print_status(f"   ✅ Pipeline completed successfully")
        self._print_status(f"   📊 Processing strategy: {strategy_name}")
        self._print_status(f"   🔢 Total samples: {self.sample_count}")
        self._print_status(f"   ⏱️  Total time: {self.timing_stats['total']:.2f}s")
        
        if self.timing_stats['inference'] > 0:
            throughput = self.sample_count / self.timing_stats['inference']
            avg_time = self.timing_stats['inference'] / self.sample_count
            self._print_status(f"⚡ Performance:")
            self._print_status(f"   📈 Throughput: {throughput:.2f} samples/sec")
            self._print_status(f"   ⏱️  Average per sample: {avg_time:.3f}s")
        
        self._print_status("")
        self._print_status("=" * 80)
        self._print_status("🚀 DEEP LEARNING PREDICTION COMPLETE")
        self._print_status("=" * 80)


def main():
    """Main entry point with command line interface."""
    parser = argparse.ArgumentParser(
        description='CryoEM Structure Predictor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Processing Strategy:
- Small datasets (≤120 samples): Single sample processing
- Large datasets (>120 samples): Optimized batch processing

Examples:
  python predictor.py -g /path/to/grids -m model.pth -o /output/path
  python predictor.py -g /path/to/grids -m model.pth -o /output/path --device cpu
        """
    )
    
    parser.add_argument('-g', '--grids_path', required=True,
                       help='Path to directory containing grid files (.npz)')
    parser.add_argument('-m', '--model_path', required=True,
                       help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('-o', '--output_path', required=True,
                       help='Output directory for saving prediction results')
    parser.add_argument('--save_output', default='True',
                       help='Save reconstructed volumes to disk (default: True)')
    parser.add_argument('--device', default='cuda',
                       help='Device for inference: cuda, cpu, cuda:0, etc. (default: cuda)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress detailed output messages')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model file not found: {args.model_path}")
        sys.exit(1)
    
    if not os.path.exists(args.grids_path):
        print(f"❌ Error: Grids directory not found: {args.grids_path}")
        sys.exit(1)
    
    # Validate device availability
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print(f"⚠️  Warning: CUDA not available, using CPU instead")
        args.device = 'cpu'
    elif args.device == 'mps' and not torch.backends.mps.is_available():
        print(f"⚠️  Warning: MPS not available, using CPU instead")
        args.device = 'cpu'
    
    # Print system information
    print("")
    print("=" * 80)
    print("🚀 DEEP LEARNING PREDICTION")
    print("=" * 80)
    print(f"📊 System Information:")
    print(f"   🖥️  Device: {args.device}")
    print(f"   🧠 CPU cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    print(f"   💾 System memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    if args.device.startswith('cuda') and torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"   🎮 GPU memory: {gpu_memory:.1f} GB")
    
    print("")
    
    # Initialize and run predictor
    predictor = CryoEMPredictor(
        model_path=args.model_path,
        grids_path=args.grids_path,
        output_path=args.output_path,
        save_output=args.save_output,
        device=args.device,
        quiet=args.quiet
    )
    
    try:
        success, reconstructed_volumes = predictor.run_prediction()
        
        if success:
            print("✅ Prediction completed successfully!")
            sys.exit(0)
        else:
            print("❌ Prediction failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n❌ Process interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()