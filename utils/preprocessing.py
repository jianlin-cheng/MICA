#!/usr/bin/env python3
"""
DataPreprocessor

This Preprocessor class performs:
        1. Combine all the domainwise structures into a single PDB structure
        2. Resamples and normalizes the input cryo-EM density map
        3. Create a 24-channel encoded volume of the same dimension as normalized density map from the combined AF3 structure containing 4 backbone atoms and 20 amino acid types in each of the channel
        4. Create grids 0f 64x64x64 for normalized density map
        5. Create grids of 64x64x64 for each channel of encoded volume from (3).

Author: Rajan Gyawali
Date: June 1, 2025
"""

import argparse
import os
import warnings
import logging
import sys
from glob import glob
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from Bio import PDB
import mrcfile
import numpy as np
from scipy.ndimage import zoom
warnings.filterwarnings("ignore")


class DataPreprocessor:
    """
    This Preprocessor class performs:
        1. Resamples and normalizes the input cryo-EM density map
        2. Create a 24-channel encoded volume of the same dimension as normalized density map from the combined AF3 structure containing 4 backbone atoms and 20 amino acid types in each of the channel
        3. Create grids 0f 64x64x64 for normalized density map
        4. Create grids of 64x64x64 for each channel of encoded volume from (3).
    """
    
    def __init__(self, map_path, AF3_results, normalized_map_path=None, quiet=True):
        """
        Initialize the AF3 docked structure combiner.
        
        Args:
            quiet: Whether to suppress BioPython parser warnings
        """
        self.map_path = map_path
        self.AF3_results = AF3_results
        self.normalized_map_path = normalized_map_path
        self.parser = PDB.PDBParser(QUIET=quiet)
        self.io = PDB.PDBIO()
        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration with timestamps."""
        # Create custom formatter for regular logs
        self.formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(self.formatter)
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)
        
        # Prevent duplicate logs
        self.logger.propagate = False

    def print_clean(self, message):
        """Print message without timestamp or log level for visual appeal."""
        print(message)
    
    def resample_and_normalize_map(self, target_voxel_size=1.0):
        """
        Resample and normalize cryo-EM density map.
        
        Args:
            target_voxel_size: Target voxel size for resampling
            
        Returns:
            None
        """
        
        self.print_clean("")
        self.print_clean("=" * 80)
        self.print_clean("🗺️  DENSITY MAP RESAMPLING AND NORMALIZATION")
        self.print_clean("=" * 80)
        success = 'False'
        try:
            self.logger.info(f"Processing density map: {os.path.basename(self.map_path)}")
            with mrcfile.open(self.map_path) as mrc:
                data = mrc.data
                voxel_size = mrc.voxel_size
                origin = mrc.header.origin
                mapc = int(mrc.header.mapc)
                mapr = int(mrc.header.mapr)
                maps = int(mrc.header.maps)
                nxstart = int(mrc.header.nxstart)
                nystart = int(mrc.header.nystart)
                nzstart = int(mrc.header.nzstart)
            
            self.logger.info(f"Original map shape: {data.shape}")
            
            # Resample map
            zoom_factors = [voxel_size.x / target_voxel_size, 
                           voxel_size.y / target_voxel_size, 
                           voxel_size.z / target_voxel_size]
            
            self.logger.info(f"Resampling with zoom factors: {zoom_factors}")
            resampled_data = zoom(data, zoom_factors, order=3)
            self.logger.info(f"✓ Resampled map shape: {resampled_data.shape}")
            
            # Normalize map
            self.logger.info("Normalizing density map...")
            norm_data = np.nan_to_num(resampled_data)
            median = np.median(norm_data)
            map_data_ = (norm_data > median) * (norm_data - median)
            positive_values = map_data_[np.where(map_data_ > 0)]
            
            if len(positive_values) > 0:
                percentile_value = np.percentile(positive_values, 99.9)
                
                if percentile_value != 0:
                    map_data_ = (map_data_ < percentile_value) * map_data_ + \
                               (map_data_ >= percentile_value) * percentile_value
                    map_data_ /= percentile_value
                    self.logger.info(f"✓ Normalized map shape: {map_data_.shape}")
                    
                    self.normalized_map_path =  os.path.join(os.path.dirname(self.AF3_results), 'resampled_normalized_map.mrc')

                    with mrcfile.new(self.normalized_map_path, overwrite=True) as mrc:
                        mrc.set_data(map_data_.astype(np.float32))
                        mrc.voxel_size = (target_voxel_size, target_voxel_size, target_voxel_size)
                        mrc.header.origin = origin
                        mrc.header.mapc = mapc
                        mrc.header.mapr = mapr
                        mrc.header.maps = maps
                        mrc.header.nxstart = nxstart
                        mrc.header.nystart = nystart
                        mrc.header.nzstart = nzstart
                        mrc.update_header_stats()
                    
                    self.logger.info(f"✓ Normalized map saved: {os.path.basename(self.normalized_map_path)}")
                    success = True
                else:
                    error_msg = "Percentile value is zero - cannot normalize"
                    self.logger.error(f"✗ Normalization failed: {error_msg}")
            else:
                error_msg = "No positive values found after thresholding"
                self.logger.error(f"✗ Normalization failed: {error_msg}")
            
        except Exception as e:
            error_msg = f"Map processing failed: {str(e)}"
            self.logger.error(f"✗ {error_msg}")
            
        self.print_clean("")
        self.print_clean(f"🎯 Map Processing Results:")
        if success == True:
            self.print_clean(f"   ✅ Map successfully resampled to {target_voxel_size} A and normalized to [0, 1] range.")
            self.print_clean(f"   📁 Normalized map path: {self.normalized_map_path}")
            self.print_clean("")
        else:
            self.print_clean(f"   ❌ Map Resampling and Normalization Failed")
    
    def transform_coordinates(self, coord, origin, shape):
        """Transform PDB coordinates to density map indices."""
        coord_shifted = coord - np.array((origin.x, origin.y, origin.z))
        indices = coord_shifted / 1.0
        indices = np.round(indices).astype(int)
        indices = np.clip(indices, 0, np.array(shape) - 1)
        return indices

    def get_aa_channel_index(self, residue_name, amino_acids, backbone_atoms):
        """Get channel index for amino acid type."""
        try:
            return len(backbone_atoms) + amino_acids.index(residue_name)
        except ValueError:
            return -1
        
    def save_channel_encoding(self, args):
        """
        Save a single channel encoding to MRC file.
        """
        (channel_idx, channel_name, channel_data, output_dir) = args
        
        try:
            output_path = os.path.join(output_dir, f"{channel_name}_encoding.mrc")
            
            with mrcfile.new(output_path, overwrite=True) as mrc:
                mrc.set_data(channel_data.astype(np.float32))
                mrc.header.origin = self.origin
                mrc.header.nxstart = self.nxstart
                mrc.header.nystart = self.nystart
                mrc.header.nzstart = self.nzstart
                mrc.header.mapc = self.mapc
                mrc.header.mapr = self.mapr
                mrc.header.maps = self.maps
                mrc.voxel_size = 1
                mrc.update_header_stats()
            
            return {
                'success': True,
                'channel_idx': channel_idx,
                'channel_name': channel_name,
                'output_path': output_path,
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'channel_idx': channel_idx,
                'channel_name': channel_name,
                'output_path': None,
                'error': str(e)
            }
            
    def create_AF3_encodings(self, combined_docked_model_path):
        """
        Create AF3 feature encodings from combined PDB structure and normalized density map.
        """
        self.print_clean("")
        self.print_clean("=" * 80)
        self.print_clean("🧬 AF3 FEATURES ENCODING")
        self.print_clean("=" * 80)
        success = False
        try:
            self.logger.info(f"Creating AF3 encodings...")
            self.logger.info(f"Using reference map: {os.path.basename(self.normalized_map_path)}")
            
            # Read density map to get reference dimensions and parameters
            with mrcfile.open(self.normalized_map_path) as mrc:
                self.map_data = mrc.data
                self.shape = mrc.data.shape
                self.voxel_size = mrc.voxel_size
                self.origin = mrc.header.origin
                self.mapc = int(mrc.header.mapc)
                self.mapr = int(mrc.header.mapr)
                self.maps = int(mrc.header.maps)
                self.nxstart = int(mrc.header.nxstart)
                self.nystart = int(mrc.header.nystart)
                self.nzstart = int(mrc.header.nzstart)
            
            self.logger.info(f"Reference map shape: {self.shape}")
            
            # Define backbone atoms and amino acids
            backbone_atoms = ['CA', 'N', 'C', 'O']
            amino_acids = [
                'ALA', 'CYS', 'ASP', 'GLU', 'PHE', 
                'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
                'MET', 'ASN', 'PRO', 'GLN', 'ARG', 
                'SER', 'THR', 'VAL', 'TRP', 'TYR'
            ]
            
            num_channels = len(backbone_atoms) + len(amino_acids)
            channel_names = backbone_atoms + amino_acids
            
            self.logger.info(f"Creating {num_channels} feature channels")
            
            # Initialize feature volume (channels, depth, height, width)
            feature_volume = np.zeros((num_channels, *self.shape))
            structure = self.parser.get_structure('protein', combined_docked_model_path)
            
            # Encode features
            atoms_processed = 0
            residues_processed = 0
            
            for model in structure:
                for chain in model:
                    for residue in chain:
                        # Only process standard amino acids
                        if residue.get_id()[0] == ' ':
                            res_name = residue.get_resname()
                            residues_processed += 1
                            
                            # Get amino acid channel index
                            aa_idx = self.get_aa_channel_index(res_name, amino_acids, backbone_atoms)
                            
                            for atom in residue:
                                # Transform coordinates to map indices
                                idx = self.transform_coordinates(atom.get_coord(), self.origin, self.shape)
                                atoms_processed += 1
                                
                                # Process backbone atoms
                                if atom.get_name() in backbone_atoms:
                                    atom_idx = backbone_atoms.index(atom.get_name())
                                    feature_volume[atom_idx, idx[2], idx[1], idx[0]] = 1.0
                                
                                # Add amino acid type if valid
                                if aa_idx >= 0:
                                    feature_volume[aa_idx, idx[2], idx[1], idx[0]] = 1.0
            
            self.logger.info(f"✓ Processed {residues_processed} residues, {atoms_processed} atoms")
            
            self.AF3_encodings = os.path.join(os.path.dirname(self.AF3_results), 'AF3_encodings')
            
            # Create output directory
            os.makedirs(self.AF3_encodings, exist_ok=True)            
            # Save each channel as separate MRC file
            successful_channels = 0
            failed_channels = 0
            
            total_cores = mp.cpu_count()
            max_workers = min(max(1, int(total_cores * 0.5)), 24)
            self.logger.info(f"Saving {num_channels} channels encodings using {max_workers} CPU cores in parallel")
            mp_args = []
            for channel_idx, channel_name in enumerate(channel_names):
                    channel_data = feature_volume[channel_idx]
                    mp_args.append((channel_idx, channel_name, channel_data, self.AF3_encodings))
                    
            with mp.Pool(processes=max_workers) as pool:
                results = pool.map(self.save_channel_encoding, mp_args)
            
            # Process results
            for result in results:
                if result['success']:
                    successful_channels += 1
                    self.logger.info(f"✓ Saved {result['channel_name']} encoding")
                else:
                    failed_channels += 1
                    self.logger.error(f"✗ Failed to save {result['channel_name']} encoding: {result['error']}")
                    
            if successful_channels == 24: success = True
                    
            if success == True:
                self.print_clean("")
                self.print_clean(f"🎯 Encoding Results:")
                self.print_clean(f"   ✅ Feature encoding successful")
                self.print_clean(f"   🧬 Channels created: {successful_channels}/{len(channel_names)}")
                self.print_clean(f"   🔬 Residues processed: {residues_processed}")
                self.print_clean(f"   ⚛️  Atoms processed: {atoms_processed}")
                self.print_clean(f"   📁 AF3 encodings directory: {self.AF3_encodings}")
                self.print_clean("")
        
                return success
            
        except Exception as e:
            error_msg = f"AF3 encoding failed: {str(e)}"
            self.print_clean(f"   ❌ Encoding failed: {error_msg}")
        return success


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description='AF3 Docked Structure Combiner - Combine multiple AF3 docked PDB structures into unified models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process AF3 results directory
  python combine_af3.py -a /path/to/AF3_results
  
  # Process with quiet output
  python combine_af3.py -a /path/to/AF3_results --quiet
        """
    )
    parser.add_argument('-m', '--map_path',
                       help='Path to cryo-EM density map file (.map, .mrc, .ccp4)')
    parser.add_argument('-a', '--AF3_results', required=True, 
                       help='AF3 results directory containing subdirectories with docked models')
    parser.add_argument('--quiet', action='store_true', 
                       help='Suppress detailed output')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.map_path):
        print(f"❌ Error: Cryo-EM density map path not found: {args.map_path}")
        sys.exit(1)
    if not os.path.exists(args.AF3_results):
        print(f"❌ Error: AF3 results directory not found: {args.AF3_results}")
        sys.exit(1)
    
    data_processor = DataPreprocessor(map_path=args.map_path, AF3_results=args.AF3_results, quiet=args.quiet)
    
    # Log startup information
    data_processor.print_clean("")
    data_processor.logger.info(f"AF3 results directory: {args.AF3_results}")
    data_processor.logger.info(f"Cryo-EM map path: {args.map_path}")
    
    try:
        data_processor.resample_and_normalize_map()  
        combined_docked_model_path = os.path.join(os.path.dirname(args.AF3_results), f'{os.path.basename(os.path.dirname(args.AF3_results))}_af3_docked.pdb')       
        if os.path.exists(combined_docked_model_path):
            encoding_result = data_processor.create_AF3_encodings(combined_docked_model_path) 
            
    except KeyboardInterrupt:
        data_processor.print_clean("")
        data_processor.print_clean("❌ Process interrupted by user")
        return 130
    except Exception as e:
        data_processor.logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())