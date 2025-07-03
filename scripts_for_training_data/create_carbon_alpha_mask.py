#!/usr/bin/env python3
"""
Carbon Alpha Mask Generator

Author: Rajan Gyawali
Date: June 01, 2025

This tool generates carbon alpha masks with the following labels:
0: Non-structural elements (background)
1: Neighboring voxels (26-neighbor connectivity)
2: Non-CA atoms
3: Carbon Alpha (CA) atoms
"""

import argparse
import os
from glob import glob
import numpy as np
import mrcfile
from Bio.PDB import *
from pathlib import Path
from typing import Tuple, List, Dict


class CarbonAlphaMask:
    """
    Generate carbon alpha masks with the following labels:
    0: Non-structural elements (background)
    1: Neighboring voxels (26-neighbor connectivity)
    2: Non-CA atoms
    3: Carbon Alpha (CA) atoms
    """
    
    def __init__(self, map_path: str):
        """
        Initialize the mask generator with a density map file.
        
        Args:
            map_path: Path to the MRC density map file
        """
        self.map_path = Path(map_path)
        self._load_map_data()
        
    def _load_map_data(self):
        """Load and store density map data and metadata."""
        try:
            with mrcfile.open(self.map_path) as mrc:
                self.map_data = mrc.data
                self.shape = mrc.data.shape
                self.voxel_size = mrc.voxel_size
                self.origin = mrc.header.origin
                
                # Store map metadata
                self.mapc = int(mrc.header.mapc)
                self.mapr = int(mrc.header.mapr)
                self.maps = int(mrc.header.maps)
                self.nxstart = int(mrc.header.nxstart)
                self.nystart = int(mrc.header.nystart)
                self.nzstart = int(mrc.header.nzstart)
                
        except Exception as e:
            print(f"Failed to load map file {self.map_path}: {str(e)}")
            raise
            
    def transform_coordinates(self, coord: np.ndarray) -> np.ndarray:
        """
        Transform PDB coordinates to density map indices.
        
        Args:
            coord: Array of [x, y, z] coordinates from PDB
            
        Returns:
            Array of corresponding indices in the density map
        """
        # Shift coordinates by map origin
        coord_shifted = coord - np.array([self.origin.x, 
                                        self.origin.y, 
                                        self.origin.z])
        
        # Convert to map indices (assuming 1Å voxel size)
        indices = coord_shifted / 1.0
        
        # Round to nearest integer and ensure within bounds
        indices = np.round(indices).astype(int)
        indices = np.clip(indices, 0, np.array(self.shape) - 1)
        
        return indices
        
    def get_neighbors(self, center: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """
        Get all 26 valid neighbors for a given position.
        
        Args:
            center: (x, y, z) coordinates of center position
            
        Returns:
            List of valid neighbor positions
        """
        neighbors = []
        x, y, z = center
        
        # Iterate through all 26 possible neighbors
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    # Skip the center position
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                        
                    new_x, new_y, new_z = x + dx, y + dy, z + dz
                    
                    # Check if position is within map boundaries
                    if (0 <= new_x < self.shape[0] and
                        0 <= new_y < self.shape[1] and
                        0 <= new_z < self.shape[2]):
                        neighbors.append((new_x, new_y, new_z))
                        
        return neighbors
        
    def generate_mask(self, pdb_file: str) -> np.ndarray:
        """
        Generate mask for protein structure.
        
        Args:
            pdb_file: Path to PDB structure file
            
        Returns:
            3D numpy array with labels:
            0: background
            1: neighboring voxels
            2: non-CA atoms
            3: CA atoms
        """
        # Initialize mask with zeros
        mask = np.zeros(self.shape, dtype=np.int32)
        
        # Dictionary to track assigned positions
        assigned_positions: Dict[Tuple[int, int, int], int] = {}
        
        try:
            # Parse PDB structure
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('protein', pdb_file)
            
            # First pass: mark all atoms
            for model in structure:
                for chain in model:
                    for residue in chain:
                        for atom in residue:
                            coord = atom.get_coord()
                            idx = self.transform_coordinates(coord)
                            pos = (int(idx[2]), int(idx[1]), int(idx[0]))
                            
                            # Assign value based on atom type
                            if atom.get_name() == 'CA':
                                mask[pos] = 3  # CA atoms
                                assigned_positions[pos] = 3
                            else:
                                mask[pos] = 2  # Other atoms
                                assigned_positions[pos] = 2
            
            # Second pass: mark neighbors
            atom_positions = list(assigned_positions.keys())
            for pos in atom_positions:
                neighbors = self.get_neighbors(pos)
                
                # Mark unassigned neighbors with 1
                for neighbor in neighbors:
                    if neighbor not in assigned_positions:
                        mask[neighbor] = 1
                        assigned_positions[neighbor] = 1
                        
        except Exception as e:
            print(f"Error generating mask: {str(e)}")
            raise
            
        return mask
        
    def save_mask(self, mask: np.ndarray, output_path: str):
        """
        Save the generated mask as an MRC file.
        
        Args:
            mask: 3D numpy array of mask data
            output_path: Path where mask will be saved
        """
        try:
            with mrcfile.new(output_path, overwrite=True) as mrc:
                # Save mask data
                mrc.set_data(mask.astype(np.float32))
                
                # Copy metadata from input map
                mrc.header.origin = self.origin
                mrc.header.nxstart = self.nxstart
                mrc.header.nystart = self.nystart
                mrc.header.nzstart = self.nzstart
                mrc.header.mapc = self.mapc
                mrc.header.mapr = self.mapr
                mrc.header.maps = self.maps
                mrc.voxel_size = 1
                
                # Update statistics
                mrc.update_header_stats()
                
        except Exception as e:
            print(f"Error saving mask to {output_path}: {str(e)}")
            raise 


def main():
    parser = argparse.ArgumentParser(
        description="Generate Carbon Alpha masks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Author: Rajan Gyawali
        Date: June 01, 2025

        This tool generates masks with the following labels:
        0: Background (non-structural elements)
        1: Neighboring voxels (26-neighbor connectivity)
        2: Non-CA atoms
        3: Carbon Alpha (CA) atoms

        Example usage:
        %(prog)s --base-dir /path/to/raw/data --output-dir /path/to/processed/data
                """
    )
    
    parser.add_argument("--base-dir", default="Training_Dataset/Raw_Data",
                       help="Base directory containing subdirectories with MRC/MAP files")
    parser.add_argument("--output-dir", default="Training_Dataset/Processed_Data",
                       help="Output directory for processed files")
    
    args = parser.parse_args()
    
    BASE_DIR = sorted(glob(f"{args.base_dir}/*"))
    OUTPUT_DIR = args.output_dir

    for i, directory in enumerate(BASE_DIR):
        emd_id = directory.split('/')[-1]
        pdb_files = glob(f"{directory}/*.pdb")
        
        # Find PDB file with 8-character filename
        pdb_file = None
        for p in pdb_files:
            if len(p.split("/")[-1]) == 8:
                pdb_file = p
                break
        
        if pdb_file is None:
            print(f"No suitable PDB file found for EMD ID: {emd_id}")
            continue
            
        output_directory = f"{OUTPUT_DIR}/{emd_id}"
        os.makedirs(output_directory, exist_ok=True)
        normalized_map = f"{output_directory}/resampled_normalized_map.mrc"
        save_path = f"{output_directory}/carbon_alpha_mask.mrc"
        
        try:
            mask_gen = CarbonAlphaMask(normalized_map)
            mask = mask_gen.generate_mask(pdb_file)
            mask_gen.save_mask(mask, save_path)
            print(f"Generated Carbon Alpha Mask for EMD ID: {emd_id} | Completed {i + 1} density maps ...")
        except Exception as e:
            print(f"Failed for density map with EMD ID: {emd_id} - Error: {str(e)}")


if __name__ == "__main__":
    main()    
        
