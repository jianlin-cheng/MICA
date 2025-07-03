#!/usr/bin/env python3
"""
Amino Acid Mask Generator

Author: Rajan Gyawali
Date: June 01, 2025

This tool generates amino acid masks from protein structures.
The mask identifies the 26 neighbors of each CA atom while maintaining the 
CA position itself as zero.
"""

import argparse
import os
from glob import glob
import numpy as np
from Bio.PDB import *
import mrcfile
from pathlib import Path
from typing import Tuple, List, Dict


class AminoAcidMaskGenerator:
    """
    A class to generate amino acid masks from protein structures.
    The mask identifies the 26 neighbors of each CA atom while maintaining the 
    CA position itself as zero.
    """
    
    def __init__(self, map_path: str):
        """
        Initialize the mask generator with a density map file.
        
        Args:
            map_path: Path to the MRC density map file
        """
        self.map_path = Path(map_path)
        
        # Standard amino acid mapping (1-20)
        self.aa_mapping = {
            'ALA': 1, 'CYS': 2, 'ASP': 3, 'GLU': 4, 'PHE': 5,
            'GLY': 6, 'HIS': 7, 'ILE': 8, 'LYS': 9, 'LEU': 10,
            'MET': 11, 'ASN': 12, 'PRO': 13, 'GLN': 14, 'ARG': 15,
            'SER': 16, 'THR': 17, 'VAL': 18, 'TRP': 19, 'TYR': 20
        }
        
        # Load map data and metadata
        self._load_map_data()
        
    def _load_map_data(self):
        """Load and store density map data and metadata."""
        try:
            with mrcfile.open(self.map_path) as mrc:
                self.map_data = mrc.data
                self.shape = mrc.data.shape
                self.voxel_size = mrc.voxel_size
                self.origin = mrc.header.origin
                
                # Store map axis order information
                self.mapc = int(mrc.header.mapc)
                self.mapr = int(mrc.header.mapr)
                self.maps = int(mrc.header.maps)
                
                # Store starting points
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
        
    def generate_mask(self, pdb_path: str) -> np.ndarray:
        """
        Generate amino acid mask from PDB structure.
        
        Args:
            pdb_path: Path to PDB structure file
            
        Returns:
            3D numpy array with amino acid labels for neighbor voxels
        """
        # Initialize mask with zeros
        mask = np.zeros(self.shape, dtype=np.int32)
        
        # Dictionary to track assigned positions and their amino acid types
        assigned_positions: Dict[Tuple[int, int, int], int] = {}
        
        try:
            # Parse PDB structure
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('protein', pdb_path)
            
            # First pass: collect all CA positions and their amino acid types
            ca_positions = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if ('CA' in residue and 
                            residue.get_resname() in self.aa_mapping):
                            # Get CA coordinates and amino acid type
                            coord = residue['CA'].get_coord()
                            idx = self.transform_coordinates(coord)
                            pos = (int(idx[2]), int(idx[1]), int(idx[0]))
                            aa_type = self.aa_mapping[residue.get_resname()]
                            ca_positions.append((pos, aa_type))
            
            # Second pass: assign neighbors
            for ca_pos, aa_type in ca_positions:
                # Get all valid neighbors
                neighbors = self.get_neighbors(ca_pos)
                
                # Assign amino acid type to neighbors
                for neighbor in neighbors:
                    # Only assign if not yet assigned or current type has priority
                    if (neighbor not in assigned_positions or 
                        aa_type < assigned_positions[neighbor]):
                        mask[neighbor] = aa_type
                        assigned_positions[neighbor] = aa_type
                
                # Ensure CA position itself is zero
                mask[ca_pos] = 0
                
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
        description="Generate amino acid masks from protein structures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Author: Rajan Gyawali
        Date: June 01, 2025

        This tool generates amino acid masks with the following features:
        - Maps 20 standard amino acids to labels 1-20
        - Identifies 26 neighbors of each CA atom
        - Maintains CA positions as zero
        - Uses priority system for overlapping neighbors

        Amino acid mapping:
        ALA:1, CYS:2, ASP:3, GLU:4, PHE:5, GLY:6, HIS:7, ILE:8, LYS:9, LEU:10,
        MET:11, ASN:12, PRO:13, GLN:14, ARG:15, SER:16, THR:17, VAL:18, TRP:19, TYR:20

        Example usage:
        %(prog)s
        %(prog)s --base-dir /path/to/raw/data --output-dir /path/to/processed/data
                """
    )
    
    parser.add_argument("--base-dir", default="Training_Dataset/Raw_Data",
                       help="Base directory containing subdirectories with MRC/MAP and PDB files")
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
        save_path = f"{output_directory}/amino_acid_mask.mrc"
        
        try:
            mask_gen = AminoAcidMaskGenerator(normalized_map)
            mask = mask_gen.generate_mask(pdb_file)
            mask_gen.save_mask(mask, save_path)
            print(f"Generated Amino Acid Mask for EMD ID: {emd_id} | Completed {i + 1} density maps ...")
        except Exception as e:
            print(f"Failed for density map with EMD ID: {emd_id} - Error: {str(e)}")


if __name__ == "__main__":
    main()