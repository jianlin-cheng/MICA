#!/usr/bin/env python3
"""
Features Encoder

Author: Rajan Gyawali
Date: June 01, 2025

This tool encodes protein structures into feature volumes with channels for
backbone atoms and amino acid types in specified order.
"""

import argparse
import os
from glob import glob
import numpy as np
from Bio.PDB import *
import mrcfile


class FeaturesEncoder:
    def __init__(self, map_file):
        """
        Initialize encoder with a density map file
        map_file: Path to reference density map (.mrc)
        """
        with mrcfile.open(map_file) as mrc:
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

        # Define backbone atoms
        self.backbone_atoms = ['CA', 'N', 'C', 'O']
        
        # Define amino acids in specified order
        self.amino_acids = [
            'ALA', 'CYS', 'ASP', 'GLU', 'PHE', 
            'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
            'MET', 'ASN', 'PRO', 'GLN', 'ARG', 
            'SER', 'THR', 'VAL', 'TRP', 'TYR'
        ]
        
        # Total channels = backbone atoms + amino acids
        self.num_channels = len(self.backbone_atoms) + len(self.amino_acids)

    def transform_coordinates(self, coord):
        """
        Transform PDB coordinates to density map indices
        """
        coord_shifted = coord - np.array((self.origin.x, self.origin.y, self.origin.z))
        indices = coord_shifted / 1.0
        indices = np.round(indices).astype(int)
        indices = np.clip(indices, 0, np.array(self.shape) - 1)
        return indices

    def get_aa_channel_index(self, residue_name):
        """Get channel index for amino acid type"""
        try:
            return len(self.backbone_atoms) + self.amino_acids.index(residue_name)
        except ValueError:
            return -1

    def encode_structure(self, pdb_file):
        """
        Encode protein structure into feature volume with channels for
        backbone atoms and amino acid types in specified order
        """
        # Initialize feature volume (channels, depth, height, width)
        feature_volume = np.zeros((self.num_channels, *self.shape))

        # Parse structure
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_file)

        # Encode features
        for model in structure:
            for chain in model:
                for residue in chain:
                    # Only process standard amino acids
                    if residue.get_id()[0] == ' ':
                        res_name = residue.get_resname()
                        
                        # Get amino acid channel index
                        aa_idx = self.get_aa_channel_index(res_name)
                        
                        for atom in residue:
                            # Transform coordinates to map indices
                            
                            idx = self.transform_coordinates(atom.get_coord())
                            
                            # Process backbone atoms
                            if atom.get_name() in self.backbone_atoms:
                                atom_idx = self.backbone_atoms.index(atom.get_name())
                                feature_volume[atom_idx, idx[2], idx[1], idx[0]] = 1.0
                            
                            # Add amino acid type if valid
                            if aa_idx >= 0:
                                feature_volume[aa_idx, idx[2], idx[1], idx[0]] = 1.0

        return feature_volume

    def save_channel_as_mrc(self, feature_volume, output_path, channel_idx=0):
        """Save a specific feature channel as MRC file"""
        channel_data = feature_volume[channel_idx]
        
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

    def get_channel_names(self):
        """Get list of all channel names"""
        return self.backbone_atoms + self.amino_acids


def main():
    parser = argparse.ArgumentParser(
        description="Encode protein structures into feature volumes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Author: Rajan Gyawali
    Date: June 01, 2025

    This tool encodes protein structures into multi-channel feature volumes:
    - 4 channels for backbone atoms (CA, N, C, O)
    - 20 channels for amino acid types
    - Total: 24 channels per structure

    Features:
    - Binary encoding (1.0 for atom presence, 0.0 for absence)
    - Coordinate transformation from PDB to map indices
    - Individual channel saving as MRC files
    - Detailed occupancy statistics

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
        
        # Find PDB file matching the pattern
        pdb_files = glob(f"{directory}/*af3_docked*.pdb")
        if not pdb_files:
            print(f"No AF3 docked PDB file found for EMD ID: {emd_id}")
            continue
        
        pdb_file = pdb_files[0]
        output_directory = f"{OUTPUT_DIR}/{emd_id}"
        os.makedirs(output_directory, exist_ok=True)
        normalized_map = f"{output_directory}/resampled_normalized_map.mrc"
        
        try:
            encoder = FeaturesEncoder(normalized_map)
            feature_volume = encoder.encode_structure(pdb_file)
            
            # Save individual channels
            for j, name in enumerate(encoder.get_channel_names()):
                output_path = f"{output_directory}/{name}_encoding.mrc"
                encoder.save_channel_as_mrc(feature_volume, output_path, channel_idx=j)
            
            print(f"Generated feature encodings for EMD ID: {emd_id} | Completed {i + 1} density maps ...")
        except Exception as e:
            print(f"Failed for density map with EMD ID: {emd_id} - Error: {str(e)}")


if __name__ == "__main__":
    main()