#!/usr/bin/env python3
"""
Encoding Grid Generator

Author: Rajan Gyawali
Date: June 01, 2025

This tool creates grids from various encoding files and saves each grid with its indices.
"""

import argparse
import os
from glob import glob
import numpy as np
import mrcfile
from tqdm import tqdm


def create_and_save_grids(mrc_file, output_dir, grid_size=48, padding=8):
    """
    Create grids from AF3 encoding and save each grid with its indices
    
    Args:
        mrc_file: Path to input MRC file
        output_dir: Directory to save grids
        grid_size: Size of processing grid (default 48)
        padding: Size of padding (default 8)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read MRC file
    with mrcfile.open(mrc_file) as mrc:
        density_map = mrc.data
        voxel_size = mrc.voxel_size
        origin = mrc.header.origin
        mapc = mrc.header.mapc
        mapr = mrc.header.mapr
        maps = mrc.header.maps

    # Get original shape
    orig_shape = density_map.shape
    
    # Calculate padding needed
    window_size = grid_size + 2*padding  # 48 + 2*8 = 64
    pad_end_x = window_size - (orig_shape[0] % grid_size)
    pad_end_y = window_size - (orig_shape[1] % grid_size)
    pad_end_z = window_size - (orig_shape[2] % grid_size)
    
    # Pad the density map
    padded_map = np.pad(density_map,
                        [(padding, pad_end_x),
                         (padding, pad_end_y),
                         (padding, pad_end_z)],
                        'constant')
    
    # Create and save grids
    grid_count = 0
    for i in range(0, orig_shape[0], grid_size):
        for j in range(0, orig_shape[1], grid_size):
            for k in range(0, orig_shape[2], grid_size):
                # Get grid dimensions
                di = min(grid_size, orig_shape[0]-i)
                dj = min(grid_size, orig_shape[1]-j)
                dk = min(grid_size, orig_shape[2]-k)
                
                # Extract grid with padding
                grid = padded_map[i:i+window_size,
                                j:j+window_size,
                                k:k+window_size]
                
                # Only save complete grids
                if grid.shape == (window_size, window_size, window_size):
                    # Create filename with indices
                    filename = f"grid_i{i}_j{j}_k{k}.npz"
                    filepath = os.path.join(output_dir, filename)
                    
                    # Save grid with its metadata
                    np.savez(filepath,
                            grid=grid,
                            i=i, j=j, k=k,
                            di=di, dj=dj, dk=dk,
                            orig_shape=orig_shape,
                            grid_size=grid_size,
                            padding=padding,
                            voxel_size=voxel_size,
                            origin=origin,
                            mapc=mapc,
                            mapr=mapr,
                            maps=maps)
                    
                    grid_count += 1
                    
    return grid_count


def main():
    parser = argparse.ArgumentParser(
        description="Create grids from various encoding files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Author: Rajan Gyawali
        Date: June 01, 2025

        This tool creates grids from various encoding files with the following features:
        - Automatically detects and processes all MRC files in each directory
        - Organizes output by encoding type (extracted from filename)
        - Splits large encoding files into manageable grid chunks
        - Adds padding around each grid for context
        - Saves all grids regardless of content (no filtering)
        - Saves grids with complete metadata for reconstruction

        Grid Processing:
        - Default grid size: 48x48x48 voxels
        - Default padding: 8 voxels on each side
        - Total window size: 64x64x64 voxels (48 + 2*8)
        - No density filtering (saves all complete grids)

        Encoding Types Detected:
        - Automatically extracts encoding type from filename prefix
        - Common types: CA, N, C, O (backbone atoms), ALA, CYS, etc. (amino acids)
        - Output organized as: {output_dir}/{encoding_type}_encodings/{emd_id}/

        Output:
        - Each grid saved as NPZ file with indices and metadata
        - Filename format: grid_i{x}_j{y}_k{z}.npz
        - Directory structure: {encoding_type}_encodings/{emd_id}/

Example usage:
  %(prog)s
  %(prog)s --base-dir /path/to/processed/data --output-dir /path/to/grids
        """
    )
    
    parser.add_argument("--base-dir", default="Training_Dataset/Processed_Data",
                       help="Base directory containing processed files with various encodings")
    parser.add_argument("--output-dir", default="Training_Dataset/Grids",
                       help="Output directory for encoding grid files")
    parser.add_argument("--grid-size", type=int, default=48,
                       help="Size of processing grid (default: 48)")
    parser.add_argument("--padding", type=int, default=8,
                       help="Size of padding around each grid (default: 8)")
    
    args = parser.parse_args()
    
    BASE_DIR = sorted(glob(f"{args.base_dir}/*"))
    OUTPUT_DIR = args.output_dir

    for i, directory in enumerate(BASE_DIR):
        emd_id = directory.split('/')[-1]
        encoding_files = glob(f"{directory}/*encoding*.mrc")
        
        for encoding_file in tqdm(encoding_files, 
                                desc=f"Creating grids for AF3 encoding files for {emd_id}", 
                                leave=False,
                                unit="file"):
            encoding_type = encoding_file.split("/")[-1].split("_")[0]
            output_directory = f"{OUTPUT_DIR}/{encoding_type}_encodings/{emd_id}"
            os.makedirs(output_directory, exist_ok=True)
            
            try:
                grid_count = create_and_save_grids(encoding_file, output_directory,
                                                args.grid_size, args.padding)
            except Exception as e:
                tqdm.write(f"✗ Grid creation failed for {emd_id}, encoding: {encoding_type} - Error: {str(e)}")
            
        tqdm.write(f"✓ Created {grid_count} grids of each encoding type for EMD ID: {emd_id} | Completed {i + 1} density maps")


if __name__ == "__main__":
    main()