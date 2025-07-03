#!/usr/bin/env python3
"""
MRC Map Processing Tool

This tool processes MRC/MAP files by resampling and normalizing them.

Author: Rajan Gyawali
Date: June 01, 2025
"""

import argparse
import os
from glob import glob
import mrcfile
import numpy as np
from scipy.ndimage import zoom


class MapProcessor:
    def __init__(self, input_map):
        """
        Initialize with input map
        input_map: Path to input MRC file
        """
        self.input_map = input_map
        with mrcfile.open(input_map) as mrc:
            self.data = mrc.data
            self.voxel_size = mrc.voxel_size
            self.origin = mrc.header.origin
            self.mapc = int(mrc.header.mapc)
            self.mapr = int(mrc.header.mapr)
            self.maps = int(mrc.header.maps)
            self.nxstart = int(mrc.header.nxstart)
            self.nystart = int(mrc.header.nystart)
            self.nzstart = int(mrc.header.nzstart)
    
    def resample(self, target_voxel_size=1.0):
        """Resample map to target voxel size"""
        # Calculate zoom factors
        zoom_factors = [self.voxel_size.x, self.voxel_size.y, self.voxel_size.z]
        
        # Resample data
        self.resampled_data = zoom(self.data, zoom_factors, order=3)
        self.target_voxel_size = target_voxel_size
        
        return self.resampled_data
    
    def normalize(self, data=None):
        """
        Normalize map data using threshold-based normalization
        """
        if data is None:
            data = self.resampled_data if hasattr(self, 'resampled_data') else self.data
            
        # Replace NaN values with 0
        data = np.nan_to_num(data)
        
        # Get median value
        median = np.median(data)
        
        # Threshold map by median and subtract median
        map_data_ = (data > median) * (data - median)
        
        # Get high percentile value for positive values
        positive_values = map_data_[np.where(map_data_ > 0)]
        if len(positive_values) > 0:
            percentile_value = np.percentile(positive_values, 99.9)
            
            if percentile_value != 0:
                # Clip values at percentile and normalize
                map_data_ = (map_data_ < percentile_value) * map_data_ + \
                           (map_data_ >= percentile_value) * percentile_value
                map_data_ /= percentile_value
                
                self.normalized_data = map_data_
                return self.normalized_data
            
        print('Error during normalization!!!')
        return None
    
    def process_map(self, output_path, target_voxel_size=1.0):
        """
        Complete processing pipeline:
        1. Resample
        2. Normalize
        3. Save
        """
        # Resample
        self.resample(target_voxel_size)
        
        # Normalize
        if self.normalize() is None:
            print("Processing failed during normalization")
            return
        
        # Save
        self.save_map(output_path)
        
    def save_map(self, output_path):
        """Save processed map"""
        if not hasattr(self, 'normalized_data'):
            raise ValueError("No processed data available to save")
            
        with mrcfile.new(output_path, overwrite=True) as mrc:
            mrc.set_data(self.normalized_data.astype(np.float32))
            mrc.voxel_size = (1.0, 1.0, 1.0)
            mrc.header.origin = self.origin
            mrc.header.mapc = self.mapc
            mrc.header.mapr = self.mapr
            mrc.header.maps = self.maps
            mrc.header.nxstart = self.nxstart
            mrc.header.nystart = self.nystart
            mrc.header.nzstart = self.nzstart            
            mrc.update_header_stats()


def main():
    parser = argparse.ArgumentParser(
        description="Process MRC/MAP files by resampling and normalizing them",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Author: Rajan Gyawali
        Date: June 01, 2025

        This tool processes MRC/MAP files through resampling and normalization:
        1. Resamples density maps to 1Å voxel size
        2. Applies threshold-based normalization using median and 99.9th percentile
        3. Saves processed maps as MRC files

        Example usage:
        %(prog)s
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
        try:
            emd_id = directory.split('/')[-1]
            input_density_map = f"{directory}/emd_{emd_id}.map"
            output_directory = f"{OUTPUT_DIR}/{emd_id}"
            os.makedirs(output_directory, exist_ok=True)
            normalized_density_map = f"{output_directory}/resampled_normalized_map.mrc"
            processor = MapProcessor(input_density_map)
            processor.process_map(
                normalized_density_map,
                target_voxel_size=1.0
            )
            print(f"Created resampled and normalized density map for EMD ID: {emd_id} | Completed {i + 1} density maps ...")
        except:
            print(f"Failed for normalizing map for EMD ID {emd_id}")


if __name__ == "__main__":
    main()