#!/usr/bin/env python3
"""
Uniform Grids Creator

This script performs automated grids creation for normalized cryo-EM density map and AF3 encodings.

Requirements:
- Path to normalized cryo-EM density map
- Path to directory containing AF3 encodings
- Base path to save computed grids

Author: Rajan Gyawali
Date: June 1, 2025
"""

from glob import glob
import numpy as np
import os
import mrcfile
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import logging
import time

class GridCreator:
    """
    Unified grid creator for density maps and AF3 encodings
    """
    
    def __init__(self, quiet=False):
        """
        Initialize grid creator
        
        Args:
            quiet: Whether to suppress detailed output
        """
        self.quiet = quiet
        self.setup_logging()
        self.processed_count = 0
        self.failed_count = 0
        self.failed_entries = []
    
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
    
    def transpose(self, numpy_image, axis_order, offset):
        """
        Apply transpose transformation to numpy array
        
        Args:
            numpy_image: Input numpy array
            axis_order: Order of axes transformation
            offset: Offset values
            
        Returns:
            Transposed image and transformed offset
        """
        trans_offset = []
        trans_order = []
        for i in range(3):
            for j in range(len(axis_order)):
                if axis_order[j] == i:
                    trans_offset.append(offset[j])
                    trans_order.append(j)
        image = np.transpose(numpy_image, trans_order)
        return image, trans_offset
    
    def create_grids_from_mrc(self, mrc_file, output_dir, grid_size=48, padding=8, file_prefix="grid"):
        """
        Create grids from MRC file and save each grid with its indices
        
        Args:
            mrc_file: Path to input MRC file
            output_dir: Directory to save grids
            grid_size: Size of processing grid (default 48)
            padding: Size of padding (default 8)
            file_prefix: Prefix for output files
            
        Returns:
            Number of grids created
        """
        try:
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
                nz_start = mrc.header.nzstart
                ny_start = mrc.header.nystart
                nx_start = mrc.header.nxstart
            
            # Apply transpose transformation
            axis_order = [int(maps) - 1, int(mapr) - 1, int(mapc) - 1]
            offset = [float(nz_start), float(ny_start), float(nx_start)]
            density_map, offset = self.transpose(density_map, axis_order, offset)
            
            # Get original shape
            orig_shape = density_map.shape
            self.logger.info(f"Creating grids for {os.path.basename(mrc_file)} with shape: {orig_shape}")
            
            # Calculate padding needed
            window_size = grid_size + 2 * padding  # 48 + 2*8 = 64
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
                        di = min(grid_size, orig_shape[0] - i)
                        dj = min(grid_size, orig_shape[1] - j)
                        dk = min(grid_size, orig_shape[2] - k)
                        
                        # Extract grid with padding
                        grid = padded_map[i:i+window_size,
                                        j:j+window_size,
                                        k:k+window_size]
                        
                        # Only save complete grids
                        if grid.shape == (window_size, window_size, window_size):
                            # Create filename with indices
                            filename = f"{file_prefix}_i{i}_j{j}_k{k}.npz"
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
            
            self.logger.info(f"✓ Created {grid_count} grids from {os.path.basename(mrc_file)}")
            return grid_count, offset
            
        except Exception as e:
            error_msg = f"Grid creation failed for {os.path.basename(mrc_file)}: {str(e)}"
            self.logger.error(f"✗ {error_msg}")
            return 0, None
    
    def process_single_mrc_task(self, args):
        """Helper function for multiprocessing"""
        mrc_file, output_dir, grid_size, padding, file_prefix = args
        try:
            grid_count, _ = self.create_grids_from_mrc(mrc_file, output_dir, grid_size, padding, file_prefix)
            return {
                "success": True,
                "file": os.path.basename(mrc_file),
                "grid_count": grid_count,
                "output_dir": output_dir
            }
        except Exception as e:
            return {
                "success": False,
                "file": os.path.basename(mrc_file),
                "error": str(e),
                "output_dir": output_dir
            }
    
    def create_normalized_map_grids(self, normalized_map_path, output_dir, grid_size=48, padding=8):
        """
        Create grids from normalized density map
        
        Args:
            normalized_map_path: Path to normalized MRC density map
            output_dir: Output directory for grids
            grid_size: Size of processing grid (default 48)
            padding: Size of padding (default 8)
            
        Returns:
            dict: Processing results
        """
        self.print_clean("")
        self.print_clean("=" * 80)
        self.print_clean("🗺️  CREATING GRIDS FOR NORMALIZED MAP")
        self.print_clean("=" * 80)
        
        start_time = time.time()
        
        # Validate input
        if not os.path.exists(normalized_map_path):
            error_msg = f"Normalized map not found: {normalized_map_path}"
            self.logger.error(error_msg)
            self.print_clean("")
            self.print_clean(f"💡 Expected normalized map: {normalized_map_path}")
            self.print_clean("")
            return {"success": False, "error": error_msg}
        
        self.logger.info(f"Processing normalized map: {os.path.basename(normalized_map_path)}")
        self.logger.info(f"Output directory: {output_dir}")
        self.logger.info(f"Grid size: {grid_size}, Padding: {padding}")
        
        # Create grids
        grid_count, offset = self.create_grids_from_mrc(
            mrc_file=normalized_map_path,
            output_dir=output_dir,
            grid_size=grid_size,
            padding=padding,
            file_prefix="normalized_map_grid"
        )
        
        processing_time = time.time() - start_time
        
        # Summary
        self.print_clean("")
        self.print_clean(f"🎯 Grid Creation Results:")
        if grid_count > 0:
            self.print_clean(f"   ✅ Successfully created: {grid_count} grids")
            self.print_clean(f"   📁 Output directory: {os.path.basename(output_dir)}")
            self.print_clean(f"   ⏱️  Processing time: {processing_time:.2f} seconds")
        else:
            self.print_clean(f"   ❌ Grid creation failed")
        self.print_clean("")
        
        return {
            "success": grid_count > 0,
            "grid_count": grid_count,
            "offset": offset,
            "output_directory": output_dir,
            "processing_time": processing_time,
            "input_file": normalized_map_path
        }
    
    def create_AF3_encodings_grids(self, AF3_encodings_path, output_dir, grid_size=48, padding=8, parallel=True):
        """
        Create grids from AF3 encoding channels
        
        Args:
            AF3_encodings_path: Path to directory containing AF3 encoding MRC files
            output_dir: Output directory for grids
            grid_size: Size of processing grid (default 48)
            padding: Size of padding (default 8)
            parallel: Whether to use parallel processing (default True)
            
        Returns:
            dict: Processing results
        """
        self.print_clean("")
        self.print_clean("=" * 80)
        self.print_clean("🧬 CREATING GRIDS FOR AF3 ENCODINGS")
        self.print_clean("=" * 80)
        
        start_time = time.time()
        
        # Validate input directory
        if not os.path.exists(AF3_encodings_path):
            error_msg = f"AF3 encodings directory not found: {AF3_encodings_path}"
            self.logger.error(error_msg)
            self.print_clean("")
            self.print_clean(f"💡 Expected AF3 encodings directory: {AF3_encodings_path}")
            self.print_clean("")
            return {"success": False, "error": error_msg}
        
        # Find all MRC encoding files
        encoding_files = glob(os.path.join(AF3_encodings_path, "*_encoding.mrc"))
        
        if not encoding_files:
            error_msg = f"No AF3 encoding files found in {AF3_encodings_path}"
            self.logger.error(error_msg)
            self.print_clean("")
            self.print_clean("💡 Expected file pattern: *_encoding.mrc")
            self.print_clean("")
            return {"success": False, "error": error_msg}
        
        self.logger.info(f"Found {len(encoding_files)} AF3 encoding files")
        self.logger.info(f"Output directory: {output_dir}")
        
        # Prepare tasks for processing
        tasks = []
        for encoding_file in encoding_files:
            # Extract channel name from filename (e.g., "CA_encoding.mrc" -> "CA")
            channel_name = os.path.basename(encoding_file).split("_encoding.mrc")[0]
            channel_output_dir = os.path.join(output_dir, f"{channel_name}_grids")
            
            tasks.append((
                encoding_file,
                channel_output_dir,
                grid_size,
                padding,
                f"{channel_name}_grid"
            ))
        
        successful_channels = 0
        failed_channels = 0
        total_grids = 0
        processing_errors = []
        
        if parallel and len(tasks) > 1:
            # Parallel processing
            total_cores = mp.cpu_count()
            max_workers = min(max(1, int(total_cores * 0.5)), len(tasks))
            
            self.logger.info(f"Processing {len(tasks)} channels using {max_workers} CPU cores in parallel")
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self.process_single_mrc_task, task): task for task in tasks}
                
                for future in as_completed(futures):
                    result = future.result()
                    if result["success"]:
                        successful_channels += 1
                        total_grids += result["grid_count"]
                    else:
                        failed_channels += 1
                        error_msg = f"Failed {result['file']}: {result['error']}"
                        processing_errors.append(error_msg)
                        self.logger.error(f"✗ {error_msg}")
        else:
            # Sequential processing
            self.logger.info(f"Processing {len(tasks)} channels sequentially")
            
            for task in tasks:
                result = self.process_single_mrc_task(task)
                if result["success"]:
                    successful_channels += 1
                    total_grids += result["grid_count"]
                    self.logger.info(f"✓ {result['file']}: {result['grid_count']} grids")
                else:
                    failed_channels += 1
                    error_msg = f"Failed {result['file']}: {result['error']}"
                    processing_errors.append(error_msg)
                    self.logger.error(f"✗ {error_msg}")
        
        processing_time = time.time() - start_time
        
        # Summary
        self.print_clean("")
        self.print_clean(f"🎯 AF3 Encoding Grid Results:")
        if successful_channels > 0:
            self.print_clean(f"   ✅ Successfully processed: {successful_channels}/{len(encoding_files)} channels")
            self.print_clean(f"   🔢 Total grids created: {total_grids}")
            self.print_clean(f"   📁 Output directory: {output_dir}")
            self.print_clean(f"   ⏱️  Processing time: {processing_time:.2f} seconds")
            if failed_channels > 0:
                self.print_clean(f"   ⚠️  Failed channels: {failed_channels}")
        else:
            self.print_clean(f"   ❌ All channel processing failed")
            if processing_errors:
                self.print_clean(f"   ⚠️  Processing errors: {len(processing_errors)}")
        self.print_clean("")
        
        return {
            "success": successful_channels > 0,
            "successful_channels": successful_channels,
            "failed_channels": failed_channels,
            "total_channels": len(encoding_files),
            "total_grids": total_grids,
            "output_directory": output_dir,
            "processing_time": processing_time,
            "processing_errors": processing_errors,
            "input_directory": AF3_encodings_path
        }


# Example usage and testing
def main():
    """Example usage of the GridCreator class"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create grids from normalized maps and AF3 encodings')
    parser.add_argument('--normalized_map', help='Path to normalized MRC map')
    parser.add_argument('--AF3_encodings', help='Path to AF3 encodings directory')
    parser.add_argument('--output_dir', help='Output directory for grids')
    parser.add_argument('--no_parallel', action='store_true', help='Disable parallel processing')
    parser.add_argument('--quiet', action='store_true', help='Suppress detailed output')
    
    args = parser.parse_args()
    
    if not args.normalized_map and not args.AF3_encodings:
        print("❌ Error: Must specify either --normalized_map or --AF3_encodings or both")
        parser.print_help()
        return 1
    
    if not args.output_dir:
        print("❌ Error: Must specify --output_dir")
        parser.print_help()
        return 1
    
    # Create grid creator
    grid_creator = GridCreator(quiet=args.quiet)
    
    results = {}
    overall_success = True
    
    # Process normalized map if specified
    if args.normalized_map:
        map_output_dir = os.path.join(args.output_dir, "normalized_map_grids")
        result = grid_creator.create_normalized_map_grids(
            normalized_map_path=args.normalized_map,
            output_dir=map_output_dir,
        )
        results['normalized_map'] = result
        if not result["success"]:
            overall_success = False
    
    # Process AF3 encodings if specified
    if args.AF3_encodings:
        af3_output_dir = os.path.join(args.output_dir, "AF3_encoding_grids")
        result = grid_creator.create_AF3_encodings_grids(
            AF3_encodings_path=args.AF3_encodings,
            output_dir=af3_output_dir,
            parallel=not args.no_parallel
        )
        results['AF3_encodings'] = result
        if not result["success"]:
            overall_success = False
    
    # Final summary
    grid_creator.print_clean("=" * 60)
    if overall_success:
        grid_creator.print_clean("🏁 CREATING GRIDS COMPLETE")
    else:
        grid_creator.print_clean("❌  CREATING GRIDS FAILED")
    grid_creator.print_clean("=" * 60)
    grid_creator.print_clean("")
    
    return 0 if overall_success else 1


if __name__ == "__main__":
    main()