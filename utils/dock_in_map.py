#!/usr/bin/env python3
"""
Phenix Docking Tool for CryoEM Map

This script performs automated docking of atomic models into a CryoEM 
density map using Phenix dock_in_map. It processes atomic models iteratively 
with map masking to prevent overlapping placements. Docking can be done either domainwise or chainwise. 
The default docking strategy is domainwise.

Requirements:
- Phenix software suite and its location
- Cryo-EM density map, its recommended contour level and resolution
- FASTA sequence
- Docking strategy: domainwise/chainwise, default is domainwise
- Path to AlphaFold3 Results

Author: Rajan Gyawali
Date: June 1, 2025
"""

import argparse
import os
import shutil
import subprocess
import sys
import logging
import datetime
from glob import glob

import mrcfile
import numpy as np
from Bio import PDB
from scipy.ndimage import distance_transform_edt


class PhenixDockingProcessor:
    """Main class for Phenix docking operations."""
    
    def __init__(self, phenix_command, AF3_results, log_directory=None, quiet=True):
        """
        Initialize the docking processor.
        
        Args:
            phenix_command: Command to activate Phenix environment
            log_directory: Directory to store log files
        """
        self.phenix_command = phenix_command
        self.AF3_results = AF3_results
        self.parser = PDB.PDBParser(QUIET=quiet)
        self.io = PDB.PDBIO()
        self.failed_models = []
        
        # Setup logging
        self.log_directory = log_directory or 'docking_logs'
        os.makedirs(self.log_directory, exist_ok=True)
        
        # Create timestamp for this run
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup unified logger (preserve original functionality)
        self.setup_logger()
        
        # Setup visual output
        self.setup_visual_logging()
        
        self.print_clean("")
        self.print_clean("🧬 PHENIX DOCKING TOOL")
        self.print_clean(f"⏰ Session started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.print_clean(f"📄 Log file: {self.main_log_file}")
        self.print_clean("")
        
        self.logger.info("PHENIX DOCKING TOOL - SESSION STARTED")
        self.logger.info(f"Timestamp: {self.timestamp}")
        
        # Test Phenix environment
        self.test_phenix_environment()

    def setup_visual_logging(self):
        """Setup visual logging for user-friendly output."""
        # This method provides clean output without timestamps for visual appeal
        pass

    def print_clean(self, message):
        """Print message without timestamp or log level for visual appeal."""
        print(message)

    def test_phenix_environment(self):
        """Test if Phenix environment can be activated and dock_in_map is available."""
        self.print_clean("=" * 80)
        self.print_clean("🔧 TESTING PHENIX ENVIRONMENT")
        self.print_clean("=" * 80)
        
        self.logger.info("Testing Phenix environment...")
        
        try:
            # Test command to check if Phenix is available
            test_cmd = f"bash -c 'source {self.phenix_command} && which phenix.dock_in_map'"
            
            result = subprocess.run(
                test_cmd,
                shell=True,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                phenix_path = result.stdout.strip()
                self.logger.info(f"✓ Phenix environment activated successfully")
                self.logger.info(f"✓ phenix.dock_in_map found at: {phenix_path}")
                self.print_clean("✅ Phenix environment test PASSED")
                self.print_clean(f"   📍 dock_in_map location: {phenix_path}")
                self.print_clean("")
            else:
                error_msg = f"Failed to activate Phenix environment or find dock_in_map"
                self.logger.error(f"✗ {error_msg}")
                self.logger.error(f"  Error output: {result.stderr}")
                self.print_clean("❌ Phenix environment test FAILED")
                self.print_clean(f"   💬 Error: {result.stderr}")
                raise RuntimeError(error_msg)
                
        except subprocess.TimeoutExpired:
            error_msg = "Timeout while testing Phenix environment"
            self.logger.error(error_msg)
            self.print_clean("❌ Timeout while testing Phenix environment")
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"Error testing Phenix environment: {str(e)}"
            self.logger.error(error_msg)
            self.print_clean(f"❌ Error testing Phenix environment: {str(e)}")
            raise

    def setup_logger(self):
        """Setup the unified logger for all operations."""
        self.logger = logging.getLogger('phenix_docking')
        self.logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler for unified log
        self.main_log_file = os.path.join(self.log_directory, f'phenix_docking_{self.timestamp}.log')
        file_handler = logging.FileHandler(self.main_log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Console handler for technical logs only
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)  # Only show warnings/errors in technical logs
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def run_phenix_dock_in_map(self, density_map, atomic_model, resolution, save_file, fasta_file):
        """
        Run Phenix dock_in_map command for a single atomic model.
        
        Args:
            density_map: Path to density map file
            atomic_model: Path to atomic model PDB file
            resolution: Resolution value for docking
            save_file: Output path for docked model
            fasta_file: Path to FASTA sequence file
            
        Returns:
            bool: True if docking successful, False otherwise
        """
        model_name = os.path.basename(atomic_model)
        
        # Build Phenix command arguments
        phenix_args = [
            'phenix.dock_in_map',
            f'search_model={atomic_model}',
            f'map_file={density_map}',
            'nproc=16',
            'quick=False',
            f'resolution={resolution}',
            f'pdb_out={save_file}',
            'dock_chains_individually=True',
            f'sequence={fasta_file}'
        ]
        
        # Create the full command with environment activation
        full_command = f"bash -c 'source {self.phenix_command} && {' '.join(phenix_args)}'"
        
        # Log the command in main log
        self.logger.info(f"PHENIX COMMAND for {model_name}: {full_command}")
        
        try:
            # Create individual log file for this docking run
            individual_log = os.path.join(self.log_directory, f'phenix_{model_name}_{self.timestamp}.log')
            
            with open(individual_log, 'w') as log_file:
                # Run the command with proper shell execution
                result = subprocess.run(
                    full_command,
                    shell=True,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True
                )
            
            # Read the log file to check for success indicators
            with open(individual_log, 'r') as log_file:
                log_content = log_file.read()
            
            if result.returncode == 0:
                self.logger.info(f"PHENIX SUCCESS: Docking completed for {model_name}")
                
                # Log success metrics if found in output
                if "Final score" in log_content:
                    for line in log_content.split('\n'):
                        if "Final score" in line or "CC" in line:
                            self.logger.info(f"PHENIX METRICS: {line.strip()}")
                
                return True
            else:
                self.logger.error(f"PHENIX FAILED: {model_name} - return code {result.returncode}")
                
                # Log first error found
                error_lines = [line for line in log_content.split('\n') if 'error' in line.lower()]
                if error_lines:
                    self.logger.error(f"PHENIX ERROR: {error_lines[0].strip()}")
                
                return False
                
        except subprocess.TimeoutExpired:
            error_msg = f"Timeout: Docking for {model_name} took longer than 1 hour"
            self.logger.error(f"PHENIX TIMEOUT: {error_msg}")
            return False
        except FileNotFoundError:
            error_msg = "Bash shell not found. This script requires bash to activate Phenix environment"
            self.logger.error(f"PHENIX ERROR: {error_msg}")
            return False
        except Exception as e:
            error_msg = f"Error running docking command: {str(e)}"
            self.logger.error(f"PHENIX ERROR: {error_msg}")
            return False

    def initial_map_processing(self, input_map, output_map_path, contour_level):
        """
        Process initial density map by applying contour level threshold.
        
        Args:
            input_map: Path to input map file
            output_map_path: Path for output processed map
            contour_level: Contour level threshold value
            
        Returns:
            str: Path to processed map file
        """
        self.logger.info(f"Processing initial map with contour level {contour_level}")
        
        try:
            with mrcfile.open(input_map, mode='r') as mrc:
                data = mrc.data.copy()
                voxel_size = mrc.voxel_size
                origin = mrc.header.origin
            
            # Apply contour level threshold
            clipped_data = np.where(data < contour_level, 0, data)
            
            # Save processed map
            with mrcfile.new(output_map_path, overwrite=True) as mrc:
                mrc.set_data(clipped_data.astype(np.float32))
                mrc.voxel_size = voxel_size
                mrc.header.origin = origin
            
            self.logger.info(f"Initial map processed and saved to: {output_map_path}")
            return output_map_path
            
        except Exception as e:
            error_msg = f"Error processing initial map: {str(e)}"
            self.logger.error(error_msg)
            raise

    def subsequent_map_processing(self, input_map_path, pdb_file_path, output_map_path, 
                                radius=2.0, percentage=40, centroid_method='median'):
        """
        Process density map by masking within radius of docked structure.
        
        This function masks out density around a docked structure to prevent
        subsequent dockings from overlapping with already placed models.
        
        Args:
            input_map_path: Path to input density map
            pdb_file_path: Path to docked PDB structure
            output_map_path: Path for output masked map
            radius: Masking radius in Angstroms (default: 2.0)
            percentage: Percentage of atoms closest to centroid to use for masking (default: 40)
            centroid_method: Method to calculate centroid ('mean' or 'median', default: 'median')
            
        Returns:
            str: Path to processed map file
        """
        model_name = os.path.basename(pdb_file_path)
        
        try:
            # Read atomic coordinates
            parser = PDB.PDBParser(QUIET=True)
            structure = parser.get_structure('protein', pdb_file_path)
            coords = np.array([atom.get_coord() for atom in structure.get_atoms()])
            
            # Calculate centroid
            if centroid_method == 'mean':
                centroid = np.mean(coords, axis=0)
            elif centroid_method == 'median':
                centroid = np.median(coords, axis=0)
            else:
                raise ValueError(f"Unknown centroid method: {centroid_method}")
            
            # Select atoms closest to centroid
            distances_to_centroid = np.sqrt(np.sum((coords - centroid)**2, axis=1))
            num_coords_to_use = int(len(coords) * (percentage / 100.0))
            closest_indices = np.argsort(distances_to_centroid)[:num_coords_to_use]
            selected_coords = coords[closest_indices]
            
            # Process map
            with mrcfile.open(input_map_path, mode='r') as mrc:
                map_data = mrc.data
                voxel_size = np.array([mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z])
                origin = np.array([mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z])
                
                # Convert coordinates to voxel space
                selected_voxel_coords = ((selected_coords - origin) / voxel_size).astype(int)
                
                # Create binary mask
                mask = np.zeros_like(map_data, dtype=bool)
                valid_coords = (
                    (selected_voxel_coords >= 0) & 
                    (selected_voxel_coords < np.array(map_data.shape))
                ).all(axis=1)
                
                selected_voxel_coords = selected_voxel_coords[valid_coords]
                mask[selected_voxel_coords[:, 2], selected_voxel_coords[:, 1], selected_voxel_coords[:, 0]] = True
                
                # Apply distance transform for radius-based masking
                distances = distance_transform_edt(~mask, sampling=voxel_size)
                mask = distances <= radius
                
                # Apply mask to map data
                masked_data = map_data.copy()
                masked_data[mask] = 0
                
                # Save result
                with mrcfile.new(output_map_path, overwrite=True) as mrc_out:
                    mrc_out.set_data(masked_data.astype(np.float32))
                    mrc_out.voxel_size = mrc.voxel_size
                    mrc_out.header.origin = mrc.header.origin
                
                return output_map_path
                
        except Exception as e:
            error_msg = f"Error in map masking for {model_name}: {str(e)}"
            self.logger.error(error_msg)
            raise

    def process_docking(self, density_map, contour_level, resolution, fasta_file, 
                       dock_type="domainwise"):
        """
        Process docking for multiple atomic models into a single density map.
        
        Args:
            density_map: Path to input density map file
            contour_level: Contour level threshold value
            resolution: Resolution value for docking
            fasta_file: Path to FASTA sequence file
            dock_type: Type of docking - 'chainwise' or 'domainwise' (default: 'domainwise')
            
        Returns:
            dict: Summary of results including successful and failed models
        """
        self.print_clean("=" * 80)
        self.print_clean("🎯 DOCKING PROCESS STARTING")
        self.print_clean("=" * 80)
        
        self.logger.info("STARTING DOCKING PROCESS")
        
        # Log input parameters
        self.logger.info(f"Density map: {density_map}")
        self.logger.info(f"Contour level: {contour_level}")
        self.logger.info(f"Resolution: {resolution} Å")
        self.logger.info(f"FASTA file: {fasta_file}")
        self.logger.info(f"Dock type: {dock_type}")
        
        # Validate inputs
        if not os.path.exists(density_map):
            error_msg = f"Density map not found: {density_map}"
            self.logger.error(error_msg)
            self.print_clean(f"❌ Error: {error_msg}")
            raise FileNotFoundError(error_msg)
        if not os.path.exists(fasta_file):
            error_msg = f"FASTA file not found: {fasta_file}"
            self.logger.error(error_msg)
            self.print_clean(f"❌ Error: {error_msg}")
            raise FileNotFoundError(error_msg)
        
        # Validate dock_type
        if dock_type not in ['chainwise', 'domainwise']:
            error_msg = f"Invalid dock_type: {dock_type}. Must be 'chainwise' or 'domainwise'"
            self.logger.error(error_msg)
            self.print_clean(f"❌ Error: {error_msg}")
            raise ValueError(error_msg)
        
        if dock_type == 'domainwise':
            atomic_models_directory = os.path.join(os.path.dirname(self.AF3_results), 'AF3_domains')
        else:
            atomic_models_directory = os.path.join(os.path.dirname(self.AF3_results), 'AF3_chains')
        
        # Create directories
        os.makedirs(atomic_models_directory, exist_ok=True)
        docked_models_directory = os.path.join(os.path.dirname(self.AF3_results), 'AF3_docked_models')
        temp_maps_directory = os.path.join(os.path.dirname(self.AF3_results), 'temp_maps')
        os.makedirs(docked_models_directory, exist_ok=True)
        os.makedirs(temp_maps_directory, exist_ok=True)
        
        # Find atomic models
        atomic_models = sorted(glob(f"{atomic_models_directory}/*.pdb"))
        if not atomic_models:
            error_msg = f"No atomic models found inside {atomic_models_directory}. Please check the chains or domains PDB files"
            self.logger.error(error_msg)
            self.print_clean(f"❌ Error: {error_msg}")
            raise FileNotFoundError(error_msg)
        
        self.print_clean(f"📊 DOCKING PARAMETERS:")
        self.print_clean(f"   🗺️  Density map: {os.path.basename(density_map)}")
        self.print_clean(f"   📏 Contour level: {contour_level}")
        self.print_clean(f"   🔍 Resolution: {resolution} Å")
        self.print_clean(f"   🧬 FASTA file: {os.path.basename(fasta_file)}")
        self.print_clean(f"   ⚙️  Dock type: {dock_type}")
        self.print_clean(f"   📁 Models directory: {atomic_models_directory}")
        self.print_clean(f"   🎯 Found {len(atomic_models)} models to dock")
        self.print_clean("")
        
        # Initial map processing
        self.print_clean("🔄 Processing initial density map...")
        initial_map_path = self.initial_map_processing(
            density_map, 
            f"{temp_maps_directory}/initial_processed.mrc", 
            contour_level
        )
        self.print_clean("✅ Initial map processing complete")
        self.print_clean("")
        
        # Process each atomic model
        self.print_clean("=" * 80)
        self.print_clean("🚀 STARTING ITERATIVE DOCKING")
        self.print_clean("=" * 80)
        
        current_map_path = initial_map_path
        successful_models = []
        
        for i, atomic_model in enumerate(atomic_models, 1):
            if not os.path.exists(atomic_model):
                self.print_clean(f"⚠️  [{i}/{len(atomic_models)}] Skipping {os.path.basename(atomic_model)}: file not found")
                self.failed_models.append(atomic_model)
                continue
            
            self.print_clean(f"")
            self.print_clean(f"📍 [{i}/{len(atomic_models)}] Processing: {os.path.basename(atomic_model)}")
            
            # Setup output paths
            model_name = os.path.basename(atomic_model).replace('.pdb', '')
            docked_file = f"{docked_models_directory}/{model_name}_docked.pdb"
            next_map_path = f"{temp_maps_directory}/map_after_{model_name}.mrc"
            
            # Check if already docked
            if os.path.exists(docked_file):
                self.print_clean(f"   ↻ Using existing docked file")
                try:
                    current_map_path = self.subsequent_map_processing(
                        current_map_path, 
                        docked_file, 
                        next_map_path
                    )
                    successful_models.append(atomic_model)
                    self.print_clean(f"   ✅ Map updated for existing model")
                except Exception as e:
                    self.print_clean(f"   ❌ Error processing map for existing model: {str(e)}")
                    self.failed_models.append(atomic_model)
                continue
            
            # Perform docking
            self.print_clean(f"   🎯 Running Phenix dock_in_map...")
            docking_successful = self.run_phenix_dock_in_map(
                density_map=current_map_path,
                atomic_model=atomic_model,
                resolution=resolution,
                save_file=docked_file,
                fasta_file=fasta_file
            )
            
            if docking_successful and os.path.exists(docked_file):
                try:
                    self.print_clean(f"   🗺️  Updating density map...")
                    # Update map by masking docked structure
                    current_map_path = self.subsequent_map_processing(
                        current_map_path,
                        docked_file,
                        next_map_path
                    )
                    successful_models.append(atomic_model)
                    self.print_clean(f"   ✅ Model docked successfully and map updated")
                except Exception as e:
                    self.print_clean(f"   ❌ Error processing map after docking: {str(e)}")
                    if os.path.exists(docked_file):
                        os.remove(docked_file)
                    self.failed_models.append(atomic_model)
            else:
                self.print_clean(f"   ❌ Docking failed")
                if os.path.exists(docked_file):
                    os.remove(docked_file)
                self.failed_models.append(atomic_model)
        
        # Cleanup temporary directory
        self.print_clean("")
        self.print_clean("🧹 Cleaning up temporary files...")
        try:
            shutil.rmtree(temp_maps_directory)
            self.print_clean(f"✅ Cleaned up temporary directory")
        except Exception as e:
            self.print_clean(f"⚠️  Warning: Could not remove temporary directory: {str(e)}")
        
        # Print and log final summary
        self.print_clean("")
        self.print_clean("=" * 80)
        self.print_clean("🏁 DOCKING COMPLETE")
        self.print_clean("=" * 80)
        self.print_clean(f"📊 Total models processed: {len(atomic_models)}")
        self.print_clean(f"✅ Successfully docked: {len(successful_models)}")
        self.print_clean(f"❌ Failed: {len(self.failed_models)}")
        self.print_clean(f"📁 Results saved to: {docked_models_directory}")
        self.print_clean(f"📄 Log files saved to: {self.log_directory}")
        
        self.logger.info("FINAL SUMMARY:")
        self.logger.info(f"Total: {len(atomic_models)}, Success: {len(successful_models)}, Failed: {len(self.failed_models)}")
        
        if self.failed_models:
            self.print_clean("")
            self.print_clean("❌ Failed models:")
            for model in self.failed_models:
                self.print_clean(f"   • {os.path.basename(model)}")
            self.logger.warning(f"Failed models: {[os.path.basename(m) for m in self.failed_models]}")
        
        self.print_clean("=" * 80)
        self.print_clean("")
        
        return {
            "total": len(atomic_models),
            "successful": successful_models,
            "failed": self.failed_models,
            "docked_models_directory": docked_models_directory,
            "log_directory": self.log_directory,
            "main_log_file": self.main_log_file
        }


    def generate_unique_chain_id(self, base_id, used_ids):
        """
        Generate a unique chain ID that hasn't been used.
        
        Args:
            base_id: Base chain ID to start from
            used_ids: Set of already used chain IDs
            
        Returns:
            str: Unique chain ID
        """
        chain_id = base_id
        
        if chain_id not in used_ids:
            return chain_id
        
        # Try single letters A-Z, a-z
        for i in range(26):
            chain_id = chr(ord('A') + i)
            if chain_id not in used_ids:
                return chain_id
                
        for i in range(26):
            chain_id = chr(ord('a') + i)
            if chain_id not in used_ids:
                return chain_id
        
        # If all single characters are used, use two-character IDs
        for i in range(26):
            for j in range(26):
                chain_id = chr(ord('A') + i) + chr(ord('A') + j)
                if chain_id not in used_ids:
                    return chain_id
        
        # Fallback to numbered chains
        counter = 0
        while True:
            chain_id = f"C{counter}"
            if chain_id not in used_ids:
                return chain_id
            counter += 1

    def combine_af3_docked_results(self):
        """
        Process AF3 results directory by combining docked structures into a single PDB file.
        
        Returns:
            dict: Processing results
        """
        self.print_clean("")
        self.print_clean("=" * 80)
        self.print_clean("🔗 AF3 DOCKED STRUCTURES COMBINING")
        self.print_clean("=" * 80)
        success = False
        docked_results_dir = os.path.join(os.path.dirname(self.AF3_results), 'AF3_docked_models')
        
        self.logger.info(f"Processing docked AF3 structures from: {docked_results_dir}")
        
        if not os.path.exists(docked_results_dir):
            error_msg = f"AF3 docked structures directory not found: {docked_results_dir}"
            self.logger.error(error_msg)
            self.print_clean("")
            self.print_clean(f"💡 Expected docked structures directory: {docked_results_dir}")
            self.print_clean("")
            return {"success": False, "error": error_msg}
        
        pdb_files = sorted(glob(f"{docked_results_dir}/*chain*.pdb"))
        
        if len(pdb_files) == 0:
            self.logger.error(f"No PDB files found in {docked_results_dir}")
            self.print_clean("")
            self.print_clean("💡 Expected file pattern: *chain*.pdb")
            self.print_clean("")
            return {"success": False, "error": "No PDB files found"}
        
        self.logger.info(f"Found {len(pdb_files)} PDB files to combine")
        self.logger.info(f"Starting combination of {len(pdb_files)} PDB files")
        
        # Set output path
        self.combined_docked_model_path = os.path.join(os.path.dirname(self.AF3_results), f'{os.path.basename(os.path.dirname(self.AF3_results))}_af3_docked.pdb')
        
        # Create a new structure with a single model
        combined_structure = PDB.Structure.Structure("combined")
        combined_model = PDB.Model.Model(0)
        combined_structure.add(combined_model)
        
        # Keep track of used chain IDs and processing stats
        used_chain_ids = set()
        chains_added = 0
        files_processed = 0
        processing_errors = []
        
        # Process each input PDB file
        for pdb_file in pdb_files:
            try:
                if not os.path.exists(pdb_file):
                    error_msg = f"File not found: {pdb_file}"
                    processing_errors.append(error_msg)
                    self.logger.warning(error_msg)
                    continue
                
                structure_id = os.path.splitext(os.path.basename(pdb_file))[0]
                structure = self.parser.get_structure(structure_id, pdb_file)
                file_chains = 0
                
                for chain in structure[0]:
                    original_chain_id = chain.id
                    
                    unique_chain_id = self.generate_unique_chain_id(original_chain_id, used_chain_ids)
                    if unique_chain_id != original_chain_id:
                        chain.id = unique_chain_id
                        
                    combined_model.add(chain)
                    used_chain_ids.add(unique_chain_id)
                    chains_added += 1
                    file_chains += 1
                
                files_processed += 1
                self.logger.info(f"✓ Added {file_chains} chain from {os.path.basename(pdb_file)}")
                
            except Exception as e:
                error_msg = f"Error processing {os.path.basename(pdb_file)}: {str(e)}"
                processing_errors.append(error_msg)
                self.logger.error(f"✗ {error_msg}")
        
        # Save the combined structure if any chains were added
        if chains_added > 0:
            try:
                # Create output directory if it doesn't exist
                os.makedirs(os.path.dirname(self.combined_docked_model_path), exist_ok=True)
                
                self.io.set_structure(combined_structure)
                self.io.save(self.combined_docked_model_path)
                
                self.logger.info(f"✓ Combined PDB saved to: {os.path.basename(self.combined_docked_model_path)}")
                self.logger.info(f"✓ Total chains: {chains_added}, Files processed: {files_processed}")
                
                # Summary
                self.print_clean("")
                self.print_clean(f"🎯 Combination Results:")
                self.print_clean(f"   ✅ Successfully combined: {files_processed}/{len(pdb_files)} files")
                self.print_clean(f"   🔗 Total chains: {chains_added}")
                self.print_clean(f"   📁 Output file: {os.path.basename(self.combined_docked_model_path)}")
                self.print_clean("")
                success = True
            except Exception as e:
                error_msg = f"Error saving combined structure: {str(e)}"
                self.logger.error(f"✗ {error_msg}")
                
                self.print_clean("")
                self.print_clean(f"🎯 Combination Results:")
                self.print_clean(f"   ❌ Combination failed: {error_msg}")
                self.print_clean("")
                return success
                
        else:
            error_msg = "No chains were successfully added to the combined structure"
            self.logger.error(f"✗ {error_msg}")
            
            self.print_clean("")
            self.print_clean(f"🎯 Combination Results:")
            self.print_clean(f"   ❌ Combination failed: {error_msg}")
            if processing_errors:
                self.print_clean(f"   ⚠️  Processing errors: {len(processing_errors)}")
            self.print_clean("")
        return success


 
def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description='Phenix Docking Tool for Cryo-EM Map',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

python docking_script.py \\
    --density_map emd_12345.map \\
    --contour_level 0.3 \\
    --resolution 3.2 \\
    --fasta_file protein.fasta \\
    --AF3_results "../input/protein/AF3_results/" \\
    --dock_type domainwise \\
    --phenix_act "/path/to/phenix_env.sh"
        """
        )
    
    parser.add_argument('-m', '--density_map', required=True, help='Path to density map file (.map or .mrc)')
    parser.add_argument('-c', '--contour_level', type=float, required=True, help='Contour level value')
    parser.add_argument('-r', '--resolution', type=float, required=True, help='Resolution value in Angstroms')
    parser.add_argument('-f', '--fasta_file', required=True, help='Path to FASTA sequence file')
    parser.add_argument('-a', '--AF3_results', required=True, help='Path to directory containing AlphaFold3 results (e.g., "../input/protein_AF3_results")')
    parser.add_argument('--dock_type', choices=['chainwise', 'domainwise'], default='domainwise', help='Type of docking: chainwise or domainwise atomic models (default: domainwise)')
    parser.add_argument('--phenix_act', required=True, help='Path to Phenix environment activation script (e.g., "/path/to/phenix_env.sh")')
        
    args = parser.parse_args()
    
    # Validate inputs before starting
    print("")
    print("🧬 Phenix Docking Tool for Cryo-EM Map")
    print("=" * 80)
    print("🔍 VALIDATING INPUTS...")
    
    errors = []
    if not os.path.exists(args.density_map):
        errors.append(f"Density map not found: {args.density_map}")
    if not os.path.exists(args.fasta_file):
        errors.append(f"FASTA file not found: {args.fasta_file}")
    if not os.path.exists(args.AF3_results):
        errors.append(f"AF3 results directory not found: {args.AF3_results}")
    if not os.path.exists(args.phenix_act):
        errors.append(f"Phenix activation script not found: {args.phenix_act}")
    
    if errors:
        print("❌ INPUT VALIDATION FAILED:")
        for error in errors:
            print(f"   • {error}")
        print("")
        sys.exit(1)
    
    print("✅ Input validation passed")
    print("")
    
    try:
        log_dir = os.path.join(os.path.dirname(args.AF3_results), 'docking_logs')

        processor = PhenixDockingProcessor(
            phenix_command=args.phenix_act,
            AF3_results=args.AF3_results,
            log_directory=log_dir
        )
        
        results = processor.process_docking(
            density_map=args.density_map,
            contour_level=args.contour_level,
            resolution=args.resolution,
            fasta_file=args.fasta_file,
            dock_type=args.dock_type
        )
        
        if len(results['successful']) >=1 :
            processor.combine_af3_docked_results()
        
        # Exit with appropriate code
        if results['failed']:
            sys.exit(1)  # Some failures occurred
        else:
            sys.exit(0)  # All successful
            
    except KeyboardInterrupt:
        print("")
        print("❌ Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()