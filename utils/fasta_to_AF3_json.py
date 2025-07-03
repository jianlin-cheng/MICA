#!/usr/bin/env python3
"""
FASTA to AlphaFold3 JSON Converter

This module converts FASTA sequences to AlphaFold3-compatible JSON format.
Can be used as a standalone script or imported as a module.

Author: Rajan Gyawali
Date: June 1, 2025
"""

import os
import re
import json
import argparse
import logging
from pathlib import Path
import shutil
from typing import List, Dict, Optional, Union
from datetime import datetime


class FastaToJsonConverter:
    """Convert FASTA sequences to AlphaFold3 JSON format with multichain support."""
    
    def __init__(self):
        self.failed_entries = []
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
        """Print message without timestamp or log level."""
        print(message)

    def extract_protein_info(self, fasta_content):
        """Extract protein information from FASTA content."""
        self.logger.info("Extracting protein information from FASTA content")
        
        lines = fasta_content.strip().split('\n')
        protein_info = []
        current_protein = None
        sequence = ""
        
        for line_num, line in enumerate(lines, 1):
            if line.startswith('>'):
                # Save previous protein if exists
                if current_protein and sequence:
                    current_protein['sequence'] = sequence
                    protein_info.append(current_protein)
                    sequence = ""
                
                # Parse new header
                header_match = re.match(r'>([^|]+)\|', line)
                if header_match:
                    protein_id = header_match.group(1)
                    
                    # Initialize the new protein entry
                    current_protein = {
                        'id': protein_id,
                        'header': line
                    }
                else:
                    self.logger.warning(f"Could not parse protein ID from header at line {line_num}")
                    
            elif current_protein is not None:
                # Add to current sequence
                sequence += line.strip()
        
        # Add the last protein
        if current_protein and sequence:
            current_protein['sequence'] = sequence
            protein_info.append(current_protein)
        
        self.logger.info(f"Found {len(protein_info)} proteins in FASTA file")
        return protein_info

    def process_files(self, fasta_content, base_name):
        """Process FASTA content and extract protein information."""
        self.logger.info(f"Processing FASTA content for: {base_name}")
        
        # Extract protein info from FASTA
        protein_info = self.extract_protein_info(fasta_content)
        
        if not protein_info:
            self.logger.error("No protein information found in FASTA content")
            return []
        
        protein_info_dicts = []
        
        # Process each protein
        for protein in protein_info:
            protein_info_dict = {}
            protein_id = protein['id']
            protein_info_dict['protein_id'] = protein['id']
            header = protein['header']
            
            try:
                chains = header.split("|")[1]
                
                # Remove 'Chains' prefix if present
                chains = chains.replace('Chains ', '').strip()
                chains = chains.replace('Chain ', '').strip()
                
                if '[' not in chains:
                    chain_matches = chains.split(', ')
                else:
                    chains = chains.split(', ')
                    chain_matches = [c.split("[")[0] for c in chains]
                
                protein_info_dict['chains'] = chain_matches
                protein_info_dict['len_chains'] = len(chain_matches)
                protein_info_dict['sequence'] = protein['sequence']
                protein_info_dicts.append(protein_info_dict)
                
                self.logger.info(f"Protein {protein_id}: {len(chain_matches)} chains, {len(protein['sequence'])} residues")
                
            except Exception as e:
                self.logger.error(f"Failed to process protein {protein_id}: {e}")
                continue
        
        return protein_info_dicts

    def generate_json_multichain(self, protein_id, chains, sequence):
        """
        Generate standard AF3 JSON for multiple chains.
        """
        sequence_entries = []
        
        for chain_id in chains:        
            # Clean sequence
            cleaned_sequence = sequence.replace('X', '').replace('x', '')
            
            # Skip if sequence is all X's
            if set(sequence) == {'X'}:
                self.logger.warning(f"Skipping chain {chain_id}: sequence contains only X residues")
                continue
                
            # Skip DNA sequences
            if set(cleaned_sequence).issubset({'A', 'G', 'C', 'T'}):
                self.logger.warning(f"Skipping chain {chain_id}: appears to be DNA sequence")
                continue
                
            # Skip RNA sequences
            if set(cleaned_sequence).issubset({'A', 'G', 'C', 'U'}):
                self.logger.warning(f"Skipping chain {chain_id}: appears to be RNA sequence")
                continue
            
            # Skip very short sequences
            if len(cleaned_sequence) < 2:
                self.logger.warning(f"Skipping chain {chain_id}: sequence too short ({len(cleaned_sequence)} residues)")
                continue
                
            sequence_entry = {
                "proteinChain": {
                    "id": chain_id,
                    "sequence": cleaned_sequence,
                    "count": 1
                }
            }
            sequence_entries.append(sequence_entry)
        
        if len(sequence_entries) == 0:
            self.logger.error(f"No valid sequence entries generated for protein {protein_id}")
            return None
        
        data = {
            "name": protein_id,
            "modelSeeds": [],
            "sequences": sequence_entries,
            "dialect": "alphafold3",
            "version": 1
        }
        
        return [data]

    def process_single_fasta(self, fasta_content, input_dir, base_name="protein"):
        """
        Process a single FASTA content and generate JSON files.
        
        Args:
            fasta_content (str): FASTA file content
            input_dir (str): Directory to save JSON files
            base_name (str): Base name for the input files (default: "protein")
        
        Returns:
            list: List of generated JSON file paths
        """
        self.print_clean("")
        self.print_clean("=" * 80)
        self.print_clean("🧬 FASTA TO ALPHAFOLD3 JSON CONVERTER")
        self.print_clean("=" * 80)
        
        try:
            # Create subdirectory based on base_name
            final_input_dir = os.path.join(input_dir, base_name, "AF3_JSON")
            os.makedirs(final_input_dir, exist_ok=True)
            self.logger.info(f"Output directory: {final_input_dir}")
            
            # Process the files
            protein_info_dicts = self.process_files(fasta_content, base_name)
            
            if not protein_info_dicts:
                self.logger.error(f"No valid proteins found for {base_name}")
                return []
            
            generated_files = []
            
            for protein_info_dict in protein_info_dicts:
                protein_id = protein_info_dict['protein_id']
                
                json_data_multichain = self.generate_json_multichain(
                    protein_id, 
                    protein_info_dict['chains'], 
                    protein_info_dict['sequence']
                )
                
                input_json_path_multichain = f"{final_input_dir}/{protein_id}.json"
                
                if json_data_multichain is not None:                
                    with open(input_json_path_multichain, "w") as outfile:
                        json.dump(json_data_multichain, outfile, indent=2)
                    
                    self.logger.info(f"✓ JSON saved: {protein_id}.json")
                    generated_files.append(input_json_path_multichain)
                else:
                    self.logger.error(f"✗ Failed to generate JSON for: {protein_id}")
            
            # Create results directory
            AF3_input_dir = os.path.join(input_dir, base_name, "AF3_results")
            
            os.makedirs(AF3_input_dir, exist_ok=True)
            
            # Summary and instructions
            self.print_clean("")
            self.print_clean("=" * 80)
            self.print_clean(f"🏁 CONVERSION COMPLETED - {len(generated_files)} files generated")
            self.print_clean("=" * 80)
            self.print_clean("")
            self.print_clean("📋 NEXT STEPS:")
            self.print_clean(f"   1. 📥 Download the generated JSON files from: {final_input_dir}")
            self.print_clean("   2. 🚀 Upload them to AlphaFold3 Server for prediction")
            self.print_clean("   3. 📦 Download the predicted results")
            self.print_clean(f"   4. 📁 Extract results to: {AF3_input_dir}")
            self.print_clean(f"   5. 🗂️  Results should be in format like:")
            for file in generated_files:
                self.print_clean(f"\t\t{AF3_input_dir}/{file.split('/')[-1].split('.')[0].lower()}/*model_0.cif")
            self.print_clean("")
            self.print_clean("")
            
            return generated_files
            
        except Exception as e:
            self.logger.error(f"Failed processing {base_name}: {e}")
            self.failed_entries.append(base_name)
            return []

    def process_fasta_file(self, fasta_file, input_dir="../input", base_name="protein"):
        """
        Process a FASTA file and generate JSON files.
        
        Args:
            fasta_file (str): Path to the FASTA file
            input_dir (str): Directory to save JSON files
            base_name (str): Base name for input files (default: "protein")
        
        Returns:
            list: List of generated JSON file paths
        """
        try:
            file_path = Path(fasta_file)
            if not file_path.exists():
                self.logger.error(f"FASTA file not found: {fasta_file}")
                return []
            
            self.logger.info(f"Loading FASTA file: {fasta_file}")
            
            with open(fasta_file, 'r') as f:
                fasta_content = f.read()
            
            return self.process_single_fasta(fasta_content, input_dir, base_name)
            
        except FileNotFoundError:
            self.logger.error(f"FASTA file not found: {fasta_file}")
            return []
        except Exception as e:
            self.logger.error(f"Error reading FASTA file {fasta_file}: {e}")
            return []
        
def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description='Convert FASTA sequences to AlphaFold3 JSON format')
    parser.add_argument('-f', '--fasta_file', required=True, help='Path to FASTA file')
    parser.add_argument('-i', '--input_dir', default='input', help='Base input directory')
    parser.add_argument('-n', '--name', default='protein', help='Name of the protein or cryo-EM map ID. Every inputs will be stored like input/protein/...')
    
    args = parser.parse_args()
    print()
    # Create converter instance
    converter = FastaToJsonConverter()
    
    # Process the file
    created_files = converter.process_fasta_file(
        fasta_file=args.fasta_file,
        input_dir=args.input_dir,
        base_name=args.name
    )
    
    return created_files


if __name__ == "__main__":
    main()