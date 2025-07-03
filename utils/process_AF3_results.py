#!/usr/bin/env python3
"""
AlphaFold3 Results Processing

This module converts AF3 CIF files to PDB format and split into individual chains using FASTA sequence information
and generates structures for postprocessing. It also splits the chains into individual domains.

Author: Rajan Gyawali
Date: June 1, 2025
"""

import argparse
import os
import re
import logging
from glob import glob
from Bio import PDB
import subprocess
import sys
from datetime import datetime
from pathlib import Path


class OutOfChainsError(Exception):
    """Exception raised when more than 62 chains are present in the structure."""
    pass


class AF3ResultsProcessor:
    """Convert CIF files to PDB format with FASTA sequence processing."""
    
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
        """Print message without timestamp or log level for visual appeal."""
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

    def process_fasta_files(self, fasta_content):
        """Process FASTA files and extract protein information."""
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

    def int_to_chain(self, i, base=62):
        """
        int_to_chain(int,int) -> str
        Converts a positive integer to a chain ID. Chain IDs include uppercase
        characters, numbers, and optionally lowercase letters.
        i = a positive integer to convert
        base = the alphabet size to include. Typically 36 or 62.
        """
        if i < 0:
            raise ValueError("positive integers only")
        if base < 0 or 62 < base:
            raise ValueError("Invalid base")

        quot = int(i) // base
        rem = i % base
        if rem < 26:
            letter = chr(ord("A") + rem)
        elif rem < 36:
            letter = str(rem - 26)
        else:
            letter = chr(ord("a") + rem - 36)
        if quot == 0:
            return letter
        else:
            return self.int_to_chain(quot - 1, base) + letter

    def rename_chains(self, structure):
        """Renames chains to be one-letter chains
        
        Existing one-letter chains will be kept. Multi-letter chains will be truncated
        or renamed to the next available letter of the alphabet.
        
        If more than 62 chains are present in the structure, raises an OutOfChainsError
        
        Returns a map between new and old chain IDs, as well as modifying the input structure
        """
        next_chain = 0
        # single-letters stay the same
        chainmap = {c.id: c.id for c in structure.get_chains() if len(c.id) == 1}
        for o in structure.get_chains():
            if len(o.id) != 1:
                if o.id[0] not in chainmap:
                    chainmap[o.id[0]] = o.id
                    o.id = o.id[0]
                else:
                    c = self.int_to_chain(next_chain)
                    while c in chainmap:
                        next_chain += 1
                        c = self.int_to_chain(next_chain)
                        if next_chain >= 62:
                            raise OutOfChainsError()
                    chainmap[c] = o.id
                    o.id = c
        return chainmap
    
    def get_chain_names(self, chain_file):
        """Extract chain names from PDB file."""
        chain_names = set()
        
        try:
            with open(chain_file, 'r') as file:
                for line in file:
                    if line.startswith(('ATOM', 'HETATM')):
                        chain_id = line[21].strip()
                        if chain_id:
                            chain_names.add(chain_id)
        
        except FileNotFoundError:
            self.logger.error(f"File not found: {chain_file}")
            return []
        except Exception as e:
            self.logger.error(f"Error reading PDB file: {e}")
            return []
        
        return sorted(list(chain_names))
    
    def run_merizo_command(self, command, capture_output=True, timeout=None):
        """Run a merizo command using subprocess."""
        try:
            result = subprocess.run(command, shell=True, 
                                capture_output=capture_output, text=True, timeout=timeout)
            return {"success": result.returncode == 0, "stdout": result.stdout, 
                    "stderr": result.stderr, "returncode": result.returncode}
        except subprocess.TimeoutExpired:
            return {"success": False, "stdout": None, "stderr": "Timeout", "returncode": -1}
        except Exception as e:
            return {"success": False, "stdout": None, "stderr": str(e), "returncode": -1}

    def generate_chainwise_structures_for_postprocessing(self, structure, protein_id, postprocessing_structures_location="AF3_structures"):
        """Generate chainwise structures for postprocessing."""
        model = structure[0]
        
        # Get the first chain ID
        first_chain_id = list(model.child_dict.keys())[0]
        
        new_structure = PDB.Structure.Structure("first_chain")
        new_model = PDB.Model.Model(0)
        new_structure.add(new_model)
        
        # Add the first chain to the new structure
        new_model.add(model[first_chain_id])
        
        # Save the new structure
        io = PDB.PDBIO()
        io.set_structure(new_structure)
        input_location = f"{postprocessing_structures_location}/{protein_id}"
        os.makedirs(input_location, exist_ok=True)
        io.save(f"{input_location}/ranked_0.pdb")

    def get_protein_by_id(self, protein_list, target_id):
        """Get protein information by ID."""
        matches = [protein for protein in protein_list if protein['protein_id'] == target_id]
        return matches[0] if matches else None

    def convert_single_cif_to_pdb(self, cif_file, fasta_content, AF3_results):
        """
        Convert a single CIF file to PDB format.
        
        Args:
            cif_file (str): Path to the CIF file
            fasta_content (str): FASTA sequence content
            AF3_results (str): Input directory for AF3 results
        
        Returns:
            bool: True if conversion successful, False otherwise
        """
        try:
            # Process the FASTA content
            protein_info_dicts = self.process_fasta_files(fasta_content)
            if not protein_info_dicts:
                self.logger.error("No protein information found")
                return False
            
            mmcif_parser = PDB.MMCIFParser()
            pdb_io = PDB.PDBIO()
            c_file = os.path.basename(cif_file).split('.')[0]
            
            # Extract chain index and protein info
            filename_parts = os.path.basename(cif_file).split("_")
            if len(filename_parts) >= 3:
                chain_index = filename_parts[2]
                pdb_name = protein_info_dicts[0]['protein_id'].split("_")[0]
                protein_id = f"{pdb_name}_{chain_index}"
            else:
                protein_id = protein_info_dicts[0]['protein_id']
            
            protein_dict = self.get_protein_by_id(protein_info_dicts, protein_id)
            
            structure = mmcif_parser.get_structure("protein_structure", cif_file)
            
            try:
                chainmap = self.rename_chains(structure)
                pdb_io.set_structure(structure)
                
                # Create input directory to store chains
                input_location = AF3_results.replace("results", "PDBs")
                os.makedirs(input_location, exist_ok=True)
                
                # Create directory to store structures to be utilized for post processing
                structures_location = AF3_results.replace("results", "structures")
                os.makedirs(structures_location, exist_ok=True)
                
                input_pdb_file = f"{input_location}/{c_file}.pdb"
                pdb_io.save(input_pdb_file)
                self.logger.info(f"✓ CIF converted: {c_file}.pdb")
                
                self.generate_chainwise_structures_for_postprocessing(
                    structure, protein_id, structures_location
                )
                return True
                
            except OutOfChainsError as e:
                self.logger.error(f"✗ Too many chains to represent in PDB format: {c_file}")
                self.failed_entries.append(cif_file)
                return False
                
        except Exception as e:
            self.logger.error(f"✗ CIF conversion failed: {c_file} - {e}")
            self.failed_entries.append(cif_file)
            return False

    def convert_cif_to_pdb(self, fasta_file, AF3_results):
        """
        Convert CIF files to PDB format with matching FASTA file.
        
        Args:        
            fasta_file (str): Location of FASTA file  
            AF3_results (str): Directory containing CIF files
        
        Returns:
            dict: Results summary
        """
        self.print_clean("")
        self.print_clean("=" * 80)
        self.print_clean("🔄 CIF TO PDB CONVERSION")
        self.print_clean("=" * 80)
        
        cif_files = sorted(glob(f"{AF3_results}/*/*_model_0.cif"))
        if len(cif_files) == 0:
            self.logger.error(f"No AlphaFold3 structures found in {AF3_results}")
            self.print_clean("")
            self.print_clean("💡 Expected structure format: {AF3_results}/*/{protein_id}_model_0.cif")
            self.print_clean("")
            return {"successful": [], "failed": [], "total": 0}
        
        self.logger.info(f"Found {len(cif_files)} CIF files to convert")
        successful = []
        failed = []
        
        for cif_file in cif_files:
            try:                
                if not fasta_file:
                    self.logger.error("No FASTA file provided")
                    failed.append(cif_file)
                    continue
                               
                # Read FASTA content
                with open(fasta_file, 'r') as f:
                    fasta_content = f.read()
                
                success = self.convert_single_cif_to_pdb(
                    cif_file=cif_file,
                    fasta_content=fasta_content,
                    AF3_results=AF3_results
                )
                
                if success:
                    successful.append(cif_file)
                else:
                    failed.append(cif_file)
                    
            except Exception as e:
                self.logger.error(f"Error processing {os.path.basename(cif_file)}: {e}")
                failed.append(cif_file)
        
        # Summary
        self.print_clean("")
        self.print_clean(f"🎯 CIF Conversion Results:")
        self.print_clean(f"   ✅ Successful: {len(successful)}/{len(cif_files)}")
        if failed:
            self.print_clean(f"   ❌ Failed: {len(failed)}")
        self.print_clean("")
        
        return {
            "successful": successful,
            "failed": failed,
            "total": len(cif_files)
        }
        
    def split_individual_chains(self, AF3_results):
        """
        Split PDB files to individual chains.
        
        Args:        
            AF3_results (str): Directory containing AF3 results
        """
        self.print_clean("=" * 80)
        self.print_clean("⛓️  SPLITTING CHAINS")
        self.print_clean("=" * 80)
        
        pdb_location = os.path.join(os.path.dirname(AF3_results), 'AF3_PDBs')
        chain_location = os.path.join(os.path.dirname(AF3_results), 'AF3_chains')
        os.makedirs(chain_location, exist_ok=True)
        
        pdb_files = glob(f"{pdb_location}/*.pdb")
        if not pdb_files:
            self.logger.error(f"No PDB files found in {pdb_location}")
            return
        
        self.logger.info(f"Processing {len(pdb_files)} PDB files for chain splitting")
        chain_count = 0
        
        for pdb_file in pdb_files:
            try:
                parser = PDB.PDBParser(QUIET=True)
                structure = parser.get_structure('protein', pdb_file)
                io = PDB.PDBIO()
                
                file_chains = 0
                for model in structure:
                    for chain in model:
                        # Create new structure for this chain
                        new_structure = PDB.Structure.Structure('chain')
                        new_model = PDB.Model.Model(0)
                        new_structure.add(new_model)
                        new_model.add(chain)
                        
                        # Save chain to file
                        io.set_structure(new_structure)
                        chain_file_name = pdb_file[:-4].replace("PDBs", "chains")
                        input_file = f"{chain_file_name}_chain_{chain.id}.pdb"
                        io.save(input_file)
                        file_chains += 1
                        chain_count += 1
                
                filename = os.path.basename(pdb_file)
                self.logger.info(f"✓ Split {filename}: {file_chains} chains")
                
            except Exception as e:
                self.logger.error(f"✗ Failed to split {os.path.basename(pdb_file)}: {e}")
        
        self.print_clean("")
        self.print_clean(f"🎯 Chain Splitting Results:")
        self.print_clean(f"   🔗 Total chains extracted: {chain_count}")
        self.print_clean(f"   📁 Saved to: {chain_location}")
        self.print_clean("")
        
    def split_individual_domains(self, AF3_results):
        """
        Split PDB files to individual domains using Merizo.
        
        Args:        
            AF3_results (str): Directory containing AF3 results
        """
        self.print_clean("=" * 80)
        self.print_clean("🧩 SPLITTING DOMAINS WITH MERIZO")
        self.print_clean("=" * 80)
        self.print_clean("⏳ This process may take some time - please be patient...")
        self.print_clean("")
        
        chain_location = os.path.join(os.path.dirname(AF3_results), 'AF3_chains')
        domain_location = os.path.join(os.path.dirname(AF3_results), 'AF3_domains')
        os.makedirs(domain_location, exist_ok=True)
        
        chain_files = glob(f"{chain_location}/*chain*.pdb")
        if not chain_files:
            self.logger.error(f"No chain files found in {chain_location}")
            return
        
        self.logger.info(f"Processing {len(chain_files)} chain files for domain splitting")
        processed_count = 0
        
        for chain_file in chain_files:
            try:
                chain_names = self.get_chain_names(chain_file)
                if not chain_names:
                    self.logger.warning(f"No chains found in {os.path.basename(chain_file)}")
                    continue
                
                chain = chain_names[0]
                merizo_command = f"python modules/merizo/predict.py -i {chain_file} --pdb_chain {chain} --save_pdb --save_domains -o {domain_location}"
                
                merizo_result = self.run_merizo_command(merizo_command)
                
                if merizo_result['success']:
                    filename = os.path.basename(chain_file)
                    self.logger.info(f"✓ Domains split: {filename}")
                    processed_count += 1
                else:
                    self.logger.error(f"✗ Merizo failed: {os.path.basename(chain_file)}")
                    if merizo_result['stderr']:
                        self.logger.error(f"Error: {merizo_result['stderr']}")
                        
            except Exception as e:
                self.logger.error(f"✗ Domain splitting failed: {os.path.basename(chain_file)} - {e}")
        
        self.print_clean("")
        self.print_clean(f"🎯 Domain Splitting Results:")
        self.print_clean(f"   🧩 Chains processed: {processed_count}/{len(chain_files)}")
        self.print_clean(f"   📁 Saved to: {domain_location}")
        self.print_clean("")
        
def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description='Convert AlphaFold3 CIF files to PDB format with chain and domain splitting')
    parser.add_argument('-f', '--fasta_file', required=True, help='Path to FASTA file')
    parser.add_argument('-a', '--AF3_results', required=True, help='Path to directory containing AlphaFold3 Results (Example: input/protein/AF3_results)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.fasta_file):
        print(f"❌ Error: FASTA file not found: {args.fasta_file}")
        sys.exit(1)
    
    if not os.path.exists(args.AF3_results):
        print(f"❌ Error: AF3 results directory not found: {args.AF3_results}")
        sys.exit(1)
    
    # Create converter instance
    converter = AF3ResultsProcessor()
    
    # Log startup information
    converter.print_clean("")
    converter.logger.info(f"FASTA file: {args.fasta_file}")
    converter.logger.info(f"AF3 results: {args.AF3_results}")
    
    # Process the files
    try:
        # Step 1: Convert CIF to PDB
        results = converter.convert_cif_to_pdb(
            fasta_file=args.fasta_file,
            AF3_results=args.AF3_results,
        )
        
        # Step 2: Split individual chains
        converter.split_individual_chains(AF3_results=args.AF3_results)
        
        # Step 3: Split individual domains
        converter.split_individual_domains(AF3_results=args.AF3_results)
        
        # Final summary
        converter.print_clean("=" * 80)
        converter.print_clean("🏁 AF3 RESULTS PROCESSING COMPLETE")
        converter.print_clean("=" * 80)
        converter.print_clean("")
        
    except KeyboardInterrupt:
        converter.print_clean("")
        converter.print_clean("❌ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        converter.logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()