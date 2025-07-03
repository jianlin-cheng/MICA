import argparse
import warnings
import torch
import numpy as np
import os
import multiprocessing
from utils.modeler import Solver

warnings.filterwarnings('ignore')
np.set_printoptions(threshold=np.inf, suppress=True, precision=2)

def run_MICA(modeling_config):
    # Validate protocol
    if modeling_config.protocol not in ['AF3_struct', 'AF3_struct_free']:
        return 'Wrong protocol! protocol should be [AF3_struct,AF3_struct_free]'
    
    # Validate fasta path
    if not modeling_config.fasta_path:
        return '--fasta_path is required'
    if not os.path.exists(modeling_config.fasta_path):
        return '--fasta_path: path not exists!'
    
    # Validate AF3 structure path for AF3_struct protocol
    if modeling_config.protocol == 'AF3_struct':
        if not modeling_config.AF3_structure_path:
            return '--AF3_structure_path is required when protocol is AF3_struct'
        if not os.path.exists(modeling_config.AF3_structure_path):
            return '--AF3_structure_path: path not exists!'
    
    # Validate phenix requirements
    if modeling_config.run_phenix:
        modeling_config.run_pulchra = True
        if not modeling_config.resolution:
            return '--resolution is required for run.phenix_real_space_refine'
        if not modeling_config.phenix_act:
            return '--phenix_act is required for run.phenix_real_space_refine'
    
    # Validate pulchra requirements
    if modeling_config.run_pulchra:
        if not modeling_config.pulchra_path:
            return '--pulchra_path is required for run.phenix_real_space_refine'

    MICA_modeler = Solver(modeling_config)
    return MICA_modeler.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Core arguments
    parser.add_argument('--protocol', type=str, default='AF3_struct', 
                       help='Choose among AF3_struct and AF3_struct_free')
    parser.add_argument('-m', '--map_path', type=str, required=True, 
                       help='Path of cryo-EM map.')
    parser.add_argument('-f', '--fasta_path', type=str, required=True, 
                       help='Path of fasta file.')
    parser.add_argument('-i', '--input_path', type=str, required=True, 
                       help='Path containing data for your protein. Example: input/39164 or input/protein')
    
    # Optional processing
    parser.add_argument('--run_pulchra', action='store_true', 
                       help='Whether to run pulchra for all_atom construction')
    parser.add_argument('--pulchra_path', type=str, 
                       help='Directory of pulchra, e.g.: modules/pulchra304/src/pulchra')
    parser.add_argument('--run_phenix', action='store_true', 
                       help='Whether to run phenix.real_space_refine')
    parser.add_argument('-r', '--resolution', type=float, 
                       help='Resolution of cryo-EM map, required when run_phenix_real_space_refine is open')
    parser.add_argument('--phenix_act', type=str, 
                       help='Script to activate phenix environment, e.g.: modules/phenix-1.20.1-4487/phenix_env.sh')
    parser.add_argument('--phenix_param', default='modules/phenix.eff', type=str, 
                       help='Param for phenix.real_space_refine')
    
    # Model and output
    parser.add_argument('--model_path', default='trained_models/MICA_best_model.pth', 
                       help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('-o', '--output_path', default='output', 
                       help='Output directory for saving results')
    parser.add_argument('--device', default='cuda', 
                       help='Device to run inference on (default: cuda), Options: cpu, cuda, cuda:1, cuda:2, ...')
    parser.add_argument('--no_parallel', action='store_true', 
                       help='Disable parallel processing')
    parser.add_argument('--quiet', action='store_true', 
                       help='Suppress detailed output')
    
    # Algorithm parameters
    parser.add_argument('--seed', type=int, default=2022, help='Random seed')
    parser.add_argument('--cluster_eps', type=int, default=10, help='Clustering epsilon')
    parser.add_argument('--cluster_min_points', type=int, default=10, help='Clustering min points')
    parser.add_argument('--nms_radius', type=int, default=9, help='NMS radius')
    parser.add_argument('--CA_score_thrh', type=float, default=0.3, help='CA score threshold')
    parser.add_argument('--frags_len', type=int, default=150, help='Fragment length')
    parser.add_argument('--n_hop', type=int, default=6, help='Number of hops')
    parser.add_argument('--neigh_mat_thrh', type=float, default=0.7, help='Neighbor matrix threshold')
    parser.add_argument('--mul_proc_num', type=int, default=1, help='Number of processes (1 = auto-detect)')
    parser.add_argument('--score_thrh', type=float, default=2, help='Score threshold')
    parser.add_argument('--gap_len', type=int, default=3, help='Gap length')
    parser.add_argument('--struct_len', type=int, default=5, help='Structure length')
    
    # Parse arguments
    modeling_config = parser.parse_args()
    
    # Auto-detect process count if set to 1
    if modeling_config.mul_proc_num == 1:
        cpu_cores = multiprocessing.cpu_count()
        modeling_config.mul_proc_num = max(int(cpu_cores * 0.75), 1)
    
    # Set up paths
    modeling_config.AF3_results = os.path.join(modeling_config.input_path, 'AF3_results')
    modeling_config.AF3_structure_path = os.path.join(modeling_config.input_path, 'AF3_structures')
    modeling_config.grids_path = os.path.join(modeling_config.input_path, 'grids')
    modeling_config.normalized_map_path = os.path.join(modeling_config.input_path, 'resampled_normalized_map.mrc')
    modeling_config.AF3_encodings_path = os.path.join(modeling_config.input_path, 'AF3_encodings')
    
    # Set random seed
    torch.manual_seed(modeling_config.seed)
    
    # Run MICA
    result = run_MICA(modeling_config)
    if result != 'success':
        raise Exception(result)