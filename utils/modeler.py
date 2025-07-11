"""

This module is the core which performs different functions.

Some part of the code is adapted from EModelX(+AF)

Date: June 1, 2025
"""
import os
import sys
import mrcfile
import numpy as np
import psutil
from utils.create_grids import GridCreator
from utils.predict import CryoEMPredictor
from utils.preprocessing import DataPreprocessor
import torch
import open3d as o3d
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
import networkx as nx
from multiprocessing import Pool
import time
import copy
import superpose3d
import subprocess
import shlex
import shutil
import random
from subprocess import run, DEVNULL
import logging


np.set_printoptions(threshold=np.inf,suppress=True,precision=2)

chainID_list = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','0','1','2','3','4','5','6','7','8','9']
AA_types = {"ALA":1,"CYS":2,"ASP":3,"GLU":4,"PHE":5,"GLY":6,"HIS":7,"ILE":8,"LYS":9,"LEU":10,"MET":11,"ASN":12,"PRO":13,"GLN":14,"ARG":15,"SER":16,"THR":17,"VAL":18,"TRP":19,"TYR":20}
AA_T = {AA_types[k]-1: k for k in AA_types}
AA_abb_T = {0:"A",1:"C",2:"D",3:"E",4:"F",5:"G",6:"H",7:"I",8:"K",9:"L",10:"M",11:"N",12:"P",13:"Q",14:"R",15:"S",16:"T",17:"V",18:"W",19:"Y"}
AA_abb = {AA_abb_T[k]:k for k in AA_abb_T}
abb2AA = {"A":"ALA","C":'CYS',"D":'ASP',"E":'GLU',"F":'PHE',"G":'GLY',"H":"HIS","I":"ILE","K":"LYS","L":"LEU","M":"MET","N":"ASN","P":"PRO","Q":"GLN","R":"ARG","S":"SER","T":"THR","V":"VAL","W":"TRP","Y":"TYR"}
AA2abb = {abb2AA[k]:k for k in abb2AA}

class NNPred:
    BBProb=None
    CAProb=None
    CAProb_clusted=None
    AAPred=None
    AAProb=None

class Sequence:
    """Sequence data structure with AF3 structure support."""
    def __init__(self,fasta_name,sequence):
        self.fasta_name = fasta_name
        self.sequence = sequence
        self.chain_dict = {}
        self.seq_matched_traces = []
        self.trace_matched_seqs = []
        self.trace_scores = []
        self.connect_ign=[]
        self.AF3_raw_structs=None
        self.AF3_struct=None
        self.chain_cand_mat=None

class Chain:
    """Chain data structure for sequence modeling."""
    def __init__(self, chain_id, sequence):
        self.sequence=sequence
        self.chain_id = chain_id
        self.trace_list=[]
        self.result = [-1 for _ in sequence]
        self.highConfResult = [-1 for _ in sequence]
        self.seq_matched_traces=[]
        self.trace_matched_seqs=[]

class GlobalParam:
    """Global parameter storage for multiprocessing."""
    def __init__(self):
        self.seq_cand_AA_mat=None
        self.neigh_mat=None
        self.fastas=None
        self.CA_cands=None
        self.neighbors2to6=None
        self.chain_cand_mat=None
        self.matched_chains=None
        self.trace_list=None
        self.CAProb=None


    def update(self,AA_score,pair_scores,fastas=None,CA_cands=None,neighbors2to6=None,chain_cand_mat=None,matched_chains=None,trace_list=None,CAProb=None):
        """Update global parameters for worker processes."""
        self.seq_cand_AA_mat=AA_score
        self.neigh_mat=pair_scores
        self.fastas=fastas
        self.CA_cands=CA_cands
        self.neighbors2to6=neighbors2to6
        self.chain_cand_mat=chain_cand_mat
        self.matched_chains=matched_chains
        self.trace_list=trace_list
        self.CAProb=CAProb

globalParam = GlobalParam()

def pathWalking(cand,n_hop):
    """
    Perform path walking for n-hop neighbor analysis.
    
    Args:
        cand: Starting candidate index
        n_hop: Number of hops to perform
        
    Returns:
        List of results for each hop level
    """
    global globalParam
    traces=[[cand]]
    scores=[1]
    results=[]
    for n in range(n_hop):
        tmp_traces=[]
        tmp_scores=[]
        for i, trace in enumerate(traces):
            cand = trace[-1]
            neigh_list = list(set(globalParam.neighbors2to6[cand])-set(trace))
            for neigh in neigh_list:
                tmp_traces.append(trace+[neigh])
                tmp_scores.append(scores[i]*max(globalParam.neigh_mat[cand, neigh],0.1))
        
        if tmp_traces:
            result = np.zeros([globalParam.neigh_mat.shape[0]])
            for i, trace in enumerate(tmp_traces):
                result[trace[-1]] = max(result[trace[-1]], tmp_scores[i])
            results.append(result)
            traces=tmp_traces
            scores=tmp_scores
        else:
            break
    traces.clear()
    scores.clear()
    return results

def calc_score(traces,chain_ix,this_seq):
    """
    Calculate scores for traces based on RMSD, neighbor, and AA scores.
    
    Args:
        traces: List of candidate traces
        chain_ix: Chain index
        this_seq: Sequence indices
        
    Returns:
        List of calculated scores
    """
    global globalParam
    result=[]
    for trace in traces:
        rmsd_scores=[]
        for mc in globalParam.matched_chains:
            this_coords=[]
            for p in mc[0]:
                this_coords.append(globalParam.CA_cands[trace[p]])
            rmsd_scores.append(superpose3d.Superpose3D(this_coords,mc[1])[0][0]*len(mc[0])/len(trace))
        if rmsd_scores:
            rmsd_score = min(max(np.mean(rmsd_scores)-1,0)/2,3)
        else:
            rmsd_score = 0

        neigh_score=globalParam.neigh_mat[trace[:-1],trace[1:]].mean()
        AA_score=globalParam.chain_cand_mat[chain_ix, this_seq, trace].mean()
        result.append(neigh_score+AA_score-rmsd_score)
    return result

def calc_dis(coordList1, coordList2):
    y = [coordList2 for _ in coordList1]
    y = np.array(y)
    x = [coordList1 for _ in coordList2]
    x = np.array(x)
    x = x.transpose(1, 0, 2)
    a = np.linalg.norm(np.array(x) - np.array(y), axis=2)
    return a

def localSeqStructAlign(fasta_ix,fasta_name,sub_seq):
    """
    Perform local sequence-structure alignment.
    
    Args:
        fasta_ix: FASTA index
        fasta_name: FASTA name
        sub_seq: Subsequence range
        
    Returns:
        List of alignment scores
    """
    global globalParam
    AF3_split = globalParam.fastas[fasta_name].AF3_struct[sub_seq]
    score_list=[]
    for trace in globalParam.trace_list:
        AA_score =globalParam.seq_cand_AA_mat[fasta_ix,sub_seq,trace].mean()
        nei_score =globalParam.neigh_mat[trace[:-1],trace[1:]].mean()
        this_coords = globalParam.CA_cands[trace]
        AF3_rmsd = superpose3d.Superpose3D(this_coords,AF3_split)[0][0]
        score_list.append([AA_score,nei_score,AF3_rmsd])
    return score_list

def registerScoring(fasta_ix,fasta_name,seq_ix,radius):
    """
    Perform register scoring for structure-based alignment.
    
    Args:
        fasta_ix: FASTA index
        fasta_name: FASTA name
        seq_ix: Sequence index
        radius: Radius for scoring
        
    Returns:
        List of scoring results
    """
    global globalParam
    this_seq=range(seq_ix-radius,seq_ix+radius+1)
    this_fasta=globalParam.fastas[fasta_name]
    AF3_split = this_fasta.AF3_struct[this_seq]
    chain_num = len(this_fasta.chain_dict)
    chain_list = list(this_fasta.chain_dict.keys())
    
    item_list=[]
    score_list=[]
    
    cand_set=np.where(globalParam.seq_cand_AA_mat[fasta_ix,seq_ix] > globalParam.seq_cand_AA_mat[fasta_ix,seq_ix].max()*0.85)[0]
    for cand in cand_set:
        trace=[cand]
        for i in range(radius):
            max_score=-1
            max_nei=-1
            mean_score = globalParam.seq_cand_AA_mat[fasta_ix, seq_ix+1+i].mean()
            for nei in set(globalParam.neighbors2to6[trace[-1]]) - set(trace):
                score = globalParam.seq_cand_AA_mat[fasta_ix, seq_ix+1+i, nei]
                if score > max_score:
                    max_nei = nei
                    max_score = score
            if max_score > mean_score:
                trace=trace+[max_nei]
            else:
                trace=[]
                break

            max_score=-1
            max_nei=-1
            mean_score = globalParam.seq_cand_AA_mat[fasta_ix, seq_ix-1-i].mean()
            for nei in set(globalParam.neighbors2to6[trace[0]]) - set(trace):
                score = globalParam.seq_cand_AA_mat[fasta_ix, seq_ix-1-i, nei]
                if score > max_score:
                    max_nei = nei
                    max_score = score
            if max_score > mean_score:
                trace=[max_nei]+trace
            else:
                trace=[]
                break
        if trace:
            this_coords = globalParam.CA_cands[trace]
            rmsd, R,T,_ = superpose3d.Superpose3D(this_coords,AF3_split)
            # if rmsd[0]<8:
            new_AF3 = np.dot(this_fasta.AF3_struct, R.T)+T
            trans_AF3 = np.round(new_AF3).astype(int)
            trans_AF3 = trans_AF3[np.where(np.sum(trans_AF3>=0,axis=1)==3)]
            trans_AF3 = trans_AF3[np.where(np.sum(trans_AF3 < globalParam.CAProb.shape, axis=1) == 3)]
            CA_prob_sum = np.sum(globalParam.CAProb[trans_AF3[:, 0], trans_AF3[:, 1], trans_AF3[:, 2]])
            item_list.append([trace,new_AF3[this_seq]])
            score_list.append(CA_prob_sum)
    
    

    results=[]
    if score_list:
        sort_ix = np.argsort(score_list)[::-1]
        for ix in sort_ix:
            trace,new_AF3 = item_list[ix]
            score = score_list[ix]
            this_coords = globalParam.CA_cands[trace]
            if len(results) < 3*chain_num:
                val=True
                for chain in results:
                    if np.sqrt(np.sum((chain[3]-this_coords)**2, axis=1)).mean()<8:
                        val=False
                        break
                if val:
                    results.append([score,trace,this_seq,new_AF3])
            else:
                break
    
    return results

def run_pulchra(dir, pulchar_path, pdbfile, map_id, pdb_id, logger):
    """
    Run Pulchra to generate all-atom model from CA-only model.
    
    Args:
        dir: Output directory
        pulchar_path: Path to Pulchra executable
        pdbfile: Input PDB file (CA-only)
        map_id: Cryo-EM map ID
        pdb_id: PDB ID
        logger: Logger instance
        
    Returns:
        Path to generated all-atom model or None if failed
    """
    logger.info(f"Starting Pulchra processing for PDB file: {pdbfile}")
    cspath = os.path.join(dir, f'chain_split/{map_id}_{pdb_id}')
    
    if os.path.exists(cspath):
        shutil.rmtree(cspath)
    os.makedirs(cspath)

    if not os.path.exists(pdbfile):
        logger.error(f'CA model {pdbfile} does not exist!')
        return None

    prefix = pdbfile.split('/')[-1].split('.')[0]
    lastcid = ''
    lastrid = ''
    newlines = []
    
    if not os.path.exists(cspath):
        os.makedirs(cspath)

    # Split PDB by chains
    logger.info("Splitting PDB file by chains...")
    with open(pdbfile, 'r') as aafile:
        lines = aafile.readlines()
        for l in lines:
            if not l.startswith('ATOM'):
                continue
            cid = l[21]
            rid = int(l[22:26])
            if lastcid == '':
                lastcid = cid

            if lastrid == '':
                lastrid = rid
            
            if lastcid != cid or rid - lastrid not in [0, 1]:
                if len(newlines) > 3:
                    rid_name = f'{lastrid//1000%10}{lastrid//100%10}{lastrid//10%10}{lastrid%10}'
                    with open(os.path.join(cspath, f'{prefix}_{lastcid}_{rid_name}.pdb'), 'w') as newfile:
                        for nl in newlines:
                            newfile.writelines(nl)
                newlines = []
                lastcid = cid
            newlines.append(l)
            lastrid = rid

        if len(newlines) > 3:
            rid_name = f'{lastrid//1000%10}{lastrid//100%10}{lastrid//10%10}{lastrid%10}'
            with open(os.path.join(cspath, f'{prefix}_{lastcid}_{rid_name}.pdb'), 'w') as newfile:
                for nl in newlines:
                    newfile.writelines(nl)

    # Run Pulchra on each chain
    logger.info("Running Pulchra reconstruction on chain fragments...")
    pulchar_path = os.path.abspath(pulchar_path)
    process_list = []
    n_job_per_node = 30
    filelist = os.listdir(cspath)
    
    for f in filelist:
        prefix = f.split('.')[0]
        
        if 'rebuilt' not in f and 'pdb' in f and not os.path.exists(os.path.join(cspath, prefix + '.rebuilt.pdb')):
            command = pulchar_path + ' {} -c '.format(f)
            args = shlex.split(command)
            with open(os.path.join(cspath, '{}.log'.format(prefix)), 'w') as log:
                if len(process_list) < n_job_per_node:
                    process_list.append(subprocess.Popen(args, cwd=cspath, stdout=log))
                else:
                    have_finished = False
                    while True:
                        for i in range(len(process_list)):
                            if process_list[i].poll() is not None:
                                process_list[i] = subprocess.Popen(args, cwd=cspath, stdout=log)
                                have_finished = True
                                break
                        if have_finished:
                            break
                        time.sleep(0.5)
    
    # Wait for all processes to complete
    for p in process_list:
        p.wait()
    
    # Combine rebuilt chains into single all-atom model
    logger.info("Combining rebuilt chains into all-atom model...")
    files = os.listdir(cspath)
    file_list = []
    for f in files:
        if '.rebuilt' in f:
            file_list.append(f)
    file_list.sort()
    
    aid = 1
    all_atom_model = pdbfile.split('_ca_model.pdb')[0] + '_all_atom_model.pdb'
    
    with open(all_atom_model, 'w') as acf:
        for f in file_list:
            cid = f.split('_ca_model')[-1].split('_')[1]
            with open(os.path.join(cspath, f), 'r') as pf:
                lines = pf.readlines()
                for l in lines:
                    if l.startswith('ATOM') and 'nan' not in l:
                        if len(l) < 70:
                            acf.write(l[:4] + str(aid).rjust(7, ' ') + l[11:21] + cid + l[22:54])
                            acf.write(f'  1.00  0.00           {l[13]}\n')
                        else:
                            acf.write(l[:4] + str(aid).rjust(7, ' ') + l[11:21] + cid + l[22:])
                        aid += 1
    
    logger.info(f"✓ Pulchra processing completed. All-atom model saved: {all_atom_model}")
    return all_atom_model


def get_seq(inp):
    seq_obj,chain_strs,protocol,structure_dir,pdbParser=inp
    un_exist=False
    fasta_name=seq_obj.fasta_name
    if protocol=='AF3_struct':
        AF3_path=os.path.join(structure_dir, fasta_name,'ranked_0.pdb')
        if os.path.exists(AF3_path):
            sub_seq_range=None
            AF3_s = pdbParser.get_structure(fasta_name, AF3_path)
            chain_ids = []
            for model in AF3_s:
                for chain in model:
                    chain_ids.append(chain.id)
            seq_obj.AF3_raw_structs = AF3_s
            AF3_struct=[]
            af_seq=''
            for residue in AF3_s[0][chain_ids[0]]:
                if sub_seq_range and not (sub_seq_range[0]< residue.id[1] <=sub_seq_range[1]):
                    continue
                if 'CA' in residue:
                    AF3_struct.append(residue['CA'].get_coord())
                    if residue.get_resname() in AA2abb:
                        af_seq+=AA2abb[residue.get_resname()]
                    else:
                        af_seq+='A'
            seq_obj.AF3_struct = np.array(AF3_struct)
            seq_obj.sequence = af_seq
        else:
            un_exist=True
    
    
    return seq_obj,chain_strs,un_exist

class Solver:
    def __init__(self, modeling_config):
        """
        Initialize the MICA solver.
        
        Args:
            modeling_config: Configuration object with all parameters
        """
        # Initialize logging first
        self.setup_logging()
        
        # Determine method name based on protocol
        if modeling_config.protocol == 'AF3_struct_free':
            self.method_name = 'MICA_TempFree'
        elif modeling_config.protocol == 'AF3_struct':
            self.method_name = 'MICA'
        
        # Extract IDs from file paths
        try:
            self.map_id = modeling_config.map_path.split('/')[-1].split('emd_')[-1].split('.')[0]
        except:
            self.map_id = 'unknown map'
        try:
            self.pdb_id = modeling_config.fasta_path.split('/')[-1].split('.fasta')[0]
        except:
            self.pdb_id = 'unknown PDB'
            
        self.resol = modeling_config.resolution
        
        self.logger.info(f"🚀 Starting MICA - Map ID: {self.map_id}, PDB ID: {self.pdb_id}")
        
        # Store configuration and setup paths
        self.modeling_config = modeling_config
        torch.manual_seed(modeling_config.seed)
        self.ca_model_pdb = os.path.join(
            self.modeling_config.output_path, 
            f'{self.map_id}_{self.pdb_id}_{self.method_name}_ca_model.pdb'
        )
        self.time_log = os.path.join(
            self.modeling_config.output_path, 
            f'time_cost_{self.map_id}_{self.pdb_id}_{self.method_name}.csv'
        )

        # Initialize data structures
        self.normEM = None
        self.neighbors2to6 = []
        self.neighbors0to6 = []
        self.neighbors2to7 = []
        self.neighbors0to7 = []
        self.fragModel = []
        self.ResNum = 0
        self.max_seq_len = 0
        self.fasta_list = []
        self.fastas = {}
        self.chain_id_list = []
        self.pdbParser = PDBParser(PERMISSIVE=1)
        self.time_cost = {}

        # Clustering parameters
        self.cluster_eps = modeling_config.cluster_eps
        self.cluster_min_points = modeling_config.cluster_min_points
        self.nms_radius = modeling_config.nms_radius

    def setup_logging(self):
        """Setup logging configuration with timestamps."""
        self.formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(self.formatter)
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)
        self.logger.propagate = False

    def print_clean(self, message):
        """Print message without timestamp or log level for visual appeal."""
        print(message)

    def run(self):
        """
        Run the complete MICA prediction pipeline.
           
        Returns:
            Success status string
        """
        self.print_clean("=" * 80)
        self.print_clean("🧬 MICA PIPELINE STARTING")
        self.print_clean("=" * 80)
        
        # Check sequences (if needed)
        if self.modeling_config.protocol in ['AF3_struct_free', 'AF3_struct']:
            checkSeqRes = self.checkSeq()
            if checkSeqRes != 'success':
                return checkSeqRes
        
        # Neural network processing
        self.nnProcess()
        
        # Fragment modeling
        start_time = time.time()
        self.print_clean("=" * 80)
        self.print_clean("🧩 GENERATING INITIAL FRAGMENTS")
        self.print_clean("=" * 80)
        self.logger.info("Starting fragment modeling...")
        self.fragModeling()
        self.time_cost['fragModeling'] = time.time() - start_time
        self.logger.info(f"✓ Fragment modeling completed in {self.time_cost['fragModeling']:.2f} seconds")

        # Sequence mapping and modeling
        if self.modeling_config.protocol in ['AF3_struct_free', 'AF3_struct']:
            run_result = self.seqMapAligning()
            if run_result != 'success':
                self.logger.error(f"Sequence mapping alignment failed: {run_result}")
                return run_result
            
            start_time = time.time()
            self.print_clean("=" * 80)
            self.print_clean("🏗️  BUILDING INITIAL MODEL")
            self.print_clean("=" * 80)
            self.logger.info("Starting initial modeling...")
            self.initialModelBuilding()
            self.time_cost['initialModelBuilding'] = time.time() - start_time
            self.logger.info(f"✓ Initial model building completed in {self.time_cost['initialModelBuilding']:.2f} seconds")

            start_time = time.time()
            self.print_clean("=" * 80)
            self.print_clean("🧵 RUNNING GAP FILLING")
            self.print_clean("=" * 80)
            self.logger.info("Starting gap filling...")
            self.gapFilling()
            self.time_cost['gapFilling'] = time.time() - start_time
            self.logger.info(f"✓ Gap filling completed in {self.time_cost['gapFilling']:.2f} seconds")

        # Pulchra reconstruction
        if self.modeling_config.run_pulchra:
            self.print_clean("=" * 80)
            self.print_clean("🔧 RUNNING PULCHRA")
            self.print_clean("=" * 80)
            self.logger.info("Starting Pulchra reconstruction...")
            start_time = time.time()
            all_atom_model = run_pulchra(
                self.modeling_config.output_path, 
                self.modeling_config.pulchra_path, 
                self.ca_model_pdb, 
                self.map_id, 
                self.pdb_id, 
                self.logger
            )
            self.time_cost['run_pulchra'] = time.time() - start_time
            self.logger.info(f"✓ Pulchra completed in {self.time_cost['run_pulchra']:.2f} seconds")
        
            # Phenix refinement
            if all_atom_model and self.modeling_config.run_phenix:
                self.print_clean("=" * 80)
                self.print_clean("⚡ RUNNING PHENIX REFINEMENT")
                self.print_clean("=" * 80)
                self.logger.info("Starting Phenix refinement...")
                start_time = time.time()
                self.phenix_refine(all_atom_model)
                self.time_cost['phenix_refine'] = time.time() - start_time
                self.logger.info(f"✓ Phenix refinement completed in {self.time_cost['phenix_refine']:.2f} seconds")
        
        # Record timing and finish
        self.time_record()
        
        self.print_clean("=" * 80)
        self.print_clean("🎉 MICA PIPELINE COMPLETED SUCCESSFULLY")
        self.print_clean("=" * 80)
        self.print_clean(f"📊 Total processing time: {sum(self.time_cost.values()):.2f} seconds")
        self.print_clean(f"📁 Final model saved: {all_atom_model}")
        self.print_clean("")
        
        return 'success'

    def nnProcess(self):
        """
        Process neural network predictions and clustering.
        """
        # EM preprocessing
        start_time = time.time()
        print("")
        print("=" * 80)
        print("🚀 STARTING DEEP LEARNING PREDICTION")
        print("=" * 80)
        print(f"📊 System Information:")
        print(f"   🖥️  Device: {self.modeling_config.device}")
        print(f"   🧠 CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
        print(f"   💾 System Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        if self.modeling_config.device == 'cuda' and torch.cuda.is_available():
            print(f"   🎮 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        print("")
        self.logger.info("Starting data preprocessing...")
        self.getData()
        self.time_cost['getData'] = time.time() - start_time
        self.logger.info(f"✓ Data preprocessing completed in {self.time_cost['getData']:.2f} seconds")

        # Deep Learning prediction
        start_time = time.time()
        self.logger.info("Starting Deep Learning prediction...")
        self.nnPred()
        self.time_cost['nnPred'] = time.time() - start_time
        self.logger.info(f"✓ Deep Learning prediction completed in {self.time_cost['nnPred']:.2f} seconds")
        
        # Clustering
        start_time = time.time()
        print("")
        self.print_clean("=" * 80)
        self.print_clean("🔗 CLUSTERING CANDIDATES")
        self.print_clean("=" * 80)
        self.logger.info("Starting candidate clustering...")
        self.clustering()
        self.time_cost['clustering'] = time.time() - start_time
        self.logger.info(f"✓ Clustering completed in {self.time_cost['clustering']:.2f} seconds")

    def getData(self):
        """Preprocess cryo-EM map and AF3 predicted data."""
        data_processor = DataPreprocessor(map_path=self.modeling_config.map_path, 
                                          AF3_results=self.modeling_config.AF3_results, 
                                          quiet=self.modeling_config.quiet
                                          )   
        data_processor.resample_and_normalize_map()           
        combined_docked_model_path = os.path.join(os.path.dirname(self.modeling_config.AF3_results), f'{os.path.basename(os.path.dirname(self.modeling_config.AF3_results))}_af3_docked.pdb')      
        if os.path.exists(combined_docked_model_path):
            encoding_result = data_processor.create_AF3_encodings(combined_docked_model_path)        
        
        grid_creator = GridCreator(quiet=self.modeling_config.quiet)
        results = {}
        overall_success = True
        
        map_grids_dir = os.path.join(self.modeling_config.grids_path, "normalized_map_grids")
        result = grid_creator.create_normalized_map_grids(
            normalized_map_path=self.modeling_config.normalized_map_path,
            output_dir=map_grids_dir,
        )
                
        self.offset = result['offset']
        results['normalized_map'] = result
        if not result["success"]:
            overall_success = True
        
        
        af3_grids_dir = os.path.join(self.modeling_config.grids_path, "AF3_encoding_grids")
        if os.path.exists(combined_docked_model_path):
            result = grid_creator.create_AF3_encodings_grids(
                AF3_encodings_path=self.modeling_config.AF3_encodings_path,
                output_dir=af3_grids_dir,
                parallel=not self.modeling_config.no_parallel
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
        
        self.logger.info(f"Cryo-EM map and AF3 results processed. Map Offset: {self.offset}")
                          
    def nnPred(self):
        """Neural network predictions from pre-computed files."""
        predictor = CryoEMPredictor(
        model_path=self.modeling_config.model_path,
        grids_path=self.modeling_config.grids_path,
        output_path=self.modeling_config.output_path,
        save_output=False,
        device=self.modeling_config.device,
        quiet=self.modeling_config.quiet
        )
        
        try:
            success, reconstructed_volumes = predictor.run_prediction() 
            self.AAPred = reconstructed_volumes['amino_acid_prediction']
            NNPred.AAProb = reconstructed_volumes['amino_acid_probability']
            NNPred.BBProb = reconstructed_volumes['backbone_probability']
            self.CAProb =  reconstructed_volumes['carbon_alpha_probability']
        
        except KeyboardInterrupt:
            print("\n❌ Process interrupted by user")
            sys.exit(130)
        except Exception as e:
            predictor.logger.error(f"Unexpected error: {e}")
            print(f"\n❌ Unexpected error: {e}")
            sys.exit(1)
            
        self.logger.info(f"✓ Deep Learning predictions completed...")
        self.logger.info(f"   📊 Amino acid prediction shape: {self.AAPred.shape}")
        self.logger.info(f"   📊 Amino acid probability shape: {NNPred.AAProb.shape}")
        self.logger.info(f"   📊 Backbone probability shape: {NNPred.BBProb.shape}")
        self.logger.info(f"   📊 CA probability shape: {self.CAProb.shape}")
        self.logger.info(f"Deleting all grids, normalized density map and AF3 encodings for disk space saving...")
        try:
            shutil.rmtree(self.modeling_config.grids_path) 
            os.remove(self.modeling_config.normalized_map_path) 
            shutil.rmtree(self.modeling_config.AF3_encodings_path) 
            self.logger.info(f"✓ Deleted all grids, normalized density map and AF3 encodings for disk space saving...")
        except:
            self.logger.info(f"❌ Failed to delete some temporary files. Continuing...")          
                          
    def clustering(self):
        """Perform clustering of CA candidates."""
        self.logger.info(f"Starting clustering with eps={self.cluster_eps}, min_points={self.cluster_min_points}")
        self.logger.info(f"CA score threshold: {self.modeling_config.CA_score_thrh}")
        
        pcd_numpy = np.array(np.where(self.CAProb > self.modeling_config.CA_score_thrh)).T
        pcd_raw = o3d.geometry.PointCloud()
        pcd_raw.points = o3d.utility.Vector3dVector(pcd_numpy)
        labels = np.array(pcd_raw.cluster_dbscan(eps=self.cluster_eps, min_points=self.cluster_min_points))
        
        self.logger.info(f"Initial point cloud size: {pcd_numpy.shape[0]}")
        self.logger.info(f"Number of clusters found: {labels.max()+1}")
        
        # Calculate cluster scores
        labels_scores_sum = []
        for label in range(labels.max()+1):
            pcd = pcd_numpy[np.where(labels == label)]
            labels_scores_sum.append(np.sum(NNPred.BBProb[pcd[:,0],pcd[:,1],pcd[:,2]]))
        
        labels_scores_avg = []
        for label in range(labels.max()+1):
            if labels_scores_sum[label] > np.max(labels_scores_sum)/10:
                pcd = pcd_numpy[np.where(labels == label)]
                labels_scores_avg.append(np.mean(NNPred.BBProb[pcd[:,0],pcd[:,1],pcd[:,2]]))
            else:
                labels_scores_avg.append(0)
        
        # Filter valid clusters
        val_mat = np.zeros_like(labels).astype(bool)
        max_labels_score = np.max(labels_scores_avg)
        for label in range(labels.max()+1):
            if labels_scores_avg[label] > max_labels_score/2:
                val_mat[np.where(labels == label)] = True
        
        clustered_coords = pcd_numpy[np.where(val_mat)]
        self.logger.info(f"Valid clusters selected: {np.sum(val_mat)} points from {pcd_numpy.shape[0]}")
        
        # Create clustered probability map
        NNPred.CAProb_clusted = np.zeros_like(self.CAProb)
        NNPred.CAProb_clusted[clustered_coords[:, 0], clustered_coords[:, 1], clustered_coords[:, 2]] \
            = self.CAProb[clustered_coords[:, 0], clustered_coords[:, 1], clustered_coords[:, 2]]
        
        # Build prediction list
        pred_list = []
        indexes = np.where(val_mat)
        for i in range(indexes[0].shape[0]):
            pred_list.append([
                NNPred.CAProb_clusted[pcd_numpy[indexes[0][i]][0], pcd_numpy[indexes[0][i]][1], pcd_numpy[indexes[0][i]][2]], 
                pcd_numpy[indexes[0][i]][0], 
                pcd_numpy[indexes[0][i]][1], 
                pcd_numpy[indexes[0][i]][2]
            ])
        
        pred_list = np.array(pred_list)
        pred_list = pred_list[np.argsort(-pred_list[:, 0], axis=0)]
        
        self.logger.info(f"Processing {pred_list.shape[0]} candidate positions...")
        
        # Non-maximum suppression
        CA_cands = []
        clustered_map = np.zeros_like(self.CAProb)
        
        while (pred_list.shape[0] > 0 and pred_list[0][0] >= self.modeling_config.CA_score_thrh):
            CA_cands.append([int(pred_list[0, 1]), int(pred_list[0, 2]), int(pred_list[0, 3])])
            clustered_map[int(pred_list[0, 1]), int(pred_list[0, 2]), int(pred_list[0, 3])] = 1
            delete_list = np.where(
                (pred_list[:, 1] - pred_list[0, 1]) ** 2 + (pred_list[:, 2] - pred_list[0, 2]) ** 2 + (
                        pred_list[:, 3] - pred_list[0, 3]) ** 2 <= self.nms_radius)
            pred_list = np.delete(pred_list, delete_list, 0)
        
        self.logger.info(f"Final CA candidates: {len(CA_cands)}")
        
        # Refine candidate positions
        new_cands = []
        new_AAs = []
        for cand in CA_cands:
            try:
                coord = [0, 0, 0]
                AA_list = []
                cand = np.array(cand)
                weights = self.CAProb[cand[0]-1:cand[0]+2, cand[1]-1:cand[1]+2, cand[2]-1:cand[2]+2] / \
                         np.sum(self.CAProb[cand[0]-1:cand[0]+2, cand[1]-1:cand[1]+2, cand[2]-1:cand[2]+2])
                
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        for dk in [-1, 0, 1]:
                            this_coord = cand + [di, dj, dk]
                            coord += this_coord * weights[di+1, dj+1, dk+1]
                            AA_list.append(NNPred.AAProb[:, this_coord[0], this_coord[1], this_coord[2]] * 
                                         weights[di+1, dj+1, dk+1])
                new_cands.append(coord)
                new_AAs.append(np.sum(AA_list, axis=0))
            except:
                self.logger.warning('Candidate atom found at boundary, skipping...')
                
        self.CA_cands = np.array(new_cands)
        self.CA_cands_AAProb = np.array(new_AAs).T
        round_cands = np.round(self.CA_cands).astype(int)
        self.CA_cands_AA = self.AAPred[round_cands[:,0], round_cands[:,1], round_cands[:,2]]
        
        # Calculate distances and build neighbor lists
        self.cand_self_dis = calc_dis(self.CA_cands, self.CA_cands)
        
        self.logger.info("Building neighbor lists...")
        for i in range(self.CA_cands.shape[0]):
            self.neighbors2to6.append(np.where((self.cand_self_dis[i] <= 6) * (self.cand_self_dis[i] >= 2))[0])
        for i in range(self.CA_cands.shape[0]):
            self.neighbors0to6.append(np.where(self.cand_self_dis[i] <= 6)[0])
        for i in range(self.CA_cands.shape[0]):
            self.neighbors0to7.append(np.where(self.cand_self_dis[i] <= 7)[0])
        for i in range(self.CA_cands.shape[0]):
            self.neighbors2to7.append(np.where((self.cand_self_dis[i] <= 7) * (self.cand_self_dis[i] >= 2))[0])

        # Compute neighbor scoring matrix
        self.logger.info("Computing neighbor scoring matrix...")
        self.neigh_mat = np.zeros_like(self.cand_self_dis)
        for cand in range(self.CA_cands.shape[0]):
            for neigh in self.neighbors2to6[cand]:
                BB_dens = 0
                dis = max(0, abs(self.cand_self_dis[cand, neigh] - 3.8) - 0.5)
                dis_score = max(0, 1 - dis / 2)
                for j in range(1, 5):
                    coord = np.round(j/5 * self.CA_cands[neigh] + (5-j)/5 * self.CA_cands[cand]).astype(int)
                    BB_dens += NNPred.BBProb[coord[0], coord[1], coord[2]]
                self.neigh_mat[cand, neigh] = (dis_score + BB_dens/4) / 2
        
        # Build best neighbor lists
        self.best_neigh = []
        for cand in range(self.CA_cands.shape[0]):
            neigh_list = []
            second, first = self.neigh_mat[cand].argsort()[-2:]
            if self.neigh_mat[cand, first] != 0:
                neigh_list.append(first)
            if self.neigh_mat[cand, second] != 0:
                neigh_list.append(second)
            self.best_neigh.append(neigh_list)
        
        self.logger.info(f"✓ Clustering completed - {len(self.best_neigh)} candidates with neighbors")
    
    def fragModeling(self):
        """Perform fragment modeling using graph-based approach."""
        self.logger.info(f"Starting fragment modeling with protocol: {self.modeling_config.protocol}")
        
        cand_graph = nx.Graph()
        edge_list = []
        
        self.logger.info(f"CA candidates shape: {self.CA_cands.shape}")
        self.logger.info(f"Neighbors 2to6 length: {len(self.neighbors2to6)}")
        self.logger.info(f"Neighbor matrix shape: {self.neigh_mat.shape}")
        
        # Build initial graph
        for cand in range(self.CA_cands.shape[0]):
            for neigh in self.neighbors2to6[cand]:
                if neigh > cand:
                    cand_graph.add_edge(cand, neigh)
                    edge_list.append([self.neigh_mat[cand, neigh], cand, neigh])
        
        edge_list = np.array(edge_list)
        self.logger.info(f"Initial edge list: {len(edge_list)} edges")
        
        # Prune edges to create linear fragments
        edge_list = edge_list[np.argsort(edge_list[:, 0], axis=0)]
        new_edge_list = []
        for edge in edge_list:
            cand = round(edge[1])
            neigh = round(edge[2])
            if cand_graph.degree(cand) > 2 and cand_graph.degree(neigh) > 2:
                cand_graph.remove_edge(cand, neigh)
            else:
                new_edge_list.append([self.neigh_mat[cand, neigh], cand, neigh])

        edge_list = np.array(new_edge_list)
        edge_list = edge_list[np.argsort(edge_list[:, 0], axis=0)]
        for edge in edge_list:
            cand = round(edge[1])
            neigh = round(edge[2])
            if cand_graph.degree(cand) > 2 or cand_graph.degree(neigh) > 2:
                cand_graph.remove_edge(cand, neigh)
        
        # Extract linear fragments
        fragments = []
        tmp_graph = cand_graph.copy()
        
        # Start with terminal nodes
        for node in cand_graph.nodes:
            if tmp_graph.degree(node) == 1:
                next_node = list(tmp_graph[node])[0]
                frag = [node, next_node]
                tmp_graph.remove_edge(node, next_node)
                while tmp_graph.degree(next_node) == 1:
                    neigh = list(tmp_graph[next_node])[0]
                    frag.append(neigh)
                    tmp_graph.remove_edge(next_node, neigh)
                    next_node = neigh
                fragments.append(frag)

        # Handle remaining cycles by breaking weakest edges
        while len(tmp_graph.edges()) > 0:
            edge_scores = []
            for node, neigh in tmp_graph.edges():
                edge_scores.append([self.neigh_mat[node, neigh], node, neigh])

            edge_scores = np.array(edge_scores)
            min_edge = edge_scores[np.argmin(edge_scores[:, 0])]
            node = round(min_edge[1])
            tmp_graph.remove_edge(node, round(min_edge[2]))
            
            if tmp_graph.degree(node) == 1:
                next_node = list(tmp_graph[node])[0]
                frag = [node, next_node]
                tmp_graph.remove_edge(node, next_node)
                while tmp_graph.degree(next_node) == 1:
                    neigh = list(tmp_graph[next_node])[0]
                    frag.append(neigh)
                    tmp_graph.remove_edge(next_node, neigh)
                    next_node = neigh
                fragments.append(frag)
        
        self.logger.info(f"Initial fragments generated: {len(fragments)}")
        
        # Merge fragments if too many
        max_frags = min(62, self.CA_cands.shape[0] // self.modeling_config.frags_len + 1)
        if len(fragments) > max_frags:
            tmp_fragments = copy.deepcopy(fragments)
            self.logger.info(f"Reducing fragments from {len(tmp_fragments)} to {max_frags}")
            
            while len(tmp_fragments) > max_frags:
                disMap = np.full((2*len(tmp_fragments), 2*len(tmp_fragments)), 10000)
                for i, frag1 in enumerate(tmp_fragments):
                    for j, frag2 in enumerate(tmp_fragments):
                        if i != j:
                            disMap[2*i, 2*j] = self.cand_self_dis[frag1[0], frag2[0]]
                            disMap[2*i+1, 2*j] = self.cand_self_dis[frag1[-1], frag2[0]]
                            disMap[2*i, 2*j+1] = self.cand_self_dis[frag1[0], frag2[-1]]
                            disMap[2*i+1, 2*j+1] = self.cand_self_dis[frag1[-1], frag2[-1]]
                
                best_ix = np.unravel_index(disMap.argmin(), disMap.shape)
                best_i, best_j = best_ix[0]//2, best_ix[1]//2
                
                left_trace = copy.deepcopy(tmp_fragments[best_i] if best_ix[0]%2==1 else tmp_fragments[best_i][::-1])
                right_trace = copy.deepcopy(tmp_fragments[best_j] if best_ix[1]%2==0 else tmp_fragments[best_j][::-1])

                new_frag = left_trace + right_trace
                if best_i > best_j:
                    del tmp_fragments[best_i]
                    del tmp_fragments[best_j]
                else:
                    del tmp_fragments[best_j]
                    del tmp_fragments[best_i]
                tmp_fragments.append(new_frag)
            fragments = tmp_fragments

        self.logger.info(f'Final fragment count: {len(fragments)}')
    
    def seqMapAligning(self):
        """Perform sequence-structure alignment mapping."""
        self.print_clean("=" * 80)
        self.print_clean("🔗 Cα-SEQUENCE ALIGNMENT")
        self.print_clean("=" * 80)
        
        self.prepareSeq4Align()
        
        if self.modeling_config.protocol == 'AF3_struct_free':
            start_time = time.time()
            self.logger.info("Starting Cα-sequence alignment...")
            if not self.seqStructureAlign():
                return 'seqStructureAlign error! this case is too hard!'
            self.time_cost['seqStructureAlign'] = time.time() - start_time
            self.logger.info(f"✓ Cα-sequence alignment completed in {self.time_cost['seqStructureAlign']:.2f} seconds")
            
        elif self.modeling_config.protocol == 'AF3_struct':
            start_time = time.time()
            self.logger.info("Starting Cα-sequence alignment with AF3 structure...")
            self.seqStructAlignWithAF3Structure()
            self.time_cost['seqStructAlignWithAF3Structure'] = time.time() - start_time
            self.logger.info(f"✓ AF3 structure-based alignment completed in {self.time_cost['seqStructAlignWithAF3Structure']:.2f} seconds")

        return 'success'

    def prepareSeq4Align(self):
        """Prepare sequence data for alignment."""
        self.seq_cand_AA_mat = np.zeros([len(self.fastas), self.max_seq_len, self.CA_cands.shape[0]]).astype(float)
        self.logger.info(f"Sequence candidate AA matrix shape: {self.seq_cand_AA_mat.shape}")
        self.logger.info(f"CA candidates AA probability shape: {self.CA_cands_AAProb.shape}")
        
        for i, fasta_name in enumerate(self.fastas):
            for j, AA in enumerate(self.fastas[fasta_name].sequence):
                for k, coord in enumerate(self.CA_cands):
                    if AA in AA_abb:   
                        self.seq_cand_AA_mat[i, j, k] = self.CA_cands_AAProb[AA_abb[AA], k]

    def seqStructureAlign(self):
        """Perform Cα-sequence alignment without AF3 structures."""
        self.logger.info("Starting Cα-sequence alignment without AF3 structures...")
        self.n_hop_mat = self.getNHopMat()

        connect_len = 5
        self.seq_cand_AA_mat_copy = self.seq_cand_AA_mat.copy()
        self.logger.info(f"First alignment pass with connect_len={connect_len}")
        self.quasiSeqAlign(connect_len=connect_len)
        if not self.alignedFrags:
            self.logger.error("No aligned fragments found in first pass")
            return False

        connect_len = 9
        self.seq_cand_AA_mat_copy[np.where(self.cand_match_result > 0)] = 1
        self.logger.info(f"Second alignment pass with connect_len={connect_len}")
        self.quasiSeqAlign(connect_len=connect_len)

        if not self.alignedFrags:
            self.logger.error("No aligned fragments found in second pass")
            return False
        
        self.logger.info("✓ Cα-sequence alignment completed successfully")
        return True

    def getNHopMat(self):
        """Compute N-hop matrix for path walking."""
        self.logger.info(f"Computing N-hop matrix with n_hop={self.modeling_config.n_hop}")
        
        n_hop_mat = np.zeros([self.modeling_config.n_hop,self.cand_self_dis.shape[0],self.cand_self_dis.shape[0]])
        global globalParam
        globalParam.update(self.seq_cand_AA_mat, self.neigh_mat,neighbors2to6=self.best_neigh)
        pool = Pool(self.modeling_config.mul_proc_num)
        async_results = []
        for cand in range(self.CA_cands.shape[0]):
            async_results.append(pool.apply_async(pathWalking, args=(cand,self.modeling_config.n_hop)))
        pool.close()
        pool.join()

        for cand, async_result in enumerate(async_results):
            results=async_result.get()
            for n, res in enumerate(results):
                n_hop_mat[n,cand]=res
        for n in range(n_hop_mat.shape[0]):
            for cand in range(n_hop_mat.shape[1]):
                this_sum = np.sum(n_hop_mat[n, cand])
                if this_sum != 0:
                    n_hop_mat[n, cand] /= this_sum
        self.logger.info("✓ N-hop matrix computation completed")
        return n_hop_mat

    def quasiSeqAlign(self,connect_len):
        """Perform quasi-sequence alignment."""
        self.logger.info(f"Running quasi-sequence alignment with connect_len={connect_len}")
        
        self.seq_align_score=self.seq_cand_AA_mat_copy.copy()
        for i in range(self.modeling_config.n_hop):
            self.seq_align_score+=np.pad(self.seq_cand_AA_mat_copy[:,:-i-1,:],[(0,0),(i+1,0),(0,0)],constant_values=(0,0))@self.n_hop_mat[i].T+np.pad(self.seq_cand_AA_mat_copy[:,i+1:,:],[(0,0),(0,i+1),(0,0)],constant_values=(0,0))@self.n_hop_mat[i].T

        self.seq_cand_AA_mat_copy=self.seq_cand_AA_mat.copy()
        sort_ix = (-self.seq_align_score.max(axis=0).max(axis=0)).argsort()
        self.alignedFrags=[[] for _ in range(len(self.fastas))]
        self.cand_match_result=np.zeros_like(self.seq_cand_AA_mat_copy)
        self.logger.info(f"Initialized {len(self.alignedFrags)} aligned fragment lists")
        
        used_cands=set()        
        fragments_found = 0
        
        for cand_ix in sort_ix:
            if cand_ix in used_cands:
                continue
            fasta_ix,seq_ix = np.unravel_index(self.seq_align_score[:,:,cand_ix].argmax(),self.seq_align_score.shape[:2])
            if self.seq_align_score[fasta_ix,seq_ix,cand_ix]<=self.modeling_config.score_thrh:
                continue
            fragment=self.findAlignedFrag(fasta_ix,seq_ix,cand_ix)
            if len(fragment[0])>=connect_len and np.mean(fragment[2]) > self.modeling_config.score_thrh/2:
                self.alignedFrags[fasta_ix].append(fragment)
                fragments_found += 1
                for i, cand in enumerate(fragment[0]):
                    used_cands.add(cand)
                    self.cand_match_result[fasta_ix,fragment[1][i],cand] = fragment[2][i]
                    self.seq_align_score[:,:,cand]=0
                    self.seq_cand_AA_mat_copy[:,:,cand]=0
                    if np.sum(self.cand_match_result[fasta_ix,fragment[1][i]]>0) >= len(self.fastas[self.fasta_list[fasta_ix]].chain_dict):
                        self.seq_align_score[fasta_ix,fragment[1][i],:]=0
                        self.seq_cand_AA_mat_copy[fasta_ix,fragment[1][i],:] = 0
        self.logger.info(f"✓ Found {fragments_found} aligned fragments")

    def findAlignedFrag(self,fasta_ix,seq_ix,cand_ix):
        """Find aligned fragment starting from given position."""
        traces=[[cand_ix]]
        seqs = [[seq_ix]]
        scores=[[self.seq_align_score[fasta_ix,seq_ix,cand_ix]]]
        left_seq = seq_ix
        right_seq=seq_ix
        left_val=left_seq>0
        right_val = right_seq<len(self.fastas[self.fasta_list[fasta_ix]].sequence)-1

        max_scores = self.seq_align_score.max(axis=1)
        while left_val or right_val:
            if left_val:
                left_seq=left_seq-1
                left_val=left_seq>0
                tmp_traces=[]
                tmp_seqs=[]
                tmp_scores=[]
                for i,trace in enumerate(traces):
                    for neigh in self.neighbors0to7[trace[0]]:
                        if self.seq_align_score[fasta_ix,left_seq,neigh] == max_scores[fasta_ix,neigh]>self.modeling_config.score_thrh:
                            tmp_traces.append([neigh]+trace)
                            tmp_seqs.append([left_seq]+seqs[i])
                            tmp_scores.append([self.neigh_mat[neigh,trace[0]]*self.seq_align_score[fasta_ix,left_seq,neigh]]+scores[i])
                if not tmp_traces:
                    left_val=False
                    left_seq += 1
                elif len(tmp_traces)>1:
                    max_ix=None
                    max_score=0
                    for i,trace in enumerate(tmp_traces):
                        score_sum = np.sum(tmp_scores[i])
                        if np.sum(tmp_scores[i])>max_score:
                            max_score=np.sum(score_sum)
                            max_ix =i
                    traces=[tmp_traces[max_ix]]
                    seqs=[tmp_seqs[max_ix]]
                    scores=[tmp_scores[max_ix]]
                

                else:
                    traces=tmp_traces
                    seqs=tmp_seqs
                    scores=tmp_scores

            if right_val:
                right_seq=right_seq+1
                right_val = right_seq<len(self.fastas[self.fasta_list[fasta_ix]].sequence)-1
                tmp_traces=[]
                tmp_seqs=[]
                tmp_scores=[]
                for i,trace in enumerate(traces):
                    for neigh in self.neighbors0to7[trace[-1]]:
                        if self.seq_align_score[fasta_ix,right_seq,neigh]== max_scores[fasta_ix,neigh]>self.modeling_config.score_thrh:
                            tmp_traces.append(trace+[neigh])
                            tmp_seqs.append(seqs[i]+[right_seq])
                            tmp_scores.append(scores[i]+[self.neigh_mat[trace[-1],neigh]*self.seq_align_score[fasta_ix,right_seq,neigh]])
                if not tmp_traces:
                    right_val=False
                    right_seq -= 1
                elif len(tmp_traces)>1:
                    max_ix=None
                    max_score=0
                    for i,trace in enumerate(tmp_traces):
                        score_sum = np.sum(tmp_scores[i])
                        if np.sum(tmp_scores[i])>max_score:
                            max_score=np.sum(score_sum)
                            max_ix =i
                    traces=[tmp_traces[max_ix]]
                    seqs=[tmp_seqs[max_ix]]
                    scores=[tmp_scores[max_ix]]
                else:
                    traces=tmp_traces
                    seqs=tmp_seqs
                    scores=tmp_scores

        max_ix=None
        max_score=0
        for i,trace in enumerate(traces):
            score_sum = np.sum(scores[i])
            if np.sum(scores[i])>max_score:
                max_score=np.sum(score_sum)
                max_ix =i
        if max_ix is not None:
            return [traces[max_ix], seqs[max_ix], scores[max_ix]]
        else:
            return [[], [], []]

    def seqStructAlignWithAF3Structure(self):        
        """Perform Cα-sequence alignment with AF3 structures."""
        self.logger.info("Starting Cα-sequence alignment with AF3 structures...")
        
        self.seqRegisterScoring()
        sort_fasta_ix = np.argsort(self.registerScores)[::-1]
        self.logger.info(f"Processing sequences in order of register scores: {self.registerScores}")

        seq_struct_align_score_copy=self.seq_struct_align_score.copy()
        

        self.my_comp_struct = Structure(self.pdb_id)
        self.my_comp_struct.add(Model(0))
        cand_match_result=np.zeros_like(self.seq_cand_AA_mat)
        used_cand=set()
        self.alignedFrags=[[] for _ in range(len(self.fastas))]
        for fasta_ix in sort_fasta_ix:
            fasta_name=self.fasta_list[fasta_ix]
            this_fasta=self.fastas[fasta_name]
            seq_len=len(this_fasta.sequence)
            chain_num=len(this_fasta.chain_dict)
            chain_list=list(this_fasta.chain_dict.keys())

            self.logger.info(f"Processing sequence {fasta_name} with {seq_len} residues, {chain_num} chains")
            global globalParam
            globalParam.update(seq_struct_align_score_copy, self.neigh_mat,fastas=self.fastas,CA_cands=self.CA_cands,neighbors2to6=self.neighbors2to6,CAProb=self.CAProb)

            pool = Pool(self.modeling_config.mul_proc_num)
            async_results = []
            for seq_ix in range(self.modeling_config.struct_len//2+1,seq_len-self.modeling_config.struct_len//2-1):
                async_results.append(pool.apply_async(registerScoring, args=(fasta_ix,fasta_name,seq_ix,self.modeling_config.struct_len//2+1)))
            pool.close()
            pool.join()

            AF3_match=[]
            AF3_score=[]
            score_mat=np.zeros(self.seq_struct_align_score.shape[1:])
            for async_result in async_results:
                result=async_result.get()
                if len(result)>=chain_num:
                    chains = self.registerExpand(result,fasta_ix)
                    scores=[chain[2] for chain in chains]
                    # print("Chains", len(chains), len(scores))
                    AF3_match.append(chains)
                    AF3_score.append(scores[np.argsort(scores)[-chain_num]])
                    for chain in chains:
                        this_seq,this_trace,score = chain
                        score_mat[this_seq,this_trace]+=score
            sort_ix = np.unravel_index(score_mat.argsort(axis=None)[::-1][:3*chain_num*seq_len],score_mat.shape)
            self.logger.info(f"AF3 matching results: {len(AF3_match)} matches, total score: {np.sum(AF3_score):.2f}")            
            if np.sum(AF3_score) == 0:
                self.logger.warning(f"No AF3 matches found for sequence {fasta_name}")
                continue
            models=[]
            len_list=[]
            

            for i,cand in enumerate(sort_ix[1]):
                if cand in used_cand:
                    continue
                seq_ix = sort_ix[0][i]
                this_trace = [cand]
                left_seq=seq_ix
                while left_seq>0:
                    best_score=0
                    best_nei=-1
                    for nei in set(self.neighbors2to6[this_trace[0]])-used_cand:
                        if self.neigh_mat[this_trace[0],nei]*score_mat[left_seq-1,nei]>best_score and score_mat[left_seq-1,nei]>0.9*score_mat[:,nei].max():
                            best_score=self.neigh_mat[this_trace[0],nei]*score_mat[left_seq-1,nei]
                            best_nei=nei
                    if best_score>100:
                        this_trace=[best_nei]+this_trace
                        left_seq-=1
                    else:
                        break
                
                right_seq=seq_ix
                while right_seq<seq_len-1:
                    best_score=100
                    best_nei=-1
                    for nei in set(self.neighbors2to6[this_trace[-1]])-used_cand:
                        if self.neigh_mat[this_trace[-1],nei]*score_mat[right_seq+1,nei]>best_score and score_mat[right_seq+1,nei]>0.9*score_mat[:,nei].max():
                            best_score=self.neigh_mat[this_trace[-1],nei]*score_mat[right_seq+1,nei]
                            best_nei=nei
                    if best_score>100:
                        this_trace=this_trace+[best_nei]
                        right_seq+=1
                    else:
                        break
                
                if len(this_trace)<20:
                    continue
                this_seq = list(range(left_seq,right_seq+1))[3:-3]
                this_trace=this_trace[3:-3]
                models.append([this_seq,this_trace])
                len_list.append(len(this_trace))
                cand_match_result[fasta_ix,this_seq,this_trace]=1
                score_mat[np.where(cand_match_result[fasta_ix].sum(axis=1)>=chain_num)]=0
                for cand in this_trace:
                    used_cand.add(cand)
                self.alignedFrags[fasta_ix].append([this_trace,this_seq,self.seq_struct_align_score[fasta_ix,this_seq,this_trace]])
                
    def seqRegisterScoring(self):
        """Perform sequence register scoring."""
        self.logger.info("Starting sequence register scoring...")
        self.seqStructScoring()
        result_list=[]
        for fasta_ix in range(len(self.fastas)):
            result=[]
            chain_num=len(self.fastas[self.fasta_list[fasta_ix]].chain_dict)
            for i in range(chain_num):
                chain_result=[0 for _ in range(len(self.fastas[self.fasta_list[fasta_ix]].sequence))]
                result.append(np.array(chain_result))
            result_list.append(result)


        seq_struct_align_score_copy=self.seq_struct_align_score.copy()
        self.registerScores=[]
        for fasta_ix,fasta_name in enumerate(self.fastas):
            
            this_fasta=self.fastas[fasta_name]
            seq_len=len(this_fasta.sequence)
            chain_num=len(this_fasta.chain_dict)

            self.logger.info(f"Computing register scores for {fasta_name}...")
            global globalParam
            globalParam.update(seq_struct_align_score_copy, self.neigh_mat,fastas=self.fastas,CA_cands=self.CA_cands,neighbors2to6=self.neighbors2to6,CAProb=self.CAProb)

            pool = Pool(self.modeling_config.mul_proc_num)
            async_results = []
            for seq_ix in range(self.modeling_config.struct_len//2+1,seq_len-self.modeling_config.struct_len//2-1):
                async_results.append(pool.apply_async(registerScoring, args=(fasta_ix,fasta_name,seq_ix,self.modeling_config.struct_len//2+1)))
            pool.close()
            pool.join()

            max_score=0
            for async_result in async_results:
                result=async_result.get()
                if len(result)>=chain_num and result[chain_num-1][0] > max_score:
                    max_score=result[chain_num-1][0]
            self.registerScores.append(max_score)
            self.logger.info(f"Register score for {fasta_name}: {max_score:.2f}")

    def seqStructScoring(self):
        """Perform Cα-sequence scoring."""
        self.logger.info("Starting Cα-sequence scoring...")
        
        self.n_hop_mat = self.getNHopMat()
        self.local_traces = []

        self.logger.info("Generating local traces...")
        for cand in range(self.CA_cands.shape[0]):
            trace_dict={}
            traces=[[cand]]
            scores=[0]
            for i in range(self.modeling_config.struct_len-1):
                tmp_traces=[]
                tmp_scores=[]
                for j,trace in enumerate(traces):
                    for nei in set(self.best_neigh[trace[-1]]) - set(trace):
                        tmp_traces.append(trace+[nei])
                        tmp_scores.append(scores[j]+self.neigh_mat[trace[-1],nei])
                traces= tmp_traces
                scores=tmp_scores
            for j,trace in enumerate(traces):
                if trace[-1] not in trace_dict or scores[j]> trace_dict[trace[-1]][1]:
                    if scores[j]/(self.modeling_config.struct_len-1)>0.7:
                        trace_dict[trace[-1]]=[trace,scores[j]]
            for key in trace_dict:
                self.local_traces.append(trace_dict[key][0])
        self.logger.info(f'Generated {len(self.local_traces)} local traces')
        assert(self.local_traces)
        
        self.struct_match = np.zeros_like(self.seq_cand_AA_mat)
        global globalParam
        globalParam.update(self.seq_cand_AA_mat,self.neigh_mat,fastas=self.fastas,CA_cands=self.CA_cands,trace_list=self.local_traces)


        for fasta_ix, fasta_name in enumerate(self.fastas):
            chain_num=len(self.fastas[fasta_name].chain_dict)
            async_results=[]
            pool = Pool(self.modeling_config.mul_proc_num)
            for start_j in range(len(self.fastas[fasta_name].sequence)-self.modeling_config.struct_len+1):
                seq= range(start_j,start_j+self.modeling_config.struct_len)
                async_results.append(pool.apply_async(localSeqStructAlign, args=(fasta_ix,fasta_name, seq)))
            pool.close()
            pool.join()
            align_results = []
            for async_result in async_results:
                align_results.append(async_result.get())

            for start_j, align_result in enumerate(align_results):
                seq= range(start_j,start_j+self.modeling_config.struct_len)
                for t,score_list in enumerate(align_result):
                    AA_score,nei_score,AF3_rmsd=score_list
                    score = AA_score+nei_score - min(1,max(0,AF3_rmsd-1))**2
                    for i, s in enumerate(seq):
                        self.struct_match[fasta_ix,s,self.local_traces[t][i]]=max(self.struct_match[fasta_ix,s,self.local_traces[t][i]],score)

        self.struct_match[self.struct_match<0.1]=0.1

        self.struct_match_copy=self.struct_match.copy()
        self.seq_struct_align_score=self.struct_match_copy.copy()
        for i in range(self.modeling_config.n_hop):
            self.seq_struct_align_score+=np.pad(self.struct_match_copy[:,:-i-1,:],[(0,0),(i+1,0),(0,0)],constant_values=(0,0))@self.n_hop_mat[i].T+np.pad(self.struct_match_copy[:,i+1:,:],[(0,0),(0,i+1),(0,0)],constant_values=(0,0))@self.n_hop_mat[i].T
        self.logger.info("✓ Cα-sequence scoring completed")

    def registerExpand(self, chains, fasta_ix):
        """Expand register scoring results."""
        fasta_name=self.fasta_list[fasta_ix]
        this_fasta=self.fastas[fasta_name]
        seq_len=len(this_fasta.sequence)
        chain_num=len(this_fasta.chain_dict)
        score4sort=[]
        for chain in chains:
            score4sort.append(chain[0])
        score_sort_ix =np.argsort(score4sort)[::-1]
        results=[]
        for j in score_sort_ix:
            score,this_trace,seq,_ = chains[j]
            this_seq = list(seq)

            left_seq = this_seq[0]
            right_seq = this_seq[-1]
            left_val = left_seq>0
            right_val = right_seq < seq_len-1
            while left_val or right_val:
                if left_val:
                    check_len = min(len(this_trace),20)
                    this_coords=self.CA_cands[this_trace[:check_len]]
                    this_AF3_coords=this_fasta.AF3_struct[left_seq:left_seq+check_len]
                    rmsd,R,T,_ = superpose3d.Superpose3D(this_coords,this_AF3_coords)
                    old_rmsd=rmsd[0]
                    trans_AF3 = np.dot(this_fasta.AF3_struct, R.T)+T
                    dis_AF3_trace = np.sqrt(np.sum((self.CA_cands-trans_AF3[left_seq-1])**2, axis=1))
                    if old_rmsd < 5 and dis_AF3_trace.min()<3:
                        left_seq-=1
                        this_trace=[dis_AF3_trace.argmin()]+this_trace
                        left_val = left_seq>0
                    else:
                        left_val = False

                if right_val:
                    check_len = min(len(this_trace),20)
                    this_coords=self.CA_cands[this_trace[-check_len:]]
                    this_AF3_coords=this_fasta.AF3_struct[right_seq-check_len+1:right_seq+1]
                    rmsd,R,T,_ = superpose3d.Superpose3D(this_coords,this_AF3_coords)
                    old_rmsd=rmsd[0]
                    trans_AF3 = np.dot(this_fasta.AF3_struct, R.T)+T
                    dis_AF3_trace = np.sqrt(np.sum((self.CA_cands-trans_AF3[right_seq+1])**2, axis=1))
                    if old_rmsd < 5 and dis_AF3_trace.min()<3:
                        right_seq+=1
                        this_trace=this_trace+[dis_AF3_trace.argmin()]
                        right_val = right_seq < seq_len-1
                    else:
                        right_val = False

            this_seq= [_ for _ in range(left_seq,right_seq+1)]
            rmsd,R,T,_ = superpose3d.Superpose3D(self.CA_cands[this_trace],this_fasta.AF3_struct[this_seq])
            trans_AF3=np.dot(this_fasta.AF3_struct, R.T)+T
            trans_AF3 = np.round(trans_AF3).astype(int)
            trans_AF3 = trans_AF3[np.where(np.sum(trans_AF3>=0,axis=1)==3)]
            trans_AF3 = trans_AF3[np.where(np.sum(trans_AF3 < self.CAProb.shape, axis=1) == 3)]
            CA_prob_sum = np.sum(self.CAProb[trans_AF3[:, 0], trans_AF3[:, 1], trans_AF3[:, 2]])
            results.append([this_seq, this_trace,CA_prob_sum])
        return results
 
    def initialModelBuilding(self):
        """Perform initial model building from aligned fragments."""
        self.logger.info(f"Starting initial model building with {len(self.alignedFrags)} aligned fragments")
        for fasta_ix in range(len(self.alignedFrags)):
            self.fastas[self.fasta_list[fasta_ix]].seq_matched_traces=[]
            self.fastas[self.fasta_list[fasta_ix]].trace_matched_seqs=[]
            self.fastas[self.fasta_list[fasta_ix]].trace_scores=[]
            for fragment in self.alignedFrags[fasta_ix]:
                self.fastas[self.fasta_list[fasta_ix]].seq_matched_traces.append(fragment[0])
                self.fastas[self.fasta_list[fasta_ix]].trace_matched_seqs.append(fragment[1])
                AA_score = self.seq_cand_AA_mat[fasta_ix,fragment[1],fragment[0]]
                neigh_score=self.neigh_mat[fragment[0][:-1],fragment[0][1:]]
                self.fastas[self.fasta_list[fasta_ix]].trace_scores.append((AA_score[1:]+AA_score[:-1])*neigh_score)

        self.used_cands=set()
        for fasta_ix, fasta_name in enumerate(self.fastas):
            this_fasta = self.fastas[fasta_name]
            score_lists=[]
            matched_traces=[]
            unused_traces=set([_ for _ in range(len(this_fasta.trace_matched_seqs))])
            for seq_ix in range(len(this_fasta.sequence)):
                matched_trace=[]
                score_list=[]
                for s,seqs in enumerate(this_fasta.trace_matched_seqs):
                    if seq_ix in seqs:
                        i = seq_ix-seqs[0]
                        partion= i/len(seqs)
                        score_list.append(np.sum(this_fasta.trace_scores[s])+2*partion*(1-partion))
                        matched_trace.append(s)
                        
                matched_traces.append(np.array(matched_trace)[np.argsort(score_list)[::-1]])
                score_lists.append(np.sum(score_list))
                
            if not matched_traces:
                continue
            max_seq_ix = np.argmax(score_lists)
            self.logger.info(f"Processing {len(matched_traces)} traces for sequence {fasta_name}")

            chain_list = list(this_fasta.chain_dict.keys())
            self.logger.info(f"Chain list: {chain_list}")
            model={}
            for id in matched_traces[max_seq_ix]:
                if len(model)<len(chain_list):
                    model[chain_list[len(model)]]=[id]
                    unused_traces.discard(id)
            models=[model]
            left_seq=max_seq_ix
            right_seq=max_seq_ix
            while True:
                tmp_models=[]
                for trace_id in copy.deepcopy(unused_traces):
                    seqs = this_fasta.trace_matched_seqs[trace_id]
                    traces = this_fasta.seq_matched_traces[trace_id]
                    if left_seq in seqs:
                        if len(models[0]) < len(chain_list):
                            models[0][chain_list[len(model)]]=[trace_id]
                            break
                        for model in models:
                            matched_chain_ids=set()
                            for chain_id in model:
                                for ti in model[chain_id]:
                                    chain = this_fasta.trace_matched_seqs[ti]
                                    if len(set(seqs)&set(chain))>4:
                                        matched_chain_ids.add(chain_id)
                            unmatched_chain_ids = set([_ for _ in chain_list])-matched_chain_ids
                            if not unmatched_chain_ids:
                                tmp_models.append(copy.deepcopy(model))
                            elif self.modeling_config.protocol == 'AF3_struct' or matched_chain_ids:
                                if self.modeling_config.protocol == 'AF3_struct':
                                    rmsd_mat=np.full((len(matched_chain_ids)+1,len(unmatched_chain_ids)),10000.0)
                                else:
                                    rmsd_mat=np.full((len(matched_chain_ids),len(unmatched_chain_ids)),10000.0)
                                occ_chain_lists=[]
                                for i, chain_i in enumerate(matched_chain_ids):
                                    chain = model[chain_i]
                                    occ_chain_list=[-1 for _ in range(len(this_fasta.sequence))]
                                    for id in chain:
                                        for s, seq_ix in enumerate(this_fasta.trace_matched_seqs[id]):
                                            occ_chain_list[seq_ix]=this_fasta.seq_matched_traces[id][s]
                                    occ_chain_lists.append(occ_chain_list)

                                val_chain_lists=[]
                                for i, chain_i in enumerate(unmatched_chain_ids):
                                    chain = model[chain_i]
                                    val_chain_list=[-1 for _ in range(len(this_fasta.sequence))]
                                    for id in chain:
                                        for s, seq_ix in enumerate(this_fasta.trace_matched_seqs[id]):
                                            val_chain_list[seq_ix]=this_fasta.seq_matched_traces[id][s]
                                    for s, seq_ix in enumerate(seqs):
                                        val_chain_list[seq_ix]=traces[s]
                                    val_chain_lists.append(val_chain_list)
                                
                                
                                for j, chain_j in enumerate(unmatched_chain_ids):
                                    for i, chain_i in enumerate(matched_chain_ids):
                                        val_coords=[]
                                        occ_coords=[]
                                        for s in range(len(this_fasta.sequence)):
                                            if occ_chain_lists[i][s]!=-1 and val_chain_lists[j][s]!=-1:
                                                occ_coords.append(self.CA_cands[occ_chain_lists[i][s]])
                                                val_coords.append(self.CA_cands[val_chain_lists[j][s]])
                                        rmsd_mat[i,j] = superpose3d.Superpose3D(val_coords,occ_coords)[0][0]
                                    if self.modeling_config.protocol == 'AF3_struct':
                                        occ_coords=[]
                                        val_coords=[]
                                        for s in range(len(this_fasta.sequence)):
                                            if val_chain_lists[j][s]!=-1:
                                                occ_coords.append(this_fasta.AF3_struct[s])
                                                val_coords.append(self.CA_cands[val_chain_lists[j][s]])
                                        rmsd_mat[-1,j] = superpose3d.Superpose3D(val_coords,occ_coords)[0][0]

                                min_i,min_j = np.unravel_index(np.argmin(rmsd_mat),rmsd_mat.shape)
                                tmp_model=copy.deepcopy(model)
                                tmp_model[list(unmatched_chain_ids)[min_j]] = [trace_id] + tmp_model[list(unmatched_chain_ids)[min_j]]
                                tmp_models.append(tmp_model)
                            else:
                                for chain_id in unmatched_chain_ids:
                                    chain = model[chain_id]
                                    tmp_model=copy.deepcopy(model)
                                    tmp_model[chain_id] = [trace_id] + tmp_model[chain_id]
                                    tmp_models.append(tmp_model)
                        unused_traces.discard(trace_id)
                        break
                        

                    if right_seq in seqs:
                        if len(models[0]) < len(chain_list):
                            models[0][chain_list[len(model)]]=[trace_id]
                            break
                        for model in models:
                            matched_chain_ids=set()
                            for chain_id in model:
                                for ti in model[chain_id]:
                                    chain = this_fasta.trace_matched_seqs[ti]
                                    if len(set(seqs)&set(chain))>4:
                                        matched_chain_ids.add(chain_id)
                            unmatched_chain_ids = set([_ for _ in chain_list])-matched_chain_ids
                            if not unmatched_chain_ids:
                                tmp_models.append(copy.deepcopy(model))
                            elif self.modeling_config.protocol == 'AF3_struct' or matched_chain_ids:
                                if self.modeling_config.protocol == 'AF3_struct':
                                    rmsd_mat=np.full((len(matched_chain_ids)+1,len(unmatched_chain_ids)),10000.0)
                                else:
                                    rmsd_mat=np.full((len(matched_chain_ids),len(unmatched_chain_ids)),10000.0)
                                occ_chain_lists=[]
                                for i, chain_i in enumerate(matched_chain_ids):
                                    chain = model[chain_i]
                                    occ_chain_list=[-1 for _ in range(len(this_fasta.sequence))]
                                    for id in chain:
                                        for s, seq_ix in enumerate(this_fasta.trace_matched_seqs[id]):
                                            occ_chain_list[seq_ix]=this_fasta.seq_matched_traces[id][s]
                                    occ_chain_lists.append(occ_chain_list)

                                val_chain_lists=[]
                                for i, chain_i in enumerate(unmatched_chain_ids):
                                    chain = model[chain_i]
                                    val_chain_list=[-1 for _ in range(len(this_fasta.sequence))]
                                    for id in chain:
                                        for s, seq_ix in enumerate(this_fasta.trace_matched_seqs[id]):
                                            val_chain_list[seq_ix]=this_fasta.seq_matched_traces[id][s]
                                    for s, seq_ix in enumerate(seqs):
                                        val_chain_list[seq_ix]=traces[s]
                                    val_chain_lists.append(val_chain_list)

                                for j, chain_j in enumerate(unmatched_chain_ids):
                                    for i, chain_i in enumerate(matched_chain_ids):
                                        val_coords=[]
                                        occ_coords=[]
                                        for s in range(len(this_fasta.sequence)):
                                            if occ_chain_lists[i][s]!=-1 and val_chain_lists[j][s]!=-1:
                                                occ_coords.append(self.CA_cands[occ_chain_lists[i][s]])
                                                val_coords.append(self.CA_cands[val_chain_lists[j][s]])
                                        rmsd_mat[i,j] = superpose3d.Superpose3D(val_coords,occ_coords)[0][0]
                                    if self.modeling_config.protocol == 'AF3_struct':
                                        occ_coords=[]
                                        val_coords=[]
                                        for s in range(len(this_fasta.sequence)):
                                            if val_chain_lists[j][s]!=-1:
                                                occ_coords.append(this_fasta.AF3_struct[s])
                                                val_coords.append(self.CA_cands[val_chain_lists[j][s]])
                                        rmsd_mat[-1,j] = superpose3d.Superpose3D(val_coords,occ_coords)[0][0]

                                min_i,min_j = np.unravel_index(np.argmin(rmsd_mat),rmsd_mat.shape)
                                tmp_model=copy.deepcopy(model)
                                tmp_model[list(unmatched_chain_ids)[min_j]] =  tmp_model[list(unmatched_chain_ids)[min_j]]+[trace_id]
                                tmp_models.append(tmp_model)
                            else:
                                for chain_id in unmatched_chain_ids:
                                    chain = model[chain_id]
                                    tmp_model=copy.deepcopy(model)
                                    tmp_model[chain_id] = tmp_model[chain_id]+[trace_id]
                                    tmp_models.append(tmp_model)

                        unused_traces.discard(trace_id)
                        break
                if tmp_models:
                    if len(tmp_models)>1000:
                        dis_list=[]
                        for model in tmp_models:
                            dis=[]
                            for chain_id in model:
                                for i,ti in enumerate(model[chain_id][:-1]):
                                    cand1=this_fasta.seq_matched_traces[ti][-1]
                                    cand2=this_fasta.seq_matched_traces[model[chain_id][i+1]][0]
                                    seq1=this_fasta.trace_matched_seqs[ti][-1]
                                    seq2=this_fasta.trace_matched_seqs[model[chain_id][i+1]][0]
                                    sp_dis= self.cand_self_dis[cand1,cand2]
                                    seq_dis=abs(seq2-seq1)
                                    dis.append(np.sqrt(seq_dis)+sp_dis+sp_dis/(seq_dis+1))
                            dis_list.append(np.mean(dis))
                        sort_ix=np.argsort(dis_list)
                        models=[]
                        for i in range(10):
                            models.append(tmp_models[sort_ix[i]])
                    else:
                        models=tmp_models
                elif left_seq>-1 or right_seq<len(this_fasta.sequence):
                    if left_seq>-1:
                        left_seq-=1
                    if right_seq<len(this_fasta.sequence):
                        right_seq+=1
                else:
                    break

            dis_list=[]
            for model in models:
                dis=[]
                for chain_id in model:
                    for i,ti in enumerate(model[chain_id][:-1]):
                        cand1=this_fasta.seq_matched_traces[ti][-1]
                        cand2=this_fasta.seq_matched_traces[model[chain_id][i+1]][0]
                        seq1=this_fasta.trace_matched_seqs[ti][-1]
                        seq2=this_fasta.trace_matched_seqs[model[chain_id][i+1]][0]
                        sp_dis= self.cand_self_dis[cand1,cand2]
                        seq_dis=abs(seq2-seq1)
                        dis.append(np.sqrt(seq_dis)+sp_dis+sp_dis/(seq_dis+1))
                dis_list.append(np.mean(dis))
            min_ix=np.argmin(dis_list)
            model = models[min_ix]

            for j,chain_id in enumerate(model):
                chain = model[chain_id]
                score_list=[]
                for ix in chain:
                    score_list.append(np.sum(this_fasta.trace_scores[ix]))
                arg_score_ix = np.argsort(score_list)
                for i in range(len(chain)):
                    ix = chain[arg_score_ix[i]]
                    for c, cand in enumerate(this_fasta.seq_matched_traces[ix][3:-3]):
                        p = this_fasta.trace_matched_seqs[ix][3:-3][c]
                        this_fasta.chain_dict[chain_id].result[p] = cand
                for i in range(len(chain)):
                    for cand in this_fasta.chain_dict[chain_id].result:
                        if cand !=-1:
                            self.used_cands.add(cand)
        
        # Save initial model
        init_model_path = os.path.join(
            self.modeling_config.output_path, 
            f'{self.map_id}_{self.pdb_id}_{self.method_name}(init)_ca_model.pdb'
        )
        self.logger.info(f"Saving initial model: {init_model_path}")
        
        with open(init_model_path, 'w') as w:
            j = 0
            atom_ix=0
            for fasta_ix, fasta_name in enumerate(self.fastas):
                this_fasta = self.fastas[fasta_name]
                for chain_id in this_fasta.chain_dict:
                    for seq_id, cand in enumerate(this_fasta.chain_dict[chain_id].result):
                        if cand!=-1:
                            atom_ix += 1
                            w.write('ATOM')
                            w.write('{:>{}}'.format(atom_ix, 7))
                            w.write('{:>{}}'.format('CA', 4))
                            w.write('{:>{}}'.format(abb2AA[this_fasta.sequence[seq_id]], 5))
                            w.write('{:>{}}'.format(chain_id, 2))
                            w.write('{:>{}}'.format(seq_id, 4))
                            w.write('{:>{}}'.format('%.3f' %
                                    (self.CA_cands[cand][0]+self.offset[0]), 12))
                            w.write('{:>{}}'.format('%.3f' %
                                    (self.CA_cands[cand][1]+self.offset[1]), 8))
                            w.write('{:>{}}'.format('%.3f' %
                                    (self.CA_cands[cand][2]+self.offset[2]), 8))
                            w.write('  1.00                 C\n')

    def gapFilling(self):
        """Perform gap filling to fill missing regions."""
        self.logger.info("Starting gap filling process...")
        atom_ix=0
        for fasta_ix, fasta_name in enumerate(self.fastas):
            this_fasta = self.fastas[fasta_name]

            # Prepare chain candidate scoring matrix
            chain_cand_score=np.zeros([len(this_fasta.chain_dict),self.seq_cand_AA_mat.shape[1],self.seq_cand_AA_mat.shape[2]])
            chain_list=list(this_fasta.chain_dict.keys())
            for i in range(chain_cand_score.shape[0]):
                chain_id = chain_list[i]
                
                this_fasta.chain_dict[chain_id].highConfResult = copy.copy(this_fasta.chain_dict[chain_id].result)
                for c in range(chain_cand_score.shape[2]):
                    if c not in self.used_cands:
                        chain_cand_score[i,:,c]=self.seq_cand_AA_mat[fasta_ix,:,c]
                for p,cand in enumerate(this_fasta.chain_dict[chain_id].result):
                    if cand!=-1:
                        chain_cand_score[i,p,:]=0
                        chain_cand_score[:,:,cand]=0#new
                        chain_cand_score[i,p,cand]=1

            # Build enhanced candidate matrix with n-hop information
            chain_cand_mat=chain_cand_score.copy()
            for i in range(self.modeling_config.n_hop):
                chain_cand_mat+=np.pad(chain_cand_score[:,:-i-1,:],[(0,0),(i+1,0),(0,0)],constant_values=(0,0))@self.n_hop_mat[i].T+np.pad(chain_cand_score[:,i+1:,:],[(0,0),(0,i+1),(0,0)],constant_values=(0,0))@self.n_hop_mat[i].T
            for c in self.used_cands:
                chain_cand_mat[:,:,c]=0#new
            this_fasta.chain_cand_mat=chain_cand_mat

            # Identify gaps to fill
            start_ends=[]
            for i, chain_id in enumerate(this_fasta.chain_dict):
                start_end=[]
                init_model=this_fasta.chain_dict[chain_id].result
                pair=[]
                for t, cand in enumerate(init_model):
                    if cand == -1:
                        if not pair:
                            pair=[t-1]
                    else:
                        if pair:
                            pair.append(t)
                            start_end.append([pair[0],pair[1]])
                            start_ends.append([i,set(range(pair[0]+1,pair[1])),pair[0],pair[1]])
                            pair=[]
                if pair:
                    pair.append(len(init_model))
                    start_end.append([pair[0],pair[1]])
                    start_ends.append([i,set(range(pair[0]+1,pair[1])),pair[0],pair[1]])

            # Sort gaps by complexity
            unmat_len_list=[]
            for start_end1 in start_ends:
                unmat_len=0
                set1=start_end1[1]
                for start_end2 in start_ends:
                    unmat_len+=len(set1&start_end2[1])
                unmat_len_list.append(unmat_len)
            sort_ix = np.argsort(unmat_len_list)
            
            # Fill gaps in order of complexity
            for e,ix in enumerate(sort_ix):
                self.logger.info(f"Processing gap {e+1}/{len(unmat_len_list)}: positions {start_ends[ix][2]}->{start_ends[ix][3]}")

                self.fillGap(fasta_ix,start_ends[ix])

        # Clean up overlapping candidates
        cand_occ={}
        centroids={}
        for fasta_ix, fasta_name in enumerate(self.fastas):
            this_fasta= self.fastas[fasta_name]
            for chain_id in this_fasta.chain_dict:
                coord_list=[]
                for seq_id, cand in enumerate(this_fasta.chain_dict[chain_id].highConfResult):
                    if cand!=-1:
                        coord_list.append(self.CA_cands[cand])
                if coord_list:
                    centroids[(fasta_name,chain_id)] = np.array(coord_list).mean(axis=0)
                for seq_id, cand in enumerate(this_fasta.chain_dict[chain_id].result):
                    if cand!=-1:
                        if cand not in cand_occ:
                            cand_occ[cand]=[]
                        cand_occ[cand].append([fasta_name,chain_id,seq_id])
        
        # Resolve candidate conflicts
        for cand in cand_occ:
            min_dis=10000
            for fasta_name,chain_id,seq_id in cand_occ[cand]:
                dis2=np.sum((centroids[(fasta_name,chain_id)]-self.CA_cands[cand])**2)
                min_dis=min(min_dis, dis2)
            
            for fasta_name,chain_id,seq_id in cand_occ[cand]:
                this_fasta= self.fastas[fasta_name]
                dis2=np.sum((centroids[(fasta_name,chain_id)]-self.CA_cands[cand])**2)
                if dis2>min_dis+1:
                    seq_len=len(this_fasta.sequence)
                    for s in range(max(0,seq_id-2),min(seq_len,seq_id+3)):
                        if this_fasta.chain_dict[chain_id].highConfResult[s] != -1:
                            continue
                        this_fasta.chain_dict[chain_id].result[s]=-1

        # Save final model
        cand_set=set()
        self.logger.info(f"Saving gap filled model: {self.ca_model_pdb}")
        with open(self.ca_model_pdb, 'w') as w:
            atom_ix=0
            for fasta_ix, fasta_name in enumerate(self.fastas):
                this_fasta= self.fastas[fasta_name]
                for chain_id in this_fasta.chain_dict:
                    for seq_id, cand in enumerate(this_fasta.chain_dict[chain_id].result):
                        if cand!=-1 and cand not in cand_set:
                            atom_ix += 1
                            w.write('ATOM')
                            w.write('{:>{}}'.format(atom_ix, 7))
                            w.write('{:>{}}'.format('CA', 4))
                            w.write('{:>{}}'.format(abb2AA[this_fasta.sequence[seq_id]], 5))
                            w.write('{:>{}}'.format(chain_id, 2))
                            w.write('{:>{}}'.format(seq_id+1, 4))
                            w.write('{:>{}}'.format('%.3f' %
                                    (self.CA_cands[cand][0]+self.offset[0]), 12))
                            w.write('{:>{}}'.format('%.3f' %
                                    (self.CA_cands[cand][1]+self.offset[1]), 8))
                            w.write('{:>{}}'.format('%.3f' %
                                    (self.CA_cands[cand][2]+self.offset[2]), 8))
                            w.write('  1.00                 C\n')
                            cand_set.add(cand)

    def fillGap(self,fasta_ix,start_end):
        """Fill a gap in the protein chain."""
        fasta_name=self.fasta_list[fasta_ix]
        this_fasta = self.fastas[fasta_name]
        seq_len = len(this_fasta.sequence)
        chain_list=list(this_fasta.chain_dict.keys())
        this_chain_id=chain_list[start_end[0]]
        left_pos = start_end[2]
        right_pos = start_end[3]
        final_seq=range(start_end[2],start_end[3]+1)
        left_val=True
        right_val=True
        dir = 1
        
        # Handle boundary cases
        if left_pos==-1 and right_pos==seq_len:
            return
        elif left_pos==-1:
            left_traces = []
            right_traces = [[this_fasta.chain_dict[this_chain_id].result[right_pos]]]
            left_infos = []
            right_infos=[[[],[],0]]
            left_val=False
            left_seq = []
            right_seq=  [right_pos]
            dir = -1
        elif right_pos==seq_len:
            left_traces = [[this_fasta.chain_dict[this_chain_id].result[left_pos]]]
            right_traces = []
            left_infos = [[[],[],0]]
            right_infos=[]
            left_seq = [left_pos]
            right_seq=  []
            right_val=False
        else:
            left_traces = [[this_fasta.chain_dict[this_chain_id].result[left_pos]]]
            right_traces = [[this_fasta.chain_dict[this_chain_id].result[right_pos]]]
            left_infos = [[[],[],0]]
            right_infos=[[[],[],0]]
            left_seq = [left_pos]
            right_seq=  [right_pos]
        
        # Iteratively build traces from both ends
        while (left_val or right_val) and left_pos!=right_pos and left_pos<len(this_fasta.sequence)-1 and right_pos>0:
            if dir == 1:
                this_traces = left_traces
                this_infos= left_infos
                left_pos+=dir
                end=-1
                this_seq = left_seq+[left_pos]
                this_pos = left_pos
            else:
                this_traces = right_traces
                this_infos= right_infos
                right_pos+=dir
                end=0
                this_seq = [right_pos]+right_seq
                this_pos = right_pos

            # Set up structure matching if using AF3
            matched_chain=[[],[]]
            if self.modeling_config.protocol=='AF3_struct':
                matched_chain=[range(len(this_seq)),this_fasta.AF3_struct[this_seq]]
            else:
                max_len=5
                for chain_id in this_fasta.chain_dict:
                    matched_pos=[]
                    matched_coords=[]
                    for p,pos in enumerate(this_seq):
                        if this_fasta.chain_dict[chain_id].result[pos]!= -1:
                            matched_pos.append(p)
                            cand= this_fasta.chain_dict[chain_id].result[pos]
                            matched_coords.append(self.CA_cands[cand])
                    if len(matched_pos)>max_len:
                        matched_chain=[matched_pos,matched_coords]
                        max_len=len(matched_pos)

            # Generate candidate extensions
            tmp_traces=[]
            tmp_infos=[]
            tmp_scores=[]
            for ix, trace in enumerate(this_traces):
                if len(trace)-len(set(trace))>max(5,len(trace)//10):
                    continue
                this_info = this_infos[ix]
                cand = trace[-1] if dir == 1 else trace[0]
                nei_list=list(set(self.neighbors2to6[cand])-self.used_cands-set(trace))
                for neigh in nei_list:
                    new_trace = trace+[neigh] if dir == 1 else [neigh]+trace
                    
                    cand_score=this_info[0] + [this_fasta.chain_cand_mat[start_end[0], this_pos, neigh]]
                    neigh_score=this_info[1]+[self.neigh_mat[cand,neigh]]
                    sym_score = this_info[2]
                    if len(this_seq) >3 and len(this_seq)-1 in matched_chain[0]:
                        this_coords=[]
                        for p in matched_chain[0]:
                            this_coords.append(self.CA_cands[new_trace[p]])
                        sym_score = max(0,superpose3d.Superpose3D(this_coords,matched_chain[1])[0][0]-1)/2
                    
                    score = np.mean(np.array(cand_score)+np.array(neigh_score))- sym_score
                    tmp_traces.append(new_trace)
                    tmp_infos.append([cand_score, neigh_score, sym_score])
                    tmp_scores.append(score)

            # Handle trace continuation or termination        
            if not tmp_traces:
                if dir ==1:
                    left_val=False
                    dir*=-1
                    continue
                else:
                    right_val=False
                    dir*=-1
                    continue
            
            elif len(tmp_traces)>1000 or right_pos-left_pos<=2:
                this_traces=[]
                this_infos=[]
                last_dict={}
                max_score=-np.inf
                max_last=None
                for ix, trace in enumerate(tmp_traces):
                    if trace[end] not in last_dict or tmp_scores[ix] > last_dict[trace[end]][1]:
                        last_dict[trace[end]] = [trace, tmp_scores[ix],tmp_infos[ix]]
                        if tmp_scores[ix] > max_score:
                            max_score=tmp_scores[ix]
                            max_last=trace[end]

                for last in last_dict:
                    if self.cand_self_dis[last,max_last]<20:
                        this_traces.append(last_dict[last][0])
                        this_infos.append(last_dict[last][2])

                if dir == 1:
                    left_seq = left_seq+[left_pos]
                else:
                    right_seq=[right_pos]+right_seq
            else:
                if dir == 1:
                    left_seq = left_seq+[left_pos]
                else:
                    right_seq=[right_pos]+right_seq
                this_traces = tmp_traces
                this_infos = tmp_infos

            if dir ==1:
                left_traces= this_traces
                left_infos = this_infos
                
            else:
                right_traces= this_traces
                right_infos=this_infos

            if left_val and right_val:
                dir*=-1

        # Connect traces and assign final results
        max_trace=None
        max_score=-np.inf
        
        if left_traces and right_traces and len(left_traces[0])+len(right_traces[0])-1==len(final_seq):    
            for il, left_trace in enumerate(left_traces):
                for ir, right_trace in enumerate(right_traces):
                    if left_trace[-1]==right_trace[0]:
                        left_score=np.mean(np.array(left_infos[il][0])+np.array(left_infos[il][1]))- left_infos[il][2]
                        right_score=np.mean(np.array(right_infos[ir][0])+np.array(right_infos[ir][1]))- right_infos[ir][2]
                        if left_score+right_score > max_score:
                            max_trace = left_trace+right_trace[1:]
                            max_score = left_score+right_score

            if max_trace!= None:
                used_cands=set()
                for p in range(len(final_seq)//2+1):
                    left_pos=list(final_seq)[p]
                    right_pos=list(final_seq)[-p-1]
                    if max_trace[p] not in used_cands:
                        used_cands.add(max_trace[p])
                        this_fasta.chain_dict[this_chain_id].result[left_pos]=max_trace[p]
                    if max_trace[-p-1] not in used_cands:
                        used_cands.add(max_trace[-p-1])
                        this_fasta.chain_dict[this_chain_id].result[right_pos]=max_trace[-p-1]
        
        # Handle case where traces don't connect
        if max_trace is None:            
            max_left_trace=None
            max_left_score=-np.inf
            for il, left_trace in enumerate(left_traces):
                left_score = np.mean(np.array(left_infos[il][0])+np.array(left_infos[il][1]))- left_infos[il][2]
                if left_score > max_left_score:
                    max_left_trace = left_trace
                    max_left_score = left_score
                
            max_right_trace=None
            max_right_score=-np.inf
            for ir, right_trace in enumerate(right_traces):
                right_score = np.mean(np.array(right_infos[ir][0])+np.array(right_infos[ir][1]))- right_infos[ir][2]
                if right_score > max_right_score:
                    max_right_trace = right_trace
                    max_right_score = right_score

            gap = 0
            if max_left_trace is not None and max_right_trace is not None:
                gap = max(0,(self.cand_self_dis[max_left_trace[-1],max_right_trace[0]] - 3 * (right_pos-left_pos))) // 6
            
            if max_left_trace is not None:
                for p in range(len(left_seq)-int(gap)):
                    left_pos=list(left_seq)[p]
                    this_fasta.chain_dict[this_chain_id].result[left_pos]=max_left_trace[p]
                
            if max_right_trace is not None:
                for p in range(int(gap),len(right_seq)):
                    right_pos=list(right_seq)[p]
                    this_fasta.chain_dict[this_chain_id].result[right_pos]=max_right_trace[p]
    
    def phenix_refine(self, all_atom_model):
        """Run Phenix refinement on the all-atom model."""
        phenix_param = os.path.abspath(self.modeling_config.phenix_param)
        phenix_act = os.path.abspath(self.modeling_config.phenix_act)
        output_dir = os.path.abspath(self.modeling_config.output_path)
        cryoEM_map = os.path.abspath(self.modeling_config.map_path)
        all_atom_model = os.path.abspath(all_atom_model)
        
        cmd = f'bash modules/phenix.sh {phenix_act} {output_dir} "phenix.real_space_refine {all_atom_model} {cryoEM_map} {phenix_param} resolution={self.resol}"'
        self.logger.info(f"Executing Phenix command: {cmd}")
        run([cmd], stdout=DEVNULL, stderr=DEVNULL, shell=True)

    def time_record(self):
        """Record timing information to CSV file."""
        self.logger.info(f"Recording timing information: {self.time_log}")
        with open(self.time_log, 'w') as w:
            w.write('step,time\n')
            for key in self.time_cost:
                w.write(f'{key},{round(self.time_cost[key])}\n')
    
    def checkSeq(self):
        """Check and parse input FASTA sequences."""
        if os.path.exists(self.modeling_config.fasta_path):
            fasta_lines = open(self.modeling_config.fasta_path).readlines()
        else:
            self.logger.error("FASTA file not found!")
            return 'fasta not found!'
        
        un_exist_list=[]
        if self.modeling_config.protocol=='AF3_struct':
            self.logger.info('Using MICA with AF3 structures')
            
        input_list=[]
        fasta_set=set()
        seq = ''
        
        self.logger.info("Parsing FASTA file...")
        for line_n, line in enumerate(fasta_lines):
            line=line.strip()
            if line.startswith('>'):
                head = line
                split_fasta=line[1:].split('|')[0]
                fasta_name=split_fasta
                n=0
                while fasta_name in fasta_set:
                    n+=1
                    fasta_name = f'{split_fasta}_{n}'
                fasta_set.add(fasta_name)
                seq=''
            else:
                seq=seq+line
            
            if line_n >=len(fasta_lines)-1 or fasta_lines[line_n+1].startswith('>'):
                if len(line)<10:
                    continue
                for i,c in enumerate(seq):
                    if c not in AA_abb and c not in ['A','U','T','G','C']:
                        seq=seq[:i]+'A'+seq[i+1:]
                        self.logger.warning(f'Non-standard residue {c} in protein sequence changed to ALA')
                if ('U' in seq) or set(seq).issubset(set(['A','U','T','G','C'])):
                    continue
                seq_obj=Sequence(fasta_name, seq)
                try:
                    chain_strs = head.split('|')[1].split(',')
                except:
                    chain_strs=[random.choice(chainID_list)]
                    self.logger.warning(f'Parse chain number error! Chain number set as 1 for "{fasta_name}"')
                    self.logger.warning(f'Assigning chain id {chain_strs[0]} for "{fasta_name}"')
                input_list.append((seq_obj,chain_strs,self.modeling_config.protocol,self.modeling_config.AF3_structure_path,self.pdbParser))
            
        if input_list:
            self.logger.info(f"Processing {len(input_list)} sequences with multiprocessing...")
            pool = Pool(min(len(input_list),self.modeling_config.mul_proc_num))
            results = pool.map(get_seq, input_list)

            for res in results:
                seq_obj,chain_strs,un_exist=res
                fasta_name=seq_obj.fasta_name
                if un_exist:
                    un_exist_list.append(fasta_name)

                for chain_str in chain_strs:
                    try:
                        chain_id = chain_str.split(' ')[-1].split(']')[0]
                    except:
                        chain_id='A'
                        self.logger.warning(f'Parse chain ID error! Chain ID set as A for {chain_str}')
                    
                    if fasta_name not in self.fastas:
                        self.fasta_list.append(fasta_name)
                        self.fastas[fasta_name] = seq_obj
                    
                    new_chain_id=chain_id
                    if new_chain_id not in chainID_list:
                        new_chain_id = random.choice(chainID_list)
                    
                    chain_n=0
                    while new_chain_id in self.fastas[fasta_name].chain_dict and chain_n<100:
                        chain_n+=1
                        new_chain_id = random.choice(chainID_list)
                    
                    if chain_id != new_chain_id:
                        self.logger.warning(f'Wrong chain id! Using random chain id {new_chain_id}!')
                    
                    chain_id=new_chain_id
                    self.fastas[fasta_name].chain_dict[chain_id]=Chain(chain_id, seq_obj.sequence)
                    self.chain_id_list.append(chain_id)
                    self.max_seq_len=max(self.max_seq_len,len(seq_obj.sequence))
                    self.ResNum +=len(seq_obj.sequence)


        if len(self.fastas) == 0:
            self.logger.error('Error in parse fasta, terminated!')
            return 'Error in parse fasta, terminated!'
            
        if un_exist_list:
            error_msg = f'Structures not found for {un_exist_list}, Check your directory of AF3 structures!'
            self.logger.error(error_msg)
            return error_msg
        
        self.logger.info('FASTA parsing completed successfully:')
        for i, fasta_name in enumerate(self.fastas):
            chain_ids = list(self.fastas[fasta_name].chain_dict.keys())
            self.logger.info(f'   Sequence {i+1}: {fasta_name}')
            self.logger.info(f'   Chains: {" ".join(chain_ids)}')
            self.logger.info(f'   Sequence: {self.fastas[fasta_name].sequence}')
        return 'success'

    


    
