# Multimodal deep learning integration of cryo-EM and AlphaFold3 for high-accuracy protein structure determination 

Cryo-electron microscopy (cryo-EM) is a key technology for determining the structures of proteins, particularly large protein complexes. However, automatically building high-accuracy protein structures from cryo-EM density maps remains a crucial challenge. In this work, we introduce MICA, a fully automatic and multimodal deep learning approach combining cryo-EM density maps with AlphaFold3-predicted structures at both input and output levels to improve cryo-EM protein structure modeling. It first uses a multi-task encoder-decoder architecture with a feature pyramid network to predict backbone atoms, Cα atoms and amino acid types from both cryo-EM maps and AlphaFold3-predicted structures, which are used to build an initial backbone model. This model is further refined using AlphaFold3-predicted structures and density maps to build final atomic structures. MICA significantly outperforms other state-of-the-art deep learning methods in terms of both modeling accuracy and completeness and is robust to protein size and map resolution. Additionally, it builds high-accuracy structural models with an average template-based modeling score (TM-score) of 0.93 from recently released high-resolution cryo-EM density maps, showing it can be used for real-world, automated, accurate protein structure determination.

## Overview
Figure below demonstrates the overview of atomic modeling process used by MICA.

![Alt text](<assets/General_Overview.jpg>)

## 🚀 Quick Start

### 1. Clone the Repository
```
git clone https://github.com/jianlin-cheng/MICA
cd MICA
```

### 2. Set Up Conda Environment
Create and activate the conda environment using the provided YAML file:

```
conda env create -f environment.yml
conda activate MICA
```

### 3. Download MICA model
```
curl https://zenodo.org/api/records/15756654/files/trained_models.tar.gz?download=1 --output trained_models.tar.gz
tar -xzvf trained_models.tar.gz
rm trained_models.tar.gz
```

### 4. Download Sample Data for inference
```
curl https://zenodo.org/api/records/15756654/files/input.tar.gz?download=1 --output input.tar.gz
tar -xzvf input.tar.gz
rm input.tar.gz
```

### 5. Inference on sample data 
Run inference on sample data to make sure the installation has been done correctly.
```
python run.py -m input/15635/emd_15635.map -f input/15635/8at6.fasta -i input/15635 --run_pulchra --pulchra_path=modules/pulchra304/src/pulchra --resolution=3.7
```

## 📂 Running on New Dataset

### 1: Install and configure Phenix (Skip this step if you already have Phenix on your machine)

1. Visit the [PHENIX download website](https://phenix-online.org/download)
2. Click on **Request a password** using your institutional email
3. Once you get *username* and *password* go to **Download official release**
4. Download the command-line installer for your machine
5. Set up Phenix
6. Verify the Phenix installation and grab path to phenix_env.sh

**For complete instructions on installing and setting up Phenix visit [Phenix website](https://phenix-online.org/documentation/install-setup-run.html)

### 2. Inference on New Dataset
#### Prerequisites
- FASTA sequence file (e.g., `8at6.fasta`)
- Cryo-EM density map (e.g., `emd_15635.map`)
- PHENIX installed

#### Directory Structure
Your directory structure should be something like this; initially containing 8at6.fasta and emd_15635.map:
```
MICA/
└── input/
    └── 15635/
        ├── AF3_chains/
        ├── AF3_docked_models/
        ├── AF3_domains/
        ├── AF3_JSON/
        │   ├── 8AT6_1.json
        │   ├── 8AT6_2.json
        │   └── 8AT6_3.json
        ├── AF3_PDBs/
        ├── AF3_results/
        │   ├── 8at6_1/
        │   ├── 8at6_2/
        │   └── 8at6_3/
        ├── AF3_structures/
        ├── 8at6.fasta
        ├── 15635_af3_docked.pdb
        └── emd_15635.map
```

Run the following commands sequentially inside *MICA location*.

#### 2.1 Generate AlphaFold3 JSON Files
**Required Format:**
```bash
python utils/fasta_to_AF3_json.py -f <path/to/fasta/file> -n <protein_name>
```

**Example:**
```bash
python utils/fasta_to_AF3_json.py -f input/15635/8at6.fasta -n 15635
```

- Upload generated JSON files to [AlphaFold3 server](https://alphafoldserver.com)
- Download results and place in:
  - `input/15635/AF3_results/8at6_1/*model_0.cif`
  - `input/15635/AF3_results/8at6_2/*model_0.cif`
  - `input/15635/AF3_results/8at6_3/*model_0.cif`

#### 2.2 Process AlphaFold3 results
**Required Format:**
```bash
python utils/process_AF3_results.py -f <path/to/fasta/file> -a <path/to/AF3_results>
```

**Example:**
```bash
python utils/process_AF3_results.py -f input/15635/8at6.fasta -a input/15635/AF3_results
```

#### 2.3 Get Map Parameters if Cryo-EM map is avaiable in EMDB website (Optional)

**Required Format:**
```bash
python utils/emdb_extractor.py --emdb_id <EMDB_ID>
```

**Example:**
```bash
python utils/emdb_extractor.py --emdb_id 15635
```

#### 2.4 Dock domains into cryo-EM map
**Required Format:**
```bash
python utils/dock_in_map.py \
    -m <path/to/cryo-EM/map> \
    -c <contour_level> \
    -r <resolution> \
    -f <path/to/fasta/file> \
    -a <path/to/AF3_results> \
    --phenix_act <path/to/phenix/activation>
```

**Example:**
```bash
python utils/dock_in_map.py \
    -m input/15635/emd_15635.map \
    -c 0.0242 \
    -r 3.7 \
    -f input/15635/8at6.fasta \
    -a input/15635/AF3_results \
    --phenix_act ../phenix/phenix-1.20.1-4487/phenix_env.sh
```

#### 2.4 Run Data preprocessing, deep learning prediction and atomic model building
**Required Format:**
```bash
python run.py \
    -m <path/to/cryo-EM/map> \
    -f <path/to/fasta/file> \
    -i <protein/dataset_identifier> \
    --run_pulchra \
    --pulchra_path=<path/to/pulchra> \
    --run_phenix \
    --phenix_act=<path/to/phenix/activation> \
    --resolution=<resolution>
```

**Example:**
```bash
python run.py \
    -m input/15635/emd_15635.map \
    -f input/15635/8at6.fasta \
    -i input/15635 \
    --run_pulchra \
    --pulchra_path=modules/pulchra304/src/pulchra \
    --run_phenix \
    --phenix_act=../phenix/phenix-1.20.1-4487/phenix_env.sh \
    --resolution=3.7
```

### 3. Results
Final atomic model will be saved in: `output/15635_8at6_MICA_all_atom_model.pdb`

