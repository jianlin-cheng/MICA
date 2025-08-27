# Multimodal deep learning integration of cryo-EM and AlphaFold3 for high-accuracy protein structure determination 

Cryo-electron microscopy (cryo-EM) is a key technology for determining the structures of proteins, particularly large protein complexes. However, automatically building high-accuracy protein structures from cryo-EM density maps remains a crucial challenge. In this work, we introduce MICA, a fully automatic and multimodal deep learning approach combining cryo-EM density maps with AlphaFold3-predicted structures at both input and output levels to improve cryo-EM protein structure modeling. It first uses a multi-task encoder-decoder architecture with a feature pyramid network to predict backbone atoms, Cα atoms and amino acid types from both cryo-EM maps and AlphaFold3-predicted structures, which are used to build an initial backbone model. This model is further refined using AlphaFold3-predicted structures and density maps to build final atomic structures. MICA significantly outperforms other state-of-the-art deep learning methods in terms of both modeling accuracy and completeness and is robust to protein size and map resolution. Additionally, it builds high-accuracy structural models with an average template-based modeling score (TM-score) of 0.93 from recently released high-resolution cryo-EM density maps, showing it can be used for real-world, automated, accurate protein structure determination.

## 🔍 Overview
Figure below demonstrates the overview of atomic modeling process used by MICA.

![Alt text](<assets/General_Overview.jpg>)

<details>
<summary><h2>🚀 Quick Start</h2></summary>

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

### 3. Download MICA Model
```
curl https://zenodo.org/records/15756654/files/trained_models.tar.gz?download=1 --output trained_models.tar.gz
tar -xzvf trained_models.tar.gz
rm trained_models.tar.gz
```

### 4. Download Sample Data for Inference
```
curl https://zenodo.org/records/15756654/files/input.tar.gz?download=1 --output input.tar.gz
tar -xzvf input.tar.gz
rm input.tar.gz
```

### 5. Inference on Sample Data 
Run inference on sample data to make sure the installation has been done correctly.
```
python run.py -m input/15635/emd_15635.map -f input/15635/8at6.fasta -i input/15635 --run_pulchra --pulchra_path=modules/pulchra304/src/pulchra --resolution=3.7
```

</details>

<details>
<summary><h2>📂 Running on New Dataset</h2></summary>

### 🔧 Step 1: Install and Configure PHENIX (Skip this step if you already have Phenix on your machine)

1. Visit the [PHENIX download website](https://phenix-online.org/download)
2. Click on **Request a password** using your institutional email
3. Once you get *username* and *password* go to **Download official release**
4. Download the command-line installer for your machine
5. Set up Phenix
6. Verify the Phenix installation and grab path to phenix_env.sh

**For complete instructions on installing and setting up Phenix visit [PHENIX website](https://phenix-online.org/documentation/install-setup-run.html)**

### 🔮 Step 2: Inference on New Dataset
#### 📋 Prerequisites
- FASTA sequence file (e.g., `8at6.fasta`)
- Cryo-EM density map (e.g., `emd_15635.map`)
- PHENIX installed

#### 📁 Directory Structure
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

### 2.1 Generate AlphaFold3 JSON files
**Required Format:**
```bash
python utils/fasta_to_AF3_json.py -f <path/to/fasta/file> -n <protein_name or Map ID>
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

### 2.2 Get map parameters if Cryo-EM map is available in EMDB website (Optional)

**Required Format:**
```bash
python utils/emdb_extractor.py --emdb_id <EMDB_ID>
```

**Example:**
```bash
python utils/emdb_extractor.py --emdb_id 15635
```

**Output:**
This script extracts and returns the contour level and resolution parameters from the EMDB database, which are required for the docking process in subsequent steps.

**Returns:**
- `contour_level`: The recommended contour level for docking
- `resolution`: The map resolution in Angstroms

---

### 🚀 Processing Options

> **Choose ONE of the following approaches:**
> - **Option A:** Complete Pipeline (Step 2.3) - **Recommended for most users**
> - **Option B:** Step-by-Step Pipeline(Steps 2.4-2.6) - **Helpful for step-by-step results or debugging**

<details>
<summary><h3>🎯 Option A: Complete Pipeline (Recommended)</h3></summary>

### 2.3 MICA atomic model building pipeline 
This MICA pipeline is a comprehensive bash script that automates the execution of three essential protein processing programs in sequence. This pipeline streamlines the workflow from AlphaFold 3 results processing through cryo-EM map docking to final atomic model building.

**Pipeline Steps:**
The script executes three programs in the following order:
1. **process_AF3_results.py** - Process AlphaFold3 (AF3) results and divides into domains using Merizo
2. **dock_in_map.py** - Dock AF3 domain structures into cryo-EM map 
3. **run.py** - Run data preprocessing, deep learning prediction and atomic model building

**Required Format:**
```bash
./protein_pipeline.sh [OPTIONS]
```

**Interactive Mode:**
```bash
./protein_pipeline.sh
```
*Prompts for all required inputs*

**Example:**
```bash
./protein_pipeline.sh \
    -f input/15635/8at6.fasta \
    -a input/15635/AF3_results \
    -m input/15635/emd_15635.map \
    -c 0.0242 \
    -r 3.7 \
    -p modules/pulchra304/src/pulchra \
    -x ../phenix/phenix-1.20.1-4487/phenix_env.sh
```

**Command Line Options:**

| Option | Long Form | Type | Description |
|--------|-----------|------|-------------|
| `-f` | `--fasta_path` | `PATH` | Path to FASTA file **(required)** |
| `-a` | `--AF3_results_path` | `DIR` | Path to AF3 results directory **(required)** |
| `-m` | `--map_path` | `PATH` | Path to cryo-EM map file **(required)** |
| `-c` | `--contour_level` | `VALUE` | Contour level for docking **(required)** |
| `-r` | `--resolution` | `VALUE` | Resolution value **(required)** |
| `-p` | `--pulchra_path` | `PATH` | Path to Pulchra executable **(required)** |
| `-x` | `--phenix_act` | `PATH` | Path to Phenix activation script **(required)** |
| `-d` | `--device` | `DEVICE` | Device for running code (`cpu`, `cuda`, `cuda:1`, etc) **(optional)** |
| `-h` | `--help` | - | Show help message |

**Output:**
The pipeline automatically generates detailed execution timing logs and atomic models.

**Returns:**
##### A. CSV Timing Log
- ***Format:*** `{fasta_name}_execution_times_YYYYMMDD_HHMMSS.csv`
- ***Example:*** `8at6_execution_times_20240826_143022.csv`

##### B. Final Atomic Model
- ***Format:*** `output/{identifier}_{fasta_name}_MICA_all_atom_model.pdb`
- ***Example:*** `output/15635_8at6_MICA_all_atom_model.pdb`

> ✅ **If you used Option A (Step 2.3), you're done! Skip to [Step 3: Results] section.**

</details>

<details>
<summary><h3>🔧 Option B: Step-by-Step Pipeline</h3></summary>

> ⚠️ **Note:** Only use this option if you did NOT run Step 2.3 or need to debug individual steps.

### 2.4 Process AlphaFold3 results
**Required Format:**
```bash
python utils/process_AF3_results.py -f <path/to/fasta/file> -a <path/to/AF3_results>
```

**Example:**
```bash
python utils/process_AF3_results.py -f input/15635/8at6.fasta -a input/15635/AF3_results
```

### 2.5 Dock domains into cryo-EM map (Skip this step if you don't want to use AF3 at input level)
**Required Format:**
```bash
python utils/dock_in_map.py \
    -m <path/to/cryo-EM/map> \
    -c <contour_level> \
    -r <resolution> \
    -f <path/to/fasta/file> \
    -a <path/to/AF3_results> \
    -x <path/to/phenix/activation>
```

**Example:**
```bash
python utils/dock_in_map.py \
    -m input/15635/emd_15635.map \
    -c 0.0242 \
    -r 3.7 \
    -f input/15635/8at6.fasta \
    -a input/15635/AF3_results \
    -x <path/to/phenix/activation>
```

### 2.6 Run data preprocessing, deep learning prediction and atomic model building
**Required Format:**
```bash
python run.py \
    -m <path/to/cryo-EM/map> \
    -f <path/to/fasta/file> \
    -a <path/to/AF3_results> \
    -p <path/to/pulchra> \
    --run_phenix \
    -x <path/to/phenix/activation> \
    -r <resolution>
```

**Example:**
```bash
python run.py \
    -m input/15635/emd_15635.map \
    -f input/15635/8at6.fasta \
    -a input/15635/AF3_results \
    -p modules/pulchra304/src/pulchra \
    --run_phenix \
    -x <path/to/phenix/activation> \
    -r 3.7
```

</details>

---

### 🧬 Step 3: Results
Final atomic model will be saved in: `output/15635_8at6_MICA_all_atom_model.pdb`

</details>

<details>
<summary><h2>📥 Downloading Datasets and Results (Optional)</h2></summary>

This section provides instructions for downloading the training dataset, test dataset, and pre-computed results for the MICA project.

### 📊 Available Downloads

| Dataset | Size | Description | Use Case |
|---------|------|-------------|----------|
| **Training Dataset** | ~48 GB | Curated cryo-EM maps with corresponding FASTA sequences, PDB files, and AlphaFold3 structures for model training | Model development and training |
| **Test Dataset** | ~20 GB | Evaluation datasets containing cryo-EM maps and associated FASTA sequences, ground truth structures and AlphaFold3 structures | Model validation and benchmarking |
| **Pre-computed Results** | ~150 MB | MICA predictions on test results | Comparison and analysis |

### 1. Downloading Training Dataset (Optional)
```
curl https://zenodo.org/records/15756654/files/Training_Dataset.tar.gz?download=1 --output Training_Dataset.tar.gz
tar -xzvf Training_Dataset.tar.gz
rm Training_Dataset.tar.gz
```

### 2. Downloading Test Dataset (Optional)
```
curl https://zenodo.org/records/15756654/files/Test_Dataset.tar.gz?download=1 --output Test_Dataset.tar.gz
tar -xzvf Test_Dataset.tar.gz
rm Test_Dataset.tar.gz
```

### 3. Downloading Pre-computed Results for MICA (Optional)
```
curl https://zenodo.org/records/15756654/files/Results.tar.gz?download=1 --output Results.tar.gz
tar -xzvf Results.tar.gz
rm Results.tar.gz
```

</details>

<details>
<summary><h2>🔥 Training MICA </h2></summary>

This section provides comprehensive instructions for training MICA from scratch.

### 📁 Initial Training Dataset Structure
```
Training_Dataset/
└── Raw_Data/
    └── 0071/                    # Dataset entry 0071
        ├── 6qve.fasta          # Protein FASTA file
        ├── 6qve.pdb            # Ground Truth PDB structure
        ├── 0071_af3_docked.pdb # AF3 docked structure
        └── emd_0071.map        # Cryo-EM density map
```
### 🚀 Training Process

### 1. Download Training Dataset
Download Training Dataset from previous step (skip this step if you have already downloaded or are going to use your own data)

If using your own data, please compile the data in the same format as in *Initial Training Dataset Structure*

### 2. Create Full Training Data with Grids
```bash
sh create_training_data.sh
```

After running this script, your training dataset directory structure should look like:
```
Training_Dataset/
├── Grids/
│   ├── AA_masks/                # Amino acid mask files
│   ├── ALA_encodings/           # Alanine residue encodings
│   ├── ARG_encodings/           # Arginine residue encodings
│   ├── ASN_encodings/           # Asparagine residue encodings
│   ├── ASP_encodings/           # Aspartic acid residue encodings
│   ├── BB_masks/                # Backbone mask files
│   ├── C_encodings/             # Carbon atom encodings
│   ├── CA_encodings/            # Alpha carbon encodings
│   ├── CA_masks/                # Alpha carbon mask files
│   ├── CYS_encodings/           # Cysteine residue encodings
│   ├── GLN_encodings/           # Glutamine residue encodings
│   ├── GLU_encodings/           # Glutamic acid residue encodings
│   ├── GLY_encodings/           # Glycine residue encodings
│   ├── HIS_encodings/           # Histidine residue encodings
│   ├── ILE_encodings/           # Isoleucine residue encodings
│   ├── LEU_encodings/           # Leucine residue encodings
│   ├── LYS_encodings/           # Lysine residue encodings
│   ├── MET_encodings/           # Methionine residue encodings
│   ├── N_encodings/             # Nitrogen atom encodings
│   ├── normalized_maps/         # Normalized density maps
│   ├── O_encodings/             # Oxygen atom encodings
│   ├── PHE_encodings/           # Phenylalanine residue encodings
│   ├── PRO_encodings/           # Proline residue encodings
│   ├── SER_encodings/           # Serine residue encodings
│   ├── THR_encodings/           # Threonine residue encodings
│   ├── TRP_encodings/           # Tryptophan residue encodings
│   ├── TYR_encodings/           # Tyrosine residue encodings
│   └── VAL_encodings/           # Valine residue encodings
├── Processed_Data/              # Intermediate processed files
└── Raw_Data/
    └── 0071/                    # Dataset entry 0071
        ├── 6qve.fasta          # Protein sequence file
        ├── 6qve.pdb            # Experimental structure
        ├── 0071_af3_docked.pdb # AF3 docked structure
        └── emd_0071.map        # Cryo-EM density map
```

### 3. Run Training
```bash
python train.py
```

### ⚙️ Parameter Tuning

Parameters tuning can be found in `training_config.py` or passed directly to `train.py`

**Required Format:**
```bash
python train.py --batch_size <size> --learning_rate <rate> --epochs <num>
```

**Example:**
```bash
python train.py --batch_size 4 --learning_rate 0.0001 --epochs 100
```

</details>

<details>
<summary><h2>📄 Rights and Permissions</h2></summary>

Open Access \
This article is licensed under a Creative Commons Attribution 4.0 International License, which permits use, sharing, adaptation, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons license, and indicate if changes were made. The images or other third party material in this article are included in the article's Creative Commons license, unless indicated otherwise in a credit line to the material. If material is not included in the article's Creative Commons license and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/.

### 🔐 Data Usage Rights

The datasets provided with MICA are derived from publicly available sources:

- **Cryo-EM density maps**: Retrieved from the Electron Microscopy Data Bank (EMDB)
- **Protein structures**: Retrieved from the Protein Data Bank (PDB)
- **AlphaFold3 predictions**: Generated using the AlphaFold Server

</details>

<details>
<summary><h2>📚 How to Cite This Work</h2></summary>

### Primary Citation

If you use MICA in your research, please cite our paper:

```bibtex
@article{Gyawali2025.07.03.663071,
	author = {Gyawali, Rajan and Dhakal, Ashwin and Cheng, Jianlin},
	title = {Multimodal deep learning integration of cryo-EM and AlphaFold3 for high-accuracy protein structure determination},
	elocation-id = {2025.07.03.663071},
	year = {2025},
	doi = {10.1101/2025.07.03.663071},
	publisher = {Cold Spring Harbor Laboratory},
	issn = {2692-8205},
	URL = {https://www.biorxiv.org/content/early/2025/07/03/2025.07.03.663071},
	journal = {bioRxiv}
}

```

### Dataset Citation

If you use our curated datasets, please also cite:

```bibtex
@dataset{gyawali_2025_15756654,
	author = {Gyawali, Rajan and Dhakal, Ashwin and Cheng, Jianlin},
	title = {Multimodal deep learning integration of cryo-EM and AlphaFold3 for high-accuracy protein structure determination},
	year = {2025},
	publisher={Zenodo},
	doi={10.5281/zenodo.15756654},
	url={https://zenodo.org/records/15756654},
}
```

</details>

<details>
<summary><h2>🙏 Acknowledgments</h2></summary>



This project utilizes several key computational tools for data preprocessing and postprocessing. We gratefully acknowledge and thank the developers and maintainers of Phenix, Merizo, Pulchra, and EModelX(+AF) for their invaluable contributions to the structural biology community. 

**Phenix** has been utilized for docking AlphaFold3 predicted structures into cryo-EM density maps and refining the full atom model built by MICA.

**Merizo** has been used for domain segmentation from AlphaFold3-predicted structures.

**EModelX(+AF)** has been utilized for backbone tracing from the results predicted from deep learning model of the MICA.

**PULCHRA** has been used for full atom model building from the Cα-backbone model.

If you use MICA in your research, please cite the following tools as well:

#### Phenix
```bibtex
@article{Liebschner2019,
	author = {Liebschner, Dorothee and Afonine, Pavel V. and Baker, Matthew L. and Bunkóczi, Gábor and Chen, Vincent B. and Croll, Tristan I. and Hintze, Bradley and Hung, Li-Wei and Jain, Swati and McCoy, Airlie J. and Moriarty, Nigel W. and Oeffner, Robert D. and Poon, Billy K. and Prisant, Michael G. and Read, Randy J. and Richardson, Jane S. and Richardson, David C. and Sammito, Massimo D. and Sobolev, Oleg V. and Stockwell, Duncan H. and Terwilliger, Thomas C. and Urzhumtsev, Alexandre G. and Videau, Lizbeth L. and Williams, Christopher J. and Adams, Paul D.},
	title = {Macromolecular structure determination using X-rays, neutrons and electrons: recent developments in Phenix},
	journal = {Acta Crystallographica Section D},
	volume = {75},
	number = {10},
	pages = {861--877},
	year = {2019},
	month = {Oct},
	doi = {10.1107/S2059798319011471}
}
```

```bibtex
@article{Afonine2018,
	author = {Afonine, Pavel V. and Poon, Billy K. and Read, Randy J. and Sobolev, Oleg V. and Terwilliger, Thomas C. and Urzhumtsev, Alexandre and Adams, Paul D.},
	title = {Real-space refinement in PHENIX for cryo-EM and crystallography},
	journal = {Acta Crystallographica Section D},
	volume = {74},
	number = {6},
	pages = {531--544},
	year = {2018},
	month = {Jun},
	doi = {10.1107/S2059798318006551}
}
```
#### Merizo
```bibtex
@article{Lau2023,
	author = {Lau, Andy M. and Kandathil, Shaun M. and Jones, David T.},
	title = {Merizo: a rapid and accurate protein domain segmentation method using invariant point attention},
	journal = {Nature Communications},
	volume = {14},
	number = {1},
	pages = {8445},
	year = {2023},
	doi = {10.1038/s41467-023-43934-4}
}
```

#### EModelX(+AF)
```bibtex
@article{Chen2024,
	author = {Chen, Sheng and Zhang, Sen and Fang, Xiaoyu and Lin, Liang and Zhao, Huiying and Yang, Yuedong},
	title = {Protein complex structure modeling by cross-modal alignment between cryo-EM maps and protein sequences},
	journal = {Nature Communications},
	volume = {15},
	number = {1},
	pages = {8808},
	year = {2024},
	doi = {10.1038/s41467-024-53116-5}
}
```

#### PULCHRA
```bibtex
@article{Rotkiewicz2008,
	author = {Rotkiewicz, Piotr and Skolnick, Jeffrey},
	title = {Fast procedure for reconstruction of full-atom protein models from reduced representations},
	journal = {Journal of Computational Chemistry},
	volume = {29},
	number = {9},
	pages = {1460-1465},
	year = {2008},
	doi = {10.1002/jcc.20906}
}
```

#### License

Please ensure compliance with the individual licenses of each tool when using them in your research.
</details>

<details>
<summary><h2>📧 Contact</h2></summary>

**Jianlin (Jack) Cheng, PhD, AAAS Fellow**  
Curators' Distinguished Professor   
Department of Electrical Engineering and Computer Science  
University of Missouri  
Columbia, MO 65211, USA  
 **Email**: [chengji@missouri.edu](mailto:chengji@missouri.edu)

</details>
