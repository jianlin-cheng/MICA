git clone https://github.com/jianlin-cheng/MICA
cd MICA
conda env create -f environment.yml
conda activate MICA
curl https://zenodo.org/records/15756654/files/trained_models.tar.gz?download=1 --output trained_models.tar.gz
tar -xzvf trained_models.tar.gz
rm trained_models.tar.gz
curl https://zenodo.org/records/15756654/files/input.tar.gz?download=1 --output input.tar.gz
tar -xzvf input.tar.gz
rm input.tar.gz
python run.py -m input/15635/emd_15635.map -f input/15635/8at6.fasta -i input/15635 --run_pulchra --pulchra_path=modules/pulchra304/src/pulchra --resolution=3.7
