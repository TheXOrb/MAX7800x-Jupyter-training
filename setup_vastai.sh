#!/bin/bash

# Automated setup script for vast.ai MAX78002 training environment
# This script automates the entire setup process from conda installation to Jupyter kernel setup

set -e  # Exit on any error

echo "============================================"
echo "Starting automated vast.ai setup..."
echo "============================================"

# 1. Download and install Anaconda/Conda
echo "[Step 1/11] Downloading Anaconda..."
cd ~
wget -q https://repo.anaconda.com/archive/Anaconda3-2025.12-1-Linux-x86_64.sh

# 2. Install conda silently with all default settings
echo "[Step 2/11] Installing Anaconda..."
bash Anaconda3-2025.12-1-Linux-x86_64.sh -b -p ~/anaconda3

# 3. Initialize conda for bash
echo "[Step 3/11] Initializing conda..."
eval "$(~/anaconda3/bin/conda shell.bash hook)"
~/anaconda3/bin/conda init bash

# 4. Create conda environment with Python 3.8
echo "[Step 4/11] Creating conda environment with Python 3.8..."
~/anaconda3/bin/conda create -n max78-training-jupyter python=3.8 -y

# 5. Activate the conda environment (use source to make it work in scripts)
echo "[Step 5/11] Activating conda environment..."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate max78-training-jupyter

# Verify we're in the right environment
echo "Active Python: $(which python)"
echo "Active environment: $CONDA_DEFAULT_ENV"

# 6. Clone the repository with submodules
echo "[Step 6/11] Cloning MAX7800x-Jupyter-training repository..."
cd ~
if [ -d "MAX7800x-Jupyter-training" ]; then
    echo "Repository already exists, updating..."
    cd MAX7800x-Jupyter-training
    git pull
    git submodule update --init --recursive
else
    git clone --recurse-submodules https://github.com/InES-HPMM/MAX7800x-Jupyter-training
    cd MAX7800x-Jupyter-training
fi

# 7. Install pycocotools via conda
echo "[Step 7/11] Installing pycocotools..."
conda install -c conda-forge pycocotools -y

# 8. Install ipykernel for Jupyter
echo "[Step 8/11] Installing ipykernel..."
conda install ipykernel -y

# 9. Install requirements (now running in activated environment)
echo "[Step 9/11] Installing Python requirements..."
cd ~/MAX7800x-Jupyter-training

# Remove pycocotools from requirements since it's already installed via conda
echo "Removing pycocotools from requirements (already installed via conda)..."
grep -v "pycocotools" requirements-cu11.txt > requirements-cu11-temp.txt
pip install -r requirements-cu11-temp.txt
rm requirements-cu11-temp.txt

# 10. Register the kernel with Jupyter
echo "[Step 10/11] Registering kernel with Jupyter..."
python -m ipykernel install --user --name=max78-training-jupyter --display-name "MAX78 Training (Python 3.8)"

# 11. Install additional dependencies
echo "[Step 11/11] Installing additional dependencies..."
pip install seaborn wfdb imbalanced-learn
conda install -c conda-forge ipywidgets -y

# Optional: Download MIT-BIH dataset (commented out as it's large)
echo "[Optional] Downloading MIT-BIH dataset..."
cd ~/MAX7800x-Jupyter-training
wget -r -N -c -np https://physionet.org/files/mitdb/1.0.0/

echo "============================================"
echo "Setup completed successfully!"
echo "============================================"
echo ""
echo "To use the environment:"
echo "1. Run: source ~/anaconda3/bin/activate max78-training-jupyter"
echo "   OR: conda activate max78-training-jupyter"
echo "2. Navigate to: cd ~/MAX7800x-Jupyter-training"
echo "3. Start Jupyter: jupyter notebook --no-browser --port=8888 --allow-root --ip=0.0.0.0"
echo ""
echo "The Jupyter kernel 'max78-training-jupyter' is ready to use!"
echo "============================================"
