#!/bin/bash

#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --partition=batch
#SBATCH --time=10:00:00

# Activate base conda environment
source ~/.bashrc
conda activate base

# Install TotalSegmentator if not already installed

# Run your script
python batchmode.py