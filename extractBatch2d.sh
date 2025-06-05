#!/bin/bash


#SBATCH --cpus-per-task=10
#SBATCH --mem=64G  # 4 CPUs × 2500 MB = 10 GB
#SBATCH --nodes=1
#SBATCH --partition=batch
#SBATCH --time=05:00:00

# mkdir /home/p/ppetrd/mukerjeset1
pyradiomics --verbosity 5 /home/p/ppetrd/thesisCode/maskSegMatchingMUKERJE_SET1.csv \
  --jobs=10 -o /home/p/ppetrd/mukerjeset1/firstOrderResults200.csv -f csv \
  -p /home/p/ppetrd/thesisCode/params.yaml

# mkdir /home/p/ppetrd/mukerjeset2
# pyradiomics --verbosity 5 /home/p/ppetrd/thesisCode/maskSegMatchingMUKERJE_SET2.csv \
#   --jobs=6 -o /home/p/ppetrd/mukerjeset2/firstOrderResults200.csv -f csv \
#   -p /home/p/ppetrd/thesisCode/params.yaml
