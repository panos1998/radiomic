#!/bin/bash


#SBATCH --cpus-per-task=8
#SBATCH --mem=32G  # 4 CPUs × 2500 MB = 10 GB
#SBATCH --nodes=1
#SBATCH --partition=batch
#SBATCH --time=03:00:00


pyradiomics --verbosity 5 /home/p/ppetrd/thesisCode/maskSegMatchingMUKERJE_SET1.csv \
  --jobs=6 -o /home/p/ppetrd/mukerjeset1/firstOrderResults5.csv -f csv \
  -p /home/p/ppetrd/thesisCode/params4.yaml


pyradiomics --verbosity 5 /home/p/ppetrd/thesisCode/maskSegMatchingMUKERJE_SET2.csv \
  --jobs=6 -o /home/p/ppetrd/mukerjeset2/firstOrderResults5.csv -f csv \
  -p /home/p/ppetrd/thesisCode/params4.yaml
