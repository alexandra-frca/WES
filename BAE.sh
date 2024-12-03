#!/bin/sh
#SBATCH --job-name=BAE # Job name
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err 
#SBATCH --partition=large-x86
#SBATCH --account=f202407005cpcaa1x
#SBATCH --time=72:00:00 
#SBATCH --nodes=1 
#SBATCH --ntasks=1 

python scripts/TestBAE.py