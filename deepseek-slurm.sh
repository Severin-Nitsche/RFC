#!/usr/bin/env zsh

#SBATCH --job-name=annotate-deepseek
#SBATCH --output=deepseek_%J.txt
#SBATCH --partition=c23g
#SBATCH --gres=gpu:2
#SBATCH --mem=256G

#SBATCH --time=00:15:00
#SBATCH --mail-user=severin.nitsche@rwth-aachen.de
#SBATCH --mail-type=ALL

ml load GCCcore/.13.3.0
ml load Python/3.12.3
cd /home/ll464721/RFC
echo "JOB ID: $SLURM_JOB_ID"
source venv/deepseek/bin/activate
python src/deepseek.py
