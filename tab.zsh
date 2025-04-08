#!/usr/bin/env zsh

#SBATCH --job-name=train-rfc
#SBATCH --output=tab_%J.txt
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1

#SBATCH --time=01:00:00
#SBATCH --mail-user=severin.nitsche@rwth-aachen.de
#SBATCH --mail-type=END

ml load GCCcore/.12.2.0
ml load Python/3.10.8
cd /home/ll464721/RFC
echo "JOB ID: $SLURM_JOB_ID"
source venv/tab/bin/activate
python -m src.tab.train_model
