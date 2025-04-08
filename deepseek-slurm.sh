#!/usr/bin/env zsh

#SBATCH --job-name=annotate
#SBATCH --output=annotate_%J.txt
#SBATCH --partition=c23g
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=120G

#SBATCH --time=02:00:00
#SBATCH --mail-user=severin.nitsche@rwth-aachen.de
#SBATCH --mail-type=ALL

ml load GCCcore/.13.3.0
ml load Python/3.12.3
cd /home/ll464721/RFC
echo "JOB ID: $SLURM_JOB_ID"
source venv/deepseek/bin/activate
python -m src.deepseek.runner
