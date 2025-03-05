#!/usr/bin/env zsh

#SBATCH --job-name=annotate-deepseek
#SBATCH --output=deepseek_%A_%a.txt
#SBATCH --partition=c23g
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=20G

#SBATCH --array=1-3%1
#SBATCH --time=01:00:00
#SBATCH --mail-user=severin.nitsche@rwth-aachen.de
#SBATCH --mail-type=ALL

ml load GCCcore/.13.3.0
ml load Python/3.12.3
cd /home/ll464721/RFC
echo "JOB ID: $SLURM_JOB_ID"
source venv/deepseek/bin/activate
case ${SLURM_ARRAY_TASK_ID} in
  "1")
    python src/deepseek_annotate.py
    ;;
  "2")
    python src/deepseek_verify.py
    ;;
  "3")
    python src/deepseek_classify.py
    ;;
esac
