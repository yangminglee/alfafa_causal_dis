#!/bin/bash

# --- SLURM DIRECTIVES (Dynamic Job Naming) ---
# 🌟 Job Name: Named as <TARGET_TRAIT>_<ARRAY_ID>
#             (SLURM appends the array ID automatically)
#
#SBATCH --account=essb
#SBATCH --partition=tier3
#SBATCH --nodes=1        
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=10-11:45:00

#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err 

#SBATCH --mail-user=slack:@ymliee
#SBATCH --mail-type=<BEGIN,END,FAIL,ALL>

TARGET_TRAIT="$SLURM_JOB_NAME"    


# Set dynamic variables

RUN_DATE=$(date +%Y%m%d) # Get current date in YYYYMMDD format
# 🌟 SEED: Use SLURM array ID as the unique, repeatable random seed
UNIQUE_JOB_ID=${SLURM_JOB_ID:--1}
SEED="${UNIQUE_JOB_ID}${SLURM_ARRAY_TASK_ID}"

module load anaconda
source /home/ymliee/miniconda3/bin/activate
conda activate alfafa

# 🌟 Output Directory: <TRAIT_NAME>_YYYYMMDD
RESULTS_DIR="./${TARGET_TRAIT}_${RUN_DATE}"
echo "Save Results to {RESULTS_DIR}"
# Create necessary directories
mkdir -p logs
mkdir -p "$RESULTS_DIR"
OUTPUT_FILE="${RESULTS_DIR}/result_${TARGET_TRAIT}_seed${SEED}.txt"

DATA_DIR="~/data/plant/alfalfa"

echo "Starting DCL analysis (CPU Mode) for SEED: $SEED"

python feature_select_lasso_pca.py \
    --trait $TARGET_TRAIT \
    --output "$OUTPUT_FILE" \
    --seed $SEED

echo "DML analysis for Repetition ID $SEED finished."
conda deactivate
