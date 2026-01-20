#!/bin/bash
# Usage: sbatch submit_causal_dis.slurm -t <t_args> -n <n_args>
# Notice: submitting job from sinteractive will lead to errors
#SBATCH --job-name=job_causal_dis_${NAME_SUFFIX}
#SBATCH --job-name=job_causal_dis    # Job named-checkzone
#SBATCH --account=essb
#SBATCH --output=%x_%j.out          # Standard output log file
#SBATCH --error=%x_%j.err            # Error log file
#SBATCH --time=12:00:00              # Time limit (hh:mm:ss)
#SBATCH --partition=tier3          # Partition/queue name (modify based on your cluster)
#SBATCH --ntasks=1                   # Number of tasks
#SBATCH --cpus-per-task=12            # Number of CPU cores per task
#SBATCH --mem=100G                     # Memory per node
##SBATCH --gres=gpu:1                 # Request a GPU if needed (remove if not applicable)
#SBATCH --mail-user=slack:@ymliee
#SBATCH --mail-type=<BEGIN,END,FAIL,ALL>
# Load necessary modules (if required)
module load anaconda  # Load Anaconda module if needed
source /home/ymliee/miniconda3/bin/activate
# Activate the Conda environment
conda activate alfafa

# Run the Python script
python causal_dis.py "$@" 

# Deactivate the Conda environment (optional, but good practice)
conda deactivate
