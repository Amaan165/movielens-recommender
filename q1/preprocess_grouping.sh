#!/bin/bash
#SBATCH --mail-user=nac8810@nyu.edu
#SBATCH --job-name=minhash_pipeline
#SBATCH --output=/scratch/nac8810/logs/%x-%j.out
#SBATCH --error=/scratch/nac8810/logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --partition=short

# === 1. Prepare environment ===
module purge

# === 2. Define project paths ===
PROJECT_DIR=/home/nac8810/capstone-bdcs-69/temp_experiments
SCRATCH_RUN_DIR=/scratch/nac8810/minhash_run-${SLURM_JOB_ID}
PYTHON_PATH=/scratch/work/public/dask/2021.10.0/bin/python

INPUT_PARQUET=/home/nac8810/capstone-bdcs-69/temp_experiments/ratings.csv  # Must already be in your working directory or scratch
GROUPED_PARQUET=/home/nac8810/capstone-bdcs-69/temp_experiments/grouped_user_movies.parquet

# === 3. Setup runtime directory ===
mkdir -p $SCRATCH_RUN_DIR
cd $SCRATCH_RUN_DIR

# === 4. Copy scripts ===
cp $PROJECT_DIR/preprocess_grouping.py .

# === 5. Run Dask preprocessing ===
echo "🟡 Step 1: Grouping movieIds with Dask at $(date)"
$PYTHON_PATH preprocess_grouping.py \
  --input $INPUT_PARQUET \
  --output $GROUPED_PARQUET \
  --workers 8 \
  --scale 4

echo "✅ All steps completed at $(date)"
