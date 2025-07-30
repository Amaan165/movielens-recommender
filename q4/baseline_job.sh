#!/bin/bash
#SBATCH --mail-user=nac8810@nyu.edu
#SBATCH --job-name=baseline_q3_k_100
#SBATCH --output=/scratch/nac8810/logs/%x-%j.out
#SBATCH --error=/scratch/nac8810/logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --partition=short
# === 1. Prepare environment ===
module purge

# === 2. Define project paths ===
PROJECT_DIR=/home/nac8810/capstone-bdcs-69/temp_experiments/q3_train_val_test_split
SCRATCH_RUN_DIR=/scratch/nac8810/baseline_q3_run-${SLURM_JOB_ID}
PYTHON_PATH=/scratch/work/public/dask/2021.10.0/bin/python

TRAIN_PARQUET=/home/nac8810/capstone-bdcs-69/temp_experiments/q3_train_val_test_split/dataset_random_shuffle/train.parquet
VAL_PARQUET=/home/nac8810/capstone-bdcs-69/temp_experiments/q3_train_val_test_split/dataset_random_shuffle/val.parquet
TEST_PARQUET=/home/nac8810/capstone-bdcs-69/temp_experiments/q3_train_val_test_split/dataset_random_shuffle/test.parquet
MOVIE_STATS=/home/nac8810/capstone-bdcs-69/temp_experiments/q3_train_val_test_split/dataset_random_shuffle/movie_stats.parquet
OUTPUT_DIR=$SCRATCH_RUN_DIR/results_baseline_q3

# === 3. Setup runtime directory ===
mkdir -p $SCRATCH_RUN_DIR
mkdir -p $OUTPUT_DIR
cd $SCRATCH_RUN_DIR

# === 4. Copy your script locally ===
cp $PROJECT_DIR/baseline_q4.py .


# === 5. Run the recommender system ===
echo "🟡 Running baseline_q4.py at $(date)"
$PYTHON_PATH baseline_q4.py \
  --val $VAL_PARQUET \
  --test $TEST_PARQUET \
  --movie-stats $MOVIE_STATS \
  --output-dir $OUTPUT_DIR \
  --workers 4 \
  --threads 1 \
  --memory 4GB \
  --top-k 100 \
  --beta-range 500

echo "✅ baseline_q4.py finished at $(date)"
