#!/bin/bash

#SBATCH --job-name="train_roberta_semeval"
#SBATCH --mail-user= # enter email
#SBATCH --mail-type=ALL
#SBATCH --time=2:00:00
#SBATCH --partition=alpha
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=6G
#SBATCH --output=slurm/out-%j.out
#SBATCH --error=slurm/err-%j.out

# ------------------------------ #
# Module and Environment Setup
# ------------------------------ #

# Purge any previously loaded modules and load the required ones
module purge
module load GCCcore/13.2.0
module load Python/3.11.5

# Define project-related paths
PROJECT_DIR="."  # Enter the absolute path to your project directory if needed; leave empty for current directory
SCRIPT_NAME="training.py"
OUTPUT_DIR="${PROJECT_DIR}/outputs"

# Determine virtual environment directory. If PROJECT_DIR is empty, use the current directory.
VENV_DIR="${PROJECT_DIR}/.venv"

# Check if the virtual environment exists; if not, create one.
if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found. Creating one at $VENV_DIR..."
    python -m venv "$VENV_DIR"
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

# Upgrade pip and install required packages
pip install --upgrade pip
pip install --upgrade pandas
pip install --upgrade scikit-learn
pip install --upgrade transformers
pip install --upgrade torch
pip install --upgrade datasets
pip install --upgrade "accelerate>=0.26.0"
pip install --upgrade evaluate
pip install --upgrade sentencepiece
pip install --upgrade protobuf

# ------------------------------ #
# Prepare Output Directory
# ------------------------------ #

# Ensure that the output directory exists
mkdir -p "$OUTPUT_DIR"

# ------------------------------ #
# Run the Training Script
# ------------------------------ #

# Execute the training script using srun; output and errors are redirected to specified log files
srun python "${PROJECT_DIR}/${SCRIPT_NAME}" > "$OUTPUT_DIR/final_configs.log" 2> "$OUTPUT_DIR/final_configs_error.log"
