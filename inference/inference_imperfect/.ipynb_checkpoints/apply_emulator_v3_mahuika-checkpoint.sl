#!/bin/bash -l
#SBATCH --job-name=maui_GPU_job
#SBATCH --partition=niwa_work
#SBATCH --time=00:15:00
#SBATCH --cluster=maui_ancil
#SBATCH --mem=180G
#SBATCH --gpus-per-node=A100:1
#SBATCH --cpus-per-task=8
#SBATCH --account=niwap03712
#SBATCH --output log/%j-%x.out
#SBATCH --error log/%j-%x.out



module purge # optional
module load NeSI
module load gcc/9.3.0
module load Miniforge3
#module load CDO/1.9.5-GCC-7.1.0
#module load Miniconda3/4.12.0
module load cuDNN/8.6.0.163-CUDA-11.8.0
nvidia-smi

# activate /nesi/project/niwa00018/queenle/ml_env_v2

# code directory
# Base path
CODE_DIR="/nesi/project/niwa03712/queenle/ML_emulator"
BASE_PATH="/nesi/project/niwa03712/CMIP6_data/Downscaled_Preprocessed"
output_dir='/nesi/nobackup/niwa00018/ML_Downscaled_CMIP6/'
cd $CODE_DIR

# Arguments passed to the script
# $1: Scenario string (e.g., historical, ssp370)
# $2: Model name (e.g., ACCESS-ESM1-5)
# $3: Realization string (e.g., r37i1p1f1)
# $4: Variable (e.g., tasmax)
# Determine if $1 contains 'historical' or 'ssp'
if [[ $1 == *"historical"* ]]; then
    CMIP_TYPE="CMIP"
    SCENARIO="historical"
elif [[ $1 == *"ssp"* ]]; then
    CMIP_TYPE="ScenarioMIP"
    SCENARIO=$1
else
    echo "Error: Unknown scenario type in \$1"
    exit 1
fi

variable=$4
gcm=$2
variant=$3
/nesi/project/niwa00018/rampaln/envs/ml_env_v2/bin/python ./code/applying_ml_model/v3/run_emulator_v4.py "GAN" $variable $BASE_PATH $gcm $SCENARIO $variant $output_dir "perfect_emulator" $CODE_DIR
#/nesi/project/niwa00018/queenle/ml_env_v2/bin/python3 ./code/applying_ml_model/v3/run_emulator_v4.py "GAN" $variable $BASE_PATH $gcm $SCENARIO $variant $output_dir "perfect_emulator" $CODE_DIR