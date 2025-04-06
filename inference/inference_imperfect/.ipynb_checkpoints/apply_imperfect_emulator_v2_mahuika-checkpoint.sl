#!/bin/bash -l
#SBATCH --job-name=GPU_job
#SBATCH --partition=niwa_work
#SBATCH --time=14:59:00
#SBATCH --cluster=maui_ancil
#SBATCH --mem=350G
#SBATCH --gpus-per-node=A100:1
#SBATCH --account=niwap03712
#SBATCH --mail-type=ALL
#SBATCH --output log/%j-%x.out
#SBATCH --error log/%j-%x.out

module purge && module load NeSI
#module load gcc/9.3.0
module load cuDNN/8.6.0.163-CUDA-11.8.0
#nvidia-smi

# activate /nesi/project/niwa00018/queenle/ml_env_v2

# code directory
cd /nesi/project/niwa03712/rampaln/apply_model_laura_test/applying_ml_model/v2
# Base path
BASE_PATH="/nesi/nobackup/niwa00018/morrishdg/Downscaled_Preprocessed"
# Once we are happy with the number of GCMs/ QA/QC we store this data on /nesi/project/niwa00018.
output_dir='/nesi/nobackup/niwa00018/rampaln/ML_emulator_downscaled/'

# Arguments passed to the script
# $1: Scenario string (e.g., historical, ssp370)
# $2: Institution name (e.g., CSIRO)
# $3: Model name (e.g., ACCESS-ESM1-5)
# $4: Realization string (e.g., r37i1p1f1)

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
# Construct the path
GCM_string_name="$BASE_PATH/$CMIP_TYPE/$2/$3/$SCENARIO/$4/day/${CMIP_TYPE}_${2}_${3}_${SCENARIO}_${4}_day*.nc"
OUTPUT_FILENAME_imperfect="$output_dir/$CMIP_TYPE/$2/$3/$SCENARIO/$4/day/ML_Downscaled_${5}_${CMIP_TYPE}_${2}_${3}_${SCENARIO}_${4}_day_imperfect_trained.nc"
echo "processing data from $GCM_string_name to $OUTPUT_FILENAME"
/nesi/project/niwa00018/queenle/ml_env_v2/bin/python3 run_emulator_v2.py "GAN" $5 $GCM_string_name $OUTPUT_FILENAME_imperfect "imperfect_emulator"
OUTPUT_FILENAME_perfect="$output_dir/$CMIP_TYPE/$2/$3/$SCENARIO/$4/day/ML_Downscaled_${5}_${CMIP_TYPE}_${2}_${3}_${SCENARIO}_${4}_day_perfect_trained.nc"
/nesi/project/niwa00018/queenle/ml_env_v2/bin/python3 run_emulator_v2.py "GAN" $5 $GCM_string_name $OUTPUT_FILENAME_perfect "perfect_emulator"
