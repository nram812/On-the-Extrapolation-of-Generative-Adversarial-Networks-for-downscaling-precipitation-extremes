#!/bin/bash -l
#SBATCH --job-name=GPU_job
#SBATCH --partition=hgx
#SBATCH --time=03:59:00
#SBATCH --mem=320G
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=A100:1
#SBATCH --account=niwa03712
#SBATCH --output log/%j-%x.out
#SBATCH --error log/%j-%x.out

module purge # optional
module load NeSI
module load gcc/9.3.0
#module load CDO/1.9.5-GCC-7.1.0
#module load Miniconda3/4.12.0
module load cuDNN/8.1.1.33-CUDA-11.2.0
module load Miniforge3
# set the experiment name that we are implementing

# Change directory to a working path that you are on. 
#cd /nesi/project/niwa00018/ML_downscaling_CCAM/On-the-Extrapolation-of-Generative-Adversarial-Networks-for-downscaling-precipitation-extremes

/nesi/project/niwa00018/rampaln/envs/ml_env_v2/bin/python /nesi/project/niwa00018/ML_downscaling_CCAM/On-the-Extrapolation-of-Generative-Adversarial-Networks-for-downscaling-precipitation-extremes/model_inference/model_inference_all_gcms.py $1

# the argument $1 is how you run the script .i.e. please please the config file of the GCM you plan on running it for. 
