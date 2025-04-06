#!/bin/bash -l

module use /opt/nesi/modulefiles/
module unuse /opt/niwa/CS500_centos7_skl/modules/all
module unuse /opt/niwa/share/modules/all

export SYSTEM_STRING=CS500
export OS_ARCH_STRING=centos7
export CPUARCH_STRING=skl
#export PYTHONNOUSERSITE=/home/rampaln/.local/lib/python3.9/site-packages
export PYTHONUSERBASE=/nesi/project/niwa00018/rampaln/conda_tmp
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rampaln/.local/lib/python3.9/site-packages
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64
### Note add a separate log folder that Maxime has outlined
module purge # optional
module load NIWA
#module load gcc/9.3.0
module load CDO/1.9.5-GCC-7.1.0
module load Miniforge3
# Navigate to the code directory
CODE_DIR="/nesi/project/niwa03712/queenle/ML_emulator"
cd $CODE_DIR

# Define arrays
gcm=("EC-Earth3-Veg-LR")
#gcm=("MPI-ESM1-2-HR")

variant=("r1i1p1f1")

ssps=("ssp585")
variables=("pr" "tasmax" "tasmin" "sfcwind" "sfcwindmax")

for i in "${!gcm[@]}"; do
  # Loop through GCMs and list GCM and variant pairs
  gcm="${gcm[i]}"
  variant="${variant[i]}"
  # Listed Pairs
  for item in "${ssps[@]}"; do
    ssp="$item"
    # Loop through ssp pairs
    for varname in "${variables[@]}"; do
      variable_downscaled=$varname
      sbatch -J "${ssp}_${gcm}_${variant}_${variable_downscaled}" ./code/applying_ml_model/v3/apply_emulator_v3_mahuika.sl ${ssp} "${gcm}" "${variant}" "${variable_downscaled}"
      echo "$gcm $variant $ssp $variable_downscaled"
    done
  done
done
