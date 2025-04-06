#!/bin/bash -l

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
gcm=("ACCESS-ESM1-5")

variant=(
  "r10i1p1f1" "r18i1p1f1" "r25i1p1f1" "r32i1p1f1" "r3i1p1f1"
  "r11i1p1f1" "r19i1p1f1" "r26i1p1f1" "r33i1p1f1" "r40i1p1f1"
  "r12i1p1f1" "r1i1p1f1"  "r27i1p1f1" "r34i1p1f1" "r4i1p1f1"
  "r13i1p1f1" "r20i1p1f1" "r28i1p1f1" "r35i1p1f1" "r5i1p1f1"
  "r14i1p1f1" "r21i1p1f1" "r29i1p1f1" "r36i1p1f1" "r6i1p1f1"
  "r15i1p1f1" "r22i1p1f1" "r2i1p1f1"  "r37i1p1f1" "r7i1p1f1"
  "r16i1p1f1" "r23i1p1f1" "r30i1p1f1" "r38i1p1f1" "r8i1p1f1"
  "r17i1p1f1" "r24i1p1f1" "r31i1p1f1" "r39i1p1f1" "r9i1p1f1"
)
# variant=("r10i1p1f1" "r2i1p1f1" "r4i1p1f1" "r6i1p1f1" "r8i1p1f1"
#       "r10i1p2f1" "r2i1p2f1" "r4i1p2f1" "r6i1p2f1" "r8i1p2f1"
#       "r1i1p1f1" "r3i1p1f1" "r5i1p1f1" "r7i1p1f1" "r9i1p1f1"
#       "r1i1p2f1" "r3i1p2f1" "r5i1p2f1" "r7i1p2f1")
# gcm=("CanESM5")
ssps=("historical" "ssp370")
variables=("pr" "tasmax" "tasmin" "sfcWind")

for i in "${!variant[@]}"; do
  # Loop through GCMs and list GCM and variant pairs
  gcm="${gcm[0]}"
  variant="${variant[i]}"
  echo $gcm $variant
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
