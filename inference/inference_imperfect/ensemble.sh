#!/bin/bash -l


module purge && module load NeSI
#module load gcc/9.3.0
module load cuDNN/8.6.0.163-CUDA-11.8.0
#nvidia-smi

# activate /nesi/project/niwa00018/queenle/ml_env_v2

# code directory
cd /nesi/project/niwa03712/rampaln/apply_model_laura_test/applying_ml_model/v2
# Base path
# Arguments passed to the script
# $1: Scenario string (e.g., historical, ssp370)
# $2: Institution name (e.g., CSIRO)
# $3: Model name (e.g., ACCESS-ESM1-5)
# $4: Realization string (e.g., r37i1p1f1)
items=(
 r10i1p1f1 r18i1p1f1 r25i1p1f1 r32i1p1f1 r3i1p1f1
 r11i1p1f1 r19i1p1f1 r26i1p1f1 r33i1p1f1 r40i1p1f1
 r12i1p1f1 r1i1p1f1  r27i1p1f1 r34i1p1f1 r4i1p1f1
 r13i1p1f1 r20i1p1f1 r28i1p1f1 r35i1p1f1 r5i1p1f1
 r14i1p1f1 r21i1p1f1 r29i1p1f1 r36i1p1f1 r6i1p1f1
 r15i1p1f1 r22i1p1f1 r2i1p1f1  r37i1p1f1 r7i1p1f1
 r16i1p1f1 r23i1p1f1 r30i1p1f1 r38i1p1f1 r8i1p1f1
 r17i1p1f1 r24i1p1f1 r31i1p1f1 r39i1p1f1 r9i1p1f1
)

items=("r10i1p1f1" "r2i1p1f1" "r4i1p1f1" "r6i1p1f1" "r8i1p1f1"
      "r10i1p2f1" "r2i1p2f1" "r4i1p2f1" "r6i1p2f1" "r8i1p2f1"
      "r1i1p1f1" "r3i1p1f1" "r5i1p1f1" "r7i1p1f1" "r9i1p1f1"
      "r1i1p2f1" "r3i1p2f1" "r5i1p2f1" "r7i1p2f1")
# Loop through the list
for item in "${items[@]}"; do
  echo $item
#  sbatch apply_imperfect_emulator_v2_mahuika.sl "historical" "CSIRO" "ACCESS-ESM1-5" $item "tasmin"
#  sbatch apply_imperfect_emulator_v2_mahuika.sl "ssp370" "CSIRO" "ACCESS-ESM1-5" $item "tasmin"
  sbatch apply_imperfect_emulator_v2_mahuika.sl "historical" "CCCma" "CanESM5" $item "pr"
  sbatch apply_imperfect_emulator_v2_mahuika.sl "ssp370" "CCCma" "CanESM5" $item "pr"
  sbatch apply_imperfect_emulator_v2_mahuika.sl "historical" "CCCma" "CanESM5" $item "tasmin"
  sbatch apply_imperfect_emulator_v2_mahuika.sl "ssp370" "CCCma" "CanESM5" $item "tasmin"
  sbatch apply_imperfect_emulator_v2_mahuika.sl "historical" "CCCma" "CanESM5" $item "tasmax"
  sbatch apply_imperfect_emulator_v2_mahuika.sl "ssp370" "CCCma" "CanESM5" $item "tasmax"
  # Add your processing logic here
done

