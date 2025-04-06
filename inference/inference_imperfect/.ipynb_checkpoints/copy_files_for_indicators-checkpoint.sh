#!/bin/bash -l

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rampaln/.local/lib/python3.9/site-packages
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64
### Note add a separate log folder that Maxime has outlined
module purge # optional
module load NIWA
#module load gcc/9.3.0
module load CDO/1.9.5-GCC-7.1.0
module load Miniforge3
# Navigate to the code directory
cd "/nesi/nobackup/niwa03712/"

var_list=("pr" "sfcwind" "sfcwindmax" "tasmax" "tasmin")
scenarios=("historical" "ssp126" "ssp245" "ssp370" "ssp585")


for scenario in "${scenarios[@]}"; do
  for varname in "${var_list[@]}"; do

    mv_path="ML_Downscaled_CMIP6/NIWA-REMS_CCAM_public/${scenario}/daily/${varname}"

    find ML_Downscaled_CMIP6 -type f -path "*/*/*/${scenario}/*/day/${varname}/NZ_Domain/*/${varname}_*.nc" | while read file; do
      #echo $file
      echo "Moving $file to $mv_path"
      mkdir -p "$mv_path"
      mv "$file" "$mv_path"
    done
  done

done



#/nesi/project/niwa00018/gibsonp/run_ccam/post_processing/production_runs/CCAM_CMIP6_public/ssp126/daily/pr