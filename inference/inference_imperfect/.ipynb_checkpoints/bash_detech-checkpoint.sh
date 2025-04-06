#!/bin/bash

# Define the base directory where the data is located
base_dir="/nesi/project/niwa03712/CMIP6_data/Downscaled_Preprocessed/ScenarioMIP"

# Arrays to hold the detected values
institutions=()
gcms=()
variants=()
ssps=("ssp126")
historical="historical"

# Loop through the directories and capture institutions, GCMs, and variants
for institution_dir in "$base_dir"/*; do
  if [ -d "$institution_dir" ]; then
    institution=$(basename "$institution_dir")

    # Loop through each institution folder to find GCMs
    for gcm_dir in "$institution_dir"/*; do
      if [ -d "$gcm_dir" ]; then
        gcm=$(basename "$gcm_dir")

        # Add GCM to list if it's not already present
        if [[ ! " ${gcms[@]} " =~ " ${gcm} " ]]; then
          gcms+=("$gcm")
        fi

        # Loop through GCM folder for historical and SSP scenarios
        for scenario_dir in "$gcm_dir"/*; do
          if [ -d "$scenario_dir" ]; then
            scenario=$(basename "$scenario_dir")

            # Check if the folder is for historical or one of the SSPs
            if [[ "$scenario" == "$historical" || " ${ssps[@]} " =~ " ${scenario} " ]]; then
              # Loop through variants (r1i1p1f1 type subfolders)
              for variant_dir in "$scenario_dir"/*; do
                if [ -d "$variant_dir" ]; then
                  variant=$(basename "$variant_dir")
                  variants+=("$variant")
                fi
              done
            fi
          fi
        done
      fi
    done

    # Add institution to list if it's not already present
    if [[ ! " ${institutions[@]} " =~ " ${institution} " ]]; then
      institutions+=("$institution")
    fi
  fi
done

# Display the results
echo "Institutions: ${institutions[@]}"
echo "GCMs: ${gcms[@]}"
echo "Variants: ${variants[@]}"
echo "SSPs: ${ssps[@]}"

# Loop through all combinations and print them
#for institution in "${institutions[@]}"; do
#  for gcm in "${gcms[@]}"; do
#    for variant in "${variants[@]}"; do
#      for ssp in "${ssps[@]}"; do
#        echo "$base_dir/$institution/$gcm/$ssp/$variant/day"
#      done
#    done
#  done
#done
