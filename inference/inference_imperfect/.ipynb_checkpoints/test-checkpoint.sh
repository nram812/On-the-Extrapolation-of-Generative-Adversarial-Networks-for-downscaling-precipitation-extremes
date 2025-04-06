#!/bin/bash -l
BASE_PATH="/nesi/nobackup/niwa00018/morrishdg/Downscaled_Preprocessed"

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
GCM_string_name="$BASE_PATH/$CMIP_TYPE/$2/$3/$SCENARIO/$4/day/${CMIP_TYPE}_${2}_${3}_${SCENARIO}_${4}_day_gn_?????????.nc"
echo $GCM_string_name