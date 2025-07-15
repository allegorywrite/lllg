#!/bin/bash

# Navigate to the working directory
cd ~/aist_ws/lg_lacam/

# Create results directory with timestamp
timestamp=$(date +"%Y%m%d_%H%M%S")
result_dir="results/${timestamp}"
mkdir -p "${result_dir}"

echo "Results will be saved in: ${result_dir}"

# Define maps and scenarios
declare -A maps=(
    ["den312d"]="assets/asset_for_benchmark/mapf-map/den312d.map"
    ["warehouse"]="assets/asset_for_benchmark/mapf-map/warehouse-10-20-10-2-1.map"
    ["berlin"]="assets/asset_for_benchmark/mapf-map/Berlin_1_256.map"
    ["random"]="assets/asset_for_benchmark/mapf-map/random-32-32-20.map"
    ["random_64"]="assets/asset_for_benchmark/mapf-map/random-64-64-20.map"
)

declare -A scenarios=(
    ["den312d"]="assets/asset_for_benchmark/mapf-scen-random/scen-random/den312d-random-1.scen"
    ["warehouse"]="assets/asset_for_benchmark/mapf-scen-random/scen-random/warehouse-10-20-10-2-1-random-1.scen"
    ["berlin"]="assets/asset_for_benchmark/mapf-scen-random/scen-random/Berlin_1_256-random-1.scen"
    ["random"]="assets/asset_for_benchmark/mapf-scen-random/scen-random/random-32-32-20-random-1.scen"
    ["random_64"]="assets/asset_for_benchmark/mapf-scen-random/scen-random/random-64-64-20-random-1.scen"
)

# Define methods
declare -A methods=(
    ["vanilla"]=""
    ["gg"]="--gg"
    ["lg8"]="--lg --lg_window 8"
    ["lg20"]="--lg --lg_window 20"
    ["gg_lg20"]="--gg --lg --lg_window 20"
)

# Define colorbar max values for each map
declare -A colorbar_max=(
    ["den312d"]="380"
    ["warehouse"]="240"
    ["berlin"]="130"
    ["random"]="110"
    ["random_64"]="150"
)

declare -A colorbar_min=(
    ["den312d"]="100"
    ["warehouse"]="100"
    ["berlin"]="0"
    ["random"]="0"
    ["random_64"]="0"
)

# Define agent numbers for each map
declare -A agent_numbers=(
    ["den312d"]="1000"
    ["warehouse"]="1000"
    ["berlin"]="1000"
    ["random"]="400"
    ["random_64"]="1000"
)

# Run experiments
# for map_name in den312d warehouse berlin random; do
for map_name in random_64; do
    for method_name in vanilla gg lg8 lg20 gg_lg20; do
        echo "Running ${map_name} with ${method_name}..."
        
        build/main -i "${scenarios[$map_name]}" \
                   -m "${maps[$map_name]}" \
                   -N "${agent_numbers[$map_name]}" -v 3 -t 20\
                   ${methods[$method_name]}
        
        # Copy result.txt to results directory
        cp build/result.txt "${result_dir}/${map_name}_${method_name}_result.txt"
        
        echo "Creating visualization for ${map_name}_${method_name}..."
        venv/bin/python scripts/visualize_trajectories.py \
                        --result build/result.txt \
                        --map "${maps[$map_name]}" \
                        --output "${result_dir}/${map_name}_${method_name}" \
                        --colorbar-max "${colorbar_max[$map_name]}" \
                        --colorbar-min "${colorbar_min[$map_name]}"
        
        echo "Completed ${map_name} with ${method_name}"
        echo "---"
    done
done

echo "All 16 experiments completed!"