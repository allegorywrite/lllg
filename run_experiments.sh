#!/bin/bash

# Navigate to the working directory
cd ~/aist_ws/lg_lacam/

# Create results directory with timestamp
timestamp=$(date +"%Y%m%d_%H%M%S")
result_dir="results/${timestamp}"
mkdir -p "${result_dir}/heatmap"
mkdir -p "${result_dir}/trajectory"

echo "Results will be saved in: ${result_dir}"

# Define maps and scenarios
declare -A maps=(
    ["maze-128-128-10"]="assets/asset_for_benchmark/mapf-map/maze-128-128-10.map"
    ["maze-128-128-1"]="assets/asset_for_benchmark/mapf-map/maze-128-128-1.map"
    ["maze-32-32-4"]="assets/asset_for_benchmark/mapf-map/maze-32-32-4.map"
    ["maze-32-32-2"]="assets/asset_for_benchmark/mapf-map/maze-32-32-2.map"
    ["empty-32-32"]="assets/asset_for_benchmark/mapf-map/empty-32-32.map"
    ["den312d"]="assets/asset_for_benchmark/mapf-map/den312d.map"
    ["ost003d"]="assets/asset_for_benchmark/mapf-map/ost003d.map"
    ["warehouse-20-40-10-2-1"]="assets/asset_for_benchmark/mapf-map/warehouse-20-40-10-2-1.map"
    ["warehouse-20-40-10-2-2"]="assets/asset_for_benchmark/mapf-map/warehouse-20-40-10-2-2.map"
    ["berlin"]="assets/asset_for_benchmark/mapf-map/Berlin_1_256.map"
    ["random"]="assets/asset_for_benchmark/mapf-map/random-32-32-20.map"
    ["random_64"]="assets/asset_for_benchmark/mapf-map/random-64-64-20.map"
    ["room-64-64-16"]="assets/asset_for_benchmark/mapf-map/room-64-64-16.map"
    ["room-64-64-8"]="assets/asset_for_benchmark/mapf-map/room-64-64-8.map"
)

declare -A scenarios=(
    ["maze-128-128-10"]="assets/asset_for_benchmark/mapf-scen-random/scen-random/maze-128-128-10-random-1.scen"
    ["maze-128-128-1"]="assets/asset_for_benchmark/mapf-scen-random/scen-random/maze-128-128-1-random-1.scen"
    ["maze-32-32-4"]="assets/asset_for_benchmark/mapf-scen-random/scen-random/maze-32-32-4-random-1.scen"
    ["maze-32-32-2"]="assets/asset_for_benchmark/mapf-scen-random/scen-random/maze-32-32-2-random-1.scen"
    ["empty-32-32"]="assets/asset_for_benchmark/mapf-scen-random/scen-random/empty-32-32-random-1.scen"
    ["den312d"]="assets/asset_for_benchmark/mapf-scen-random/scen-random/den312d-random-1.scen"
    ["ost003d"]="assets/asset_for_benchmark/mapf-scen-random/scen-random/ost003d-random-1.scen"
    ["warehouse-20-40-10-2-1"]="assets/asset_for_benchmark/mapf-scen-random/scen-random/warehouse-20-40-10-2-1-random-1.scen"
    ["warehouse-20-40-10-2-2"]="assets/asset_for_benchmark/mapf-scen-random/scen-random/warehouse-20-40-10-2-2-random-1.scen"
    ["berlin"]="assets/asset_for_benchmark/mapf-scen-random/scen-random/Berlin_1_256-random-1.scen"
    ["random"]="assets/asset_for_benchmark/mapf-scen-random/scen-random/random-32-32-20-random-1.scen"
    ["random_64"]="assets/asset_for_benchmark/mapf-scen-random/scen-random/random-64-64-20-random-1.scen"
    ["room-64-64-16"]="assets/asset_for_benchmark/mapf-scen-random/scen-random/room-64-64-16-random-1.scen"
    ["room-64-64-8"]="assets/asset_for_benchmark/mapf-scen-random/scen-random/room-64-64-8-random-1.scen"
)

# Define methods
declare -A methods=(
    ["vanilla"]=""
    ["gg"]="--gg"
    ["lg8"]="--lg --lg_window 8"
    ["lg20"]="--lg --lg_window 20"
    ["gg_lg20"]="--gg --lg --lg_window 20"
    ["gg_lg8"]="--gg --lg --lg_window 8"
)

# Define heatmap max values for each map
declare -A heatmap_max=(
    # ["maze-128-128-10"]="230"
    ["maze-128-128-10"]="380"
    ["maze-32-32-4"]="0"
    ["maze-32-32-2"]="0"
    ["empty-32-32"]="40"
    ["den312d"]="380"
    ["ost003d"]="380"
    ["warehouse-20-40-10-2-1"]="100"
    ["warehouse-20-40-10-2-2"]="35"
    ["berlin"]="130"
    ["random"]="110"
    ["random_64"]="150"
    ["room-64-64-16"]="600"
    ["room-64-64-8"]="600"
)

declare -A heatmap_min=(
    ["maze-128-128-10"]="0"
    ["maze-32-32-4"]="0"
    ["maze-32-32-2"]="0"
    ["empty-32-32"]="0"
    ["den312d"]="100"
    ["ost003d"]="0"
    ["warehouse-20-40-10-2-1"]="10"
    ["warehouse-20-40-10-2-2"]="5"
    ["berlin"]="0"
    ["random"]="0"
    ["random_64"]="0"
    ["room-64-64-16"]="0"
    ["room-64-64-8"]="100"
)

# Define projection density max values for trajectory3d mode
declare -A projection_max_density=(
    # ["maze-128-128-10"]="530"
    ["maze-128-128-10"]="1200"
    ["maze-32-32-4"]="0"
    ["maze-32-32-2"]="0"
    ["empty-32-32"]="760"
    ["den312d"]="50"
    ["ost003d"]="1000"
    ["warehouse"]="500"
    ["berlin"]="30"
    ["random"]="730"
    ["random_64"]="35"
)

declare -A projection_min_density=(
    ["maze-128-128-10"]="100"
    ["maze-32-32-4"]="0"
    ["maze-32-32-2"]="0"
    ["empty-32-32"]="100"
    ["den312d"]="0"
    ["ost003d"]="1000"
    ["warehouse"]="100"
    ["berlin"]="0"
    ["random"]="100"
    ["random_64"]="0"
)

# Define trajectory density max values for trajectory3d mode
declare -A trajectory_max_density=(
    ["maze-128-128-10"]="32"
    ["maze-32-32-4"]="0"
    ["maze-32-32-2"]="0"
    ["empty-32-32"]="35"
    ["den312d"]="15"
    ["ost003d"]="1000"
    ["warehouse"]="22"
    ["berlin"]="10"
    ["random"]="32"
    ["random_64"]="12"
)

declare -A trajectory_min_density=(
    ["maze-128-128-10"]="10"
    ["maze-32-32-4"]="0"
    ["maze-32-32-2"]="0"
    ["empty-32-32"]="20"
    ["den312d"]="0"
    ["ost003d"]="1000"
    ["warehouse"]="15"
    ["berlin"]="0"
    ["random"]="20"
    ["random_64"]="0"
)

# Define agent numbers for each map
declare -A agent_numbers=(
    ["maze-128-128-10"]="1000"
    ["maze-128-128-1"]="1000"
    ["maze-32-32-4"]="395"
    ["maze-32-32-2"]="300"
    ["empty-32-32"]="300"
    ["den312d"]="1000"
    ["ost003d"]="1000"
    ["warehouse-20-40-10-2-1"]="1000"
    ["warehouse-20-40-10-2-2"]="1000"
    ["berlin"]="1000"
    ["random"]="400"
    ["random_64"]="400"
    ["room-64-64-16"]="1000"
    ["room-64-64-8"]="1000"
)

# Define makespan values for trajectory3d mode
declare -A makespan_values=(
    ["maze-128-128-10"]="520"
    ["maze-32-32-4"]="0"
    ["maze-32-32-2"]="0"
    ["empty-32-32"]="60"
    ["den312d"]="200"
    ["ost003d"]="1000"
    ["warehouse"]="280"
    ["berlin"]="180"
    ["random"]="120"
    ["random_64"]="180"
)

# Run experiments
for map_name in warehouse-20-40-10-2-1; do
# for map_name in den312d warehouse berlin random random_64; do
    for method_name in vanilla gg lg20 gg_lg20; do
        echo "Running ${map_name} with ${method_name}..."
        
        build/main -i "${scenarios[$map_name]}" \
                   -m "${maps[$map_name]}" \
                   -N "${agent_numbers[$map_name]}" -v 3 -t 30\
                   ${methods[$method_name]}
        
        # Copy result.txt to results directory
        cp build/result.txt "${result_dir}/${map_name}_${method_name}_result.txt"
        
        echo "Creating visits heatmap for ${map_name}_${method_name}..."
        venv/bin/python scripts/visualize_trajectories.py \
                        --result build/result.txt \
                        --map "${maps[$map_name]}" \
                        --output "${result_dir}/heatmap/${map_name}_${method_name}" \
                        --mode "visits" \
                        --heatmap-max "${heatmap_max[$map_name]}" \
                        --heatmap-min "${heatmap_min[$map_name]}" \
                        --log-scale \
                        # --heatmap-interval 150 \
        
        # echo "Creating trajectory3d visualization for ${map_name}_${method_name}..."
        # venv/bin/python scripts/visualize_trajectories.py \
        #                 --result build/result.txt \
        #                 --map "${maps[$map_name]}" \
        #                 --output "${result_dir}/trajectory/${map_name}_${method_name}" \
        #                 --mode "trajectory3d" \
        #                 --projection-max-density "${projection_max_density[$map_name]}" \
        #                 --projection-min-density "${projection_min_density[$map_name]}" \
        #                 --trajectory-max-density "${trajectory_max_density[$map_name]}" \
        #                 --trajectory-min-density "${trajectory_min_density[$map_name]}" \
        #                 --makespan "${makespan_values[$map_name]}" \
        #                 --log-scale \
        
        echo "Completed ${map_name} with ${method_name}"
        echo "---"
    done
done

echo "All 16 experiments completed!"