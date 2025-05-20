#!/bin/bash

MAP_FILE="assets/random-32-32-20.map"
SCEN_EVEN_DIR="assets/random-32-32-20.map-scen-even/scen-even"
SCEN_RANDOM_DIR="assets/random-32-32-20.map-scen-random/scen-random"
OUTPUT_DIR="build"
EXECUTABLE="build/main"

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

echo "Starting experiments for scen-even files..."
# Run experiments for scen-even files (N=100)
# scen-even-1, 2, 3 are already done. Start from 4.
for i in $(seq 4 25); do
  SCEN_FILE_BASE="random-32-32-20-even-${i}"
  SCEN_FILE_PATH="${SCEN_EVEN_DIR}/${SCEN_FILE_BASE}.scen"
  
  # N=100, no --gg
  OUTPUT_FILE_NOGG="${OUTPUT_DIR}/result_even-${i}_N100_nogg.txt"
  CMD_NOGG="${EXECUTABLE} -m ${MAP_FILE} -i ${SCEN_FILE_PATH} -N 100 -o ${OUTPUT_FILE_NOGG}"
  echo "Executing: ${CMD_NOGG}"
  ${CMD_NOGG}
  if [ $? -ne 0 ]; then
    echo "Error executing: ${CMD_NOGG}"
  fi
  
  # N=100, with --gg
  OUTPUT_FILE_GG="${OUTPUT_DIR}/result_even-${i}_N100_gg.txt"
  CMD_GG="${EXECUTABLE} -m ${MAP_FILE} -i ${SCEN_FILE_PATH} -N 100 --gg -o ${OUTPUT_FILE_GG}"
  echo "Executing: ${CMD_GG}"
  ${CMD_GG}
  if [ $? -ne 0 ]; then
    echo "Error executing: ${CMD_GG}"
  fi
done
echo "Finished experiments for scen-even files."

echo "Starting experiments for scen-random files..."
# Run experiments for scen-random files (N=100, 200, 300)
for i in $(seq 1 25); do
  SCEN_FILE_BASE="random-32-32-20-random-${i}"
  SCEN_FILE_PATH="${SCEN_RANDOM_DIR}/${SCEN_FILE_BASE}.scen"
  
  for N_AGENTS in 100 200 300; do
    # No --gg
    OUTPUT_FILE_NOGG="${OUTPUT_DIR}/result_random-${i}_N${N_AGENTS}_nogg.txt"
    CMD_NOGG="${EXECUTABLE} -m ${MAP_FILE} -i ${SCEN_FILE_PATH} -N ${N_AGENTS} -o ${OUTPUT_FILE_NOGG}"
    echo "Executing: ${CMD_NOGG}"
    ${CMD_NOGG}
    if [ $? -ne 0 ]; then
      echo "Error executing: ${CMD_NOGG}"
      # If N is invalid for this specific random scenario, we might want to note it or skip.
      # For now, we assume the user's guidance implies these N values are generally okay for random scenarios.
    fi
    
    # With --gg
    OUTPUT_FILE_GG="${OUTPUT_DIR}/result_random-${i}_N${N_AGENTS}_gg.txt"
    CMD_GG="${EXECUTABLE} -m ${MAP_FILE} -i ${SCEN_FILE_PATH} -N ${N_AGENTS} --gg -o ${OUTPUT_FILE_GG}"
    echo "Executing: ${CMD_GG}"
    ${CMD_GG}
    if [ $? -ne 0 ]; then
      echo "Error executing: ${CMD_GG}"
    fi
  done
done
echo "Finished experiments for scen-random files."

echo "All experiments finished."
