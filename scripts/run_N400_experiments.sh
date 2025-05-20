#!/bin/bash

MAP_FILE="assets/random-32-32-20.map"
SCEN_RANDOM_DIR="assets/random-32-32-20.map-scen-random/scen-random"
OUTPUT_DIR="build"
EXECUTABLE="build/main"

N_AGENTS=100

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}/agent_${N_AGENTS}

echo "Starting experiments for scen-random files with N=${N_AGENTS}..."
# Run experiments for scen-random files (N=400 only)

for i in $(seq 1 25); do
  SCEN_FILE_BASE="random-32-32-20-random-${i}"
  SCEN_FILE_PATH="${SCEN_RANDOM_DIR}/${SCEN_FILE_BASE}.scen"
  
  # Vanilla
  OUTPUT_FILE_NOGG="${OUTPUT_DIR}/agent_${N_AGENTS}/result_random-${i}_N${N_AGENTS}_vanilla.txt"
  CMD_NOGG="${EXECUTABLE} -m ${MAP_FILE} -i ${SCEN_FILE_PATH} -N ${N_AGENTS} -o ${OUTPUT_FILE_NOGG}"
  echo "Executing: ${CMD_NOGG}"
  ${CMD_NOGG}
  if [ $? -ne 0 ]; then
    echo "Error executing: ${CMD_NOGG}"
  fi
  
  # With --gg
  OUTPUT_FILE_GG="${OUTPUT_DIR}/agent_${N_AGENTS}/result_random-${i}_N${N_AGENTS}_gg.txt"
  CMD_GG="${EXECUTABLE} -m ${MAP_FILE} -i ${SCEN_FILE_PATH} -N ${N_AGENTS} --gg -o ${OUTPUT_FILE_GG}"
  echo "Executing: ${CMD_GG}"
  ${CMD_GG}
  if [ $? -ne 0 ]; then
    echo "Error executing: ${CMD_GG}"
  fi

  # With --lg
  OUTPUT_FILE_GG="${OUTPUT_DIR}/agent_${N_AGENTS}/result_random-${i}_N${N_AGENTS}_lg.txt"
  CMD_GG="${EXECUTABLE} -m ${MAP_FILE} -i ${SCEN_FILE_PATH} -N ${N_AGENTS} --lg -o ${OUTPUT_FILE_GG}"
  echo "Executing: ${CMD_GG}"
  ${CMD_GG}
  if [ $? -ne 0 ]; then
    echo "Error executing: ${CMD_GG}"
  fi

  # With --gg --lg
  OUTPUT_FILE_GG="${OUTPUT_DIR}/agent_${N_AGENTS}/result_random-${i}_N${N_AGENTS}_gg_lg.txt"
  CMD_GG="${EXECUTABLE} -m ${MAP_FILE} -i ${SCEN_FILE_PATH} -N ${N_AGENTS} --gg --lg -o ${OUTPUT_FILE_GG}"
  echo "Executing: ${CMD_GG}"
  ${CMD_GG}
  if [ $? -ne 0 ]; then
    echo "Error executing: ${CMD_GG}"
  fi
done
echo "Finished experiments for scen-random files with N=400."
