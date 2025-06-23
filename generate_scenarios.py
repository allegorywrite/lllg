#!/usr/bin/env python3
"""
Generate scenario files for LaCAM with valid agent positions avoiding obstacles
"""

import random
import sys
import os
from math import sqrt

def read_map_file(map_file):
    """Read map file and return grid and dimensions"""
    with open(map_file, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    map_type = lines[0].strip().split()[1]  # type octile
    height = int(lines[1].strip().split()[1])
    width = int(lines[2].strip().split()[1])
    
    # Parse map grid
    grid = []
    for i in range(4, 4 + height):
        if i < len(lines):
            grid.append(lines[i].strip())
        else:
            break
    
    return grid, width, height

def get_valid_positions(grid, width, height):
    """Get all valid (non-obstacle) positions"""
    valid_positions = []
    for y in range(height):
        for x in range(width):
            if y < len(grid) and x < len(grid[y]) and grid[y][x] == '.':
                valid_positions.append((x, y))
    return valid_positions

def calculate_distance(start, goal):
    """Calculate Euclidean distance between two points"""
    return sqrt((goal[0] - start[0])**2 + (goal[1] - start[1])**2)

def generate_scenario(map_name, width, height, valid_positions, num_agents, seed=None):
    """Generate a scenario with num_agents agents"""
    if seed is not None:
        random.seed(seed)
    
    if len(valid_positions) < num_agents:
        raise ValueError(f"Not enough valid positions for {num_agents} agents. Need at least {num_agents}, but only have {len(valid_positions)}")
    
    scenario_lines = ["version 1"]
    used_starts = set()
    used_goals = set()
    
    for i in range(num_agents):
        # Select unique start position
        max_attempts = 1000
        attempts = 0
        while attempts < max_attempts:
            start = random.choice(valid_positions)
            if start not in used_starts:
                used_starts.add(start)
                break
            attempts += 1
        else:
            raise ValueError(f"Could not find unique start position for agent {i}")
        
        # Select unique goal position (different from start and not used by other agents)
        attempts = 0
        while attempts < max_attempts:
            goal = random.choice(valid_positions)
            if goal not in used_goals and goal != start:
                used_goals.add(goal)
                break
            attempts += 1
        else:
            raise ValueError(f"Could not find unique goal position for agent {i}")
        
        distance = calculate_distance(start, goal)
        
        scenario_lines.append(f"{i}\t{map_name}\t{width}\t{height}\t{start[0]}\t{start[1]}\t{goal[0]}\t{goal[1]}\t{distance:.8f}")
    
    return scenario_lines

def main():
    map_file = "assets/random-10-10.map"
    map_name = "random-10-10.map"
    scenario_dir = "assets/random-10-10.map-scen-random/scen-random"
    
    # Read map
    grid, width, height = read_map_file(map_file)
    valid_positions = get_valid_positions(grid, width, height)
    
    print(f"Map dimensions: {width}x{height}")
    print(f"Valid positions: {len(valid_positions)}")
    
    # Generate scenarios 1-20 with maximum possible agents
    # Each agent needs unique start and unique goal, but starts can overlap with other agents' goals
    max_agents = min(70, len(valid_positions))
    print(f"Maximum agents possible: {max_agents}")
    
    for scenario_num in range(1, 21):
        scenario_file = f"{scenario_dir}/random-10-10-random-{scenario_num}.scen"
        
        try:
            # Generate scenario with unique seed for each
            scenario_lines = generate_scenario(map_name, width, height, valid_positions, max_agents, seed=scenario_num)
            
            # Write scenario file
            with open(scenario_file, 'w') as f:
                f.write('\n'.join(scenario_lines) + '\n')
            
            print(f"Generated {scenario_file} with {len(scenario_lines)-1} agents")
        except ValueError as e:
            print(f"Error generating scenario {scenario_num}: {e}")

if __name__ == "__main__":
    main()