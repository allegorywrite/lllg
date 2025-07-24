#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import re
from typing import List, Tuple, Dict

def parse_map_file(map_file: str) -> Tuple[np.ndarray, int, int]:
    """Parse the map file and return the grid, width, and height."""
    with open(map_file, 'r') as f:
        lines = f.readlines()
    
    # Find dimensions
    height = int(lines[1].split()[1])
    width = int(lines[2].split()[1])
    
    # Parse map grid
    grid = np.zeros((height, width), dtype=int)
    map_start_idx = 4  # Skip type, height, width, map header lines
    
    for i in range(height):
        line = lines[map_start_idx + i].strip()
        for j in range(width):
            if j < len(line):
                if line[j] == '@' or line[j] == 'T':  # Obstacles
                    grid[i, j] = 1
                else:  # Free space
                    grid[i, j] = 0
    
    return grid, width, height

def parse_result_file(result_file: str) -> Dict:
    """Parse the result file and extract agent trajectories."""
    with open(result_file, 'r') as f:
        content = f.read()
    
    # Extract basic info
    agents_match = re.search(r'agents=(\d+)', content)
    num_agents = int(agents_match.group(1)) if agents_match else 0
    
    # Extract starts and goals
    starts_match = re.search(r'starts=(.+)', content)
    goals_match = re.search(r'goals=(.+)', content)
    
    starts = []
    goals = []
    
    if starts_match:
        start_coords = re.findall(r'\((\d+),(\d+)\)', starts_match.group(1))
        starts = [(int(x), int(y)) for x, y in start_coords]
    
    if goals_match:
        goal_coords = re.findall(r'\((\d+),(\d+)\)', goals_match.group(1))
        goals = [(int(x), int(y)) for x, y in goal_coords]
    
    # Extract solution trajectories (only between solution= and local_guidance=)
    lines = content.split('\n')
    trajectories = {}
    in_solution = False
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        if line == 'solution=':
            in_solution = True
            continue
        elif line == 'local_guidance=':
            in_solution = False
            break
        
        if in_solution and ':' in line and line:
            parts = line.split(':')
            if len(parts) == 2:
                try:
                    step = int(parts[0])
                    coords_str = parts[1]
                    coords = re.findall(r'\((\d+),(\d+)\)', coords_str)
                    step_positions = [(int(x), int(y)) for x, y in coords]
                    
                    trajectories[step] = {}
                    for agent_idx, pos in enumerate(step_positions):
                        trajectories[step][agent_idx] = pos
                except ValueError:
                    continue
    
    return {
        'num_agents': num_agents,
        'starts': starts,
        'goals': goals,
        'trajectories': trajectories
    }

def build_density_table(data: Dict, width: int, height: int, radius: float = 1.0) -> Tuple[dict, int]:
    """Build 3D density table for spatio-temporal analysis."""
    max_step = max(data['trajectories'].keys()) if data['trajectories'] else 0
    
    # Initialize density table: [time][x][y] = density_count
    density_table = {}
    for t in range(max_step + 1):
        density_table[t] = {}
        for x in range(width):
            density_table[t][x] = {}
            for y in range(height):
                density_table[t][x][y] = 0
    
    # Populate density table by incrementing counts for all trajectory points
    for agent_id in range(data['num_agents']):
        for step in range(max_step + 1):
            if step in data['trajectories'] and agent_id in data['trajectories'][step]:
                pos = data['trajectories'][step][agent_id]
                center_x, center_y = pos[0], pos[1]
                
                # Increment density for all points within radius
                for t in range(max(0, step - int(radius)), min(max_step + 1, step + int(radius) + 1)):
                    for x in range(max(0, center_x - int(radius)), min(width, center_x + int(radius) + 1)):
                        for y in range(max(0, center_y - int(radius)), min(height, center_y + int(radius) + 1)):
                            # Check if point is within radius in 3D space
                            dist = np.sqrt((t - step)**2 + (x - center_x)**2 + (y - center_y)**2)
                            if dist <= radius:
                                density_table[t][x][y] += 1
    
    return density_table, max_step

def get_density(density_table: dict, t: int, x: int, y: int) -> int:
    """Get density from pre-calculated table."""
    if (t in density_table and x in density_table[t] and y in density_table[t][x]):
        return density_table[t][x][y]
    return 0

def setup_matplotlib_style():
    """Set up matplotlib style for consistent visualization."""
    plt.rcParams.update({
        'font.size': 18,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'serif'],
        'axes.linewidth': 2.5,
        'axes.edgecolor': 'black',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.linewidth': 2.0,
        'grid.color': 'gray',
        'grid.alpha': 0.3,
        'figure.facecolor': 'white',
        'figure.edgecolor': 'black',
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 10,
        'ytick.major.size': 10,
        'xtick.minor.size': 6,
        'ytick.minor.size': 6,
        'xtick.major.pad': 10,
        'ytick.major.pad': 10,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'legend.frameon': True,
        'legend.fancybox': False,
        'legend.edgecolor': 'black',
        'legend.facecolor': 'white',
        'legend.framealpha': 1.0,
        'legend.fontsize': 16,
        'pdf.fonttype': 42,
        'ps.fonttype': 42
    })