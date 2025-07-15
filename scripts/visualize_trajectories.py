#!/home/initial/aist_ws/lg_lacam/venv/bin/python

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import argparse
import re
from typing import List, Tuple, Dict

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

def create_heatmap_visualization(grid: np.ndarray, data: Dict, width: int, height: int, 
                               output_file: str = None, heatmap_type: str = 'visits', max_value: int = None, min_value: int = None, log_scale: bool = False):
    """Create heatmap visualization showing visits or stops with congestion spectrum."""
    
    # Create heatmap matrix
    heatmap = np.zeros((height, width), dtype=int)
    
    max_step = max(data['trajectories'].keys()) if data['trajectories'] else 0
    
    if heatmap_type == 'visits':
        # Count all visits for each cell (excluding start and goal positions)
        for step in range(max_step + 1):
            if step in data['trajectories']:
                for agent_id in range(data['num_agents']):
                    if agent_id in data['trajectories'][step]:
                        pos = data['trajectories'][step][agent_id]
                        x, y = pos
                        
                        # Skip if this is start or goal position for this agent
                        is_start = (agent_id < len(data['starts']) and pos == data['starts'][agent_id])
                        is_goal = (agent_id < len(data['goals']) and pos == data['goals'][agent_id])
                        
                        if not (is_start or is_goal) and 0 <= x < width and 0 <= y < height:
                            heatmap[y, x] += 1
                            
    elif heatmap_type == 'stops':
        # Count only stops (when agent stays in same position, excluding start and goal)
        for agent_id in range(data['num_agents']):
            prev_pos = None
            for step in range(max_step + 1):
                if step in data['trajectories'] and agent_id in data['trajectories'][step]:
                    pos = data['trajectories'][step][agent_id]
                    x, y = pos
                    
                    # Check if agent stopped (same position as previous step)
                    if prev_pos is not None and prev_pos == pos:
                        # Skip if this is start or goal position for this agent
                        is_start = (agent_id < len(data['starts']) and pos == data['starts'][agent_id])
                        is_goal = (agent_id < len(data['goals']) and pos == data['goals'][agent_id])
                        
                        if not (is_start or is_goal) and 0 <= x < width and 0 <= y < height:
                            heatmap[y, x] += 1
                    
                    prev_pos = pos
    
    # Create visualization with fixed positions for spectrum and colorbar
    fig = plt.figure(figsize=(12, 14))
    
    # Fixed positions for spectrum and colorbar
    spectrum_left = 0.1
    spectrum_bottom = 0.1
    spectrum_width = 0.64
    spectrum_height = 0.13
    
    colorbar_left = 0.76
    colorbar_bottom = 0.27
    colorbar_width = 0.03
    colorbar_height = 0.4
    
    # Adjustable heatmap position and size
    heatmap_left = 0.1
    heatmap_bottom = 0.27
    heatmap_width = 0.64
    heatmap_height = 0.4
    
    # Create subplots with fixed positions
    ax_main = fig.add_axes([heatmap_left, heatmap_bottom, heatmap_width, heatmap_height])
    ax_spectrum = fig.add_axes([spectrum_left, spectrum_bottom, spectrum_width, spectrum_height])
    
    # Create combined visualization: obstacles + heatmap
    # Use a custom colormap where obstacles are black and free space shows visit counts
    display_data = np.zeros((height, width))
    
    for i in range(height):
        for j in range(width):
            if grid[i, j] == 1:  # Obstacle
                display_data[i, j] = -1  # Special value for obstacles
            else:  # Free space
                display_data[i, j] = heatmap[i, j]
    
    # Create custom colormap
    import matplotlib.colors as colors
    import matplotlib.cm as cm
    
    # Get the maximum and minimum visit count for scaling
    max_visits = np.max(heatmap)
    min_visits = np.min(heatmap)
    
    # Use provided max_value or calculated max_visits
    color_max = max_value if max_value is not None else max_visits
    # Use provided min_value or calculated min_visits
    color_min = min_value if min_value is not None else min_visits
    
    if color_max > 0: # Check against color_max, not max_visits, to handle explicit 0 max_value
        # Create mask for obstacles
        obstacle_mask = (grid == 1)
        
        # 赤比率を増やしたカラーマップと、matplotlib備え付けの深緑→赤のカラーマップ（'RdYlGn'の逆順）も用意
        from matplotlib.colors import LinearSegmentedColormap
        # 赤比率を増やしたカラーマップ
        colors_list = ['#006400', '#228B22', '#32CD32', '#ADFF2F', '#FFFF00', '#FFD700', '#FFA500', '#FF4500', '#FF0000', '#DC143C', '#B22222']  # dark green to dark red with more red shades
        visit_cmap = LinearSegmentedColormap.from_list('green_to_red', colors_list, N=256)
        # 備え付けの深緑→赤（赤が多い方が大きい値）のカラーマップ
        normal_visit_cmap = plt.get_cmap('RdYlGn_r')
        # normal_visit_cmap = plt.get_cmap('jet')
        
        # Set up normalization (linear or log scale)
        if log_scale and color_max > 1:
            from matplotlib.colors import LogNorm
            # For log scale, replace zeros with a very small value to show as deep green
            log_min = max(0.1, color_min) if color_min > 0 else 0.1
            heatmap_display = np.where(heatmap == 0, log_min, heatmap)  # Use log_min instead of masking
            visit_norm = LogNorm(vmin=log_min, vmax=color_max)
            im = ax_main.imshow(heatmap_display, cmap=normal_visit_cmap, norm=visit_norm, origin='upper')
        else:
            heatmap_display = heatmap
            visit_norm = colors.Normalize(vmin=color_min, vmax=color_max)
            im = ax_main.imshow(heatmap_display, cmap=normal_visit_cmap, norm=visit_norm, origin='upper')
        
        # Overlay obstacles in black
        ax_main.imshow(np.where(obstacle_mask, 1, np.nan), cmap='gray', origin='upper', alpha=1.0)
        
        # Create colorbar with fixed position
        cbar_ax = fig.add_axes([colorbar_left, colorbar_bottom, colorbar_width, colorbar_height])
        cbar = plt.colorbar(im, cax=cbar_ax)
        # label_suffix = ' (Log Scale)' if log_scale else ''
        if heatmap_type == 'visits':
            cbar.set_label(f'Number of Visits', rotation=270, labelpad=20)
        else:  # stops
            cbar.set_label(f'Number of Stops', rotation=270, labelpad=20)
        
        # Set colorbar ticks to meaningful values
        if log_scale and color_max > 1:
            # For log scale, use powers of 10 as ticks, starting from 1
            import math
            max_power = int(math.log10(color_max))
            min_power = int(math.log10(max(1, color_min))) if color_min > 0 else 0
            tick_values = [1] + [10**i for i in range(max(1, min_power), max_power + 1) if 10**i <= color_max]
            if color_max not in tick_values:
                tick_values.append(color_max)
            # Add tick labels to show that 0 visits appear as the lowest color
            tick_labels = ['0 (shown as deep green)'] + [str(v) for v in tick_values[1:]]
            cbar.set_ticks([log_min] + tick_values[1:])
            cbar.set_ticklabels([str(color_min)] + [str(v) for v in tick_values[1:]])
        elif color_max - color_min <= 10:
            tick_values = list(range(color_min if not log_scale else max(1, color_min), color_max + 1))
            cbar.set_ticks(tick_values)
        else:
            if log_scale:
                tick_values = [max(1, color_min), color_min + (color_max - color_min) // 4, color_min + (color_max - color_min) // 2, color_min + 3 * (color_max - color_min) // 4, color_max]
            else:
                tick_values = [color_min, color_min + (color_max - color_min) // 4, color_min + (color_max - color_min) // 2, color_min + 3 * (color_max - color_min) // 4, color_max]
            cbar.set_ticks(tick_values)
        
        # Create congestion spectrum
        # Calculate congestion distribution
        congestion_values = heatmap[grid == 0]  # Only count free cells
        congestion_values = congestion_values[congestion_values > 0]  # Only count cells with visits/stops
        
        if len(congestion_values) > 0:
            # Create histogram of congestion values
            hist_min = max(1, color_min)
            hist, bin_edges = np.histogram(congestion_values, bins=min(50, color_max - hist_min + 1), range=(hist_min, color_max))
            
            # Create histogram with colored bars
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bar_colors = normal_visit_cmap(visit_norm(bin_centers))
            
            # Apply log scale to frequency if requested
            if log_scale:
                # Use original frequency values, let matplotlib handle log scale
                hist_display = hist.astype(float)
                # Set zero frequencies to a very small value for log scale visibility
                hist_display[hist == 0] = 0.1
                ylabel = 'Frequency'
                # For zero frequency bars, use deep green color
                bar_colors_final = []
                for i, h in enumerate(hist):
                    if h == 0:
                        bar_colors_final.append('#006400')  # Deep green for zero frequency
                    else:
                        bar_colors_final.append(bar_colors[i])
            else:
                hist_display = hist
                ylabel = 'Frequency'
                bar_colors_final = bar_colors
            
            ax_spectrum.bar(bin_centers, hist_display, width=bin_edges[1]-bin_edges[0], 
                           color=bar_colors_final, edgecolor='white', linewidth=0.5, alpha=0.8)
            
            # Set spectrum plot properties
            ax_spectrum.set_xlim(max(1, color_min), color_max)
            if log_scale:
                ax_spectrum.set_yscale('log')
                ax_spectrum.set_ylim(0.1, np.max(hist) * 2)  # Use original hist max for limit
            else:
                ax_spectrum.set_ylim(0, np.max(hist_display) * 1.1)
            ax_spectrum.set_xlabel('Number of Visits')
            ax_spectrum.set_ylabel(ylabel)
            
            # Add grid for better readability
            ax_spectrum.grid(True, alpha=0.3, axis='x')
        else:
            # No congestion data, hide spectrum
            ax_spectrum.text(0.5, 0.5, 'No congestion data', ha='center', va='center', 
                           transform=ax_spectrum.transAxes, fontsize=12)
            ax_spectrum.set_xticks([])
            ax_spectrum.set_yticks([])
            
    else:
        # No visits recorded or max_value is 0, just show obstacles in black
        ax_main.imshow(grid, cmap='gray_r', origin='upper')
        ax_spectrum.text(0.5, 0.5, 'No data to visualize', ha='center', va='center', 
                        transform=ax_spectrum.transAxes, fontsize=12)
        ax_spectrum.set_xticks([])
        ax_spectrum.set_yticks([])
    
    
    # Set plot properties for main heatmap
    ax_main.set_xlim(-0.5, width - 0.5)
    ax_main.set_ylim(height - 0.5, -0.5)  # Flip y-axis to match coordinate system
    ax_main.set_aspect('equal')
    
    # Remove all axes, ticks, labels, and title for main plot
    ax_main.set_xticks([])
    ax_main.set_yticks([])
    ax_main.set_xlabel('')
    ax_main.set_ylabel('')
    ax_main.axis('off')
    
    # Spectrum and colorbar positions are already fixed, no manual adjustment needed
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"{heatmap_type.capitalize()} heatmap visualization saved to {output_file}")
    else:
        plt.show()

def create_trajectory_visualization(grid: np.ndarray, data: Dict, width: int, height: int, 
                                   output_file: str = None, show_steps: bool = False):
    """Create visualization of agent trajectories on the map."""
    
    # Colors for different agents (cycling through if more agents than colors)
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Draw map (obstacles in black, free space in white)
    ax.imshow(grid, cmap='gray_r', origin='upper')
    
    # Draw agent trajectories
    for agent_id in range(data['num_agents']):
        color = colors[agent_id % len(colors)]
        
        # Collect all positions for this agent
        agent_path = []
        max_step = max(data['trajectories'].keys()) if data['trajectories'] else 0
        
        for step in range(max_step + 1):
            if step in data['trajectories'] and agent_id in data['trajectories'][step]:
                pos = data['trajectories'][step][agent_id]
                agent_path.append(pos)
        
        if agent_path:
            # Draw trajectory line
            path_x = [pos[0] for pos in agent_path]
            path_y = [pos[1] for pos in agent_path]
            ax.plot(path_x, path_y, color=color, linewidth=2, alpha=0.7, 
                   label=f'Agent {agent_id}')
            
            # Mark start position
            if agent_id < len(data['starts']):
                start_pos = data['starts'][agent_id]
                ax.plot(start_pos[0], start_pos[1], 'o', color=color, 
                       markersize=8, markeredgecolor='black', markeredgewidth=2)
                ax.text(start_pos[0], start_pos[1] - 0.3, f'S{agent_id}', 
                       ha='center', va='top', fontsize=8, fontweight='bold')
            
            # Mark goal position
            if agent_id < len(data['goals']):
                goal_pos = data['goals'][agent_id]
                ax.plot(goal_pos[0], goal_pos[1], 's', color=color, 
                       markersize=8, markeredgecolor='black', markeredgewidth=2)
                ax.text(goal_pos[0], goal_pos[1] + 0.3, f'G{agent_id}', 
                       ha='center', va='bottom', fontsize=8, fontweight='bold')
            
            # Show step numbers along the path if requested
            if show_steps and len(agent_path) > 1:
                for i, pos in enumerate(agent_path[::5]):  # Show every 5th step
                    ax.text(pos[0] + 0.1, pos[1] + 0.1, str(i*5), 
                           fontsize=6, color=color, alpha=0.8)
    
    # Set plot properties
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)  # Flip y-axis to match coordinate system
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Multi-Agent Path Finding Trajectories')
    
    # Add legend
    if data['num_agents'] <= 10 and data['trajectories']:  # Only show legend if not too many agents and trajectories exist
        handles, labels = ax.get_legend_handles_labels()
        if handles:  # Only create legend if there are labeled elements
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add info text
    info_text = f"Agents: {data['num_agents']}\nSteps: {max(data['trajectories'].keys()) if data['trajectories'] else 0}"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_file}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize MAPF agent data')
    parser.add_argument('--result', '-r', required=True, 
                       help='Path to result.txt file')
    parser.add_argument('--map', '-m', required=True, 
                       help='Path to map file')
    parser.add_argument('--output', '-o', 
                       help='Output PDF file prefix (if not specified, display on screen)')
    parser.add_argument('--mode', choices=['trajectory', 'visits', 'stops', 'both'], default='both',
                       help='Visualization mode: trajectory, visits heatmap, stops heatmap, or both heatmaps')
    parser.add_argument('--show-steps', action='store_true',
                       help='Show step numbers along trajectories (trajectory mode only)')
    parser.add_argument('--colorbar-max', type=int,
                       help='Set the maximum value for the colorbar in heatmap modes (e.g., --colorbar-max 100)')
    parser.add_argument('--colorbar-min', type=int,
                       help='Set the minimum value for the colorbar in heatmap modes (e.g., --colorbar-min 0)')
    parser.add_argument('--log-scale', action='store_true',
                       help='Use logarithmic scale for heatmap and spectrum visualization')
    
    args = parser.parse_args()
    
    # Parse input files
    print("Parsing map file...")
    grid, width, height = parse_map_file(args.map)
    
    print("Parsing result file...")
    data = parse_result_file(args.result)

    args.log_scale = True
    
    print(f"Found {data['num_agents']} agents with {len(data['trajectories'])} trajectory steps")
    
    # Create visualization
    if args.mode == 'trajectory':
        print("Creating trajectory visualization...")
        create_trajectory_visualization(grid, data, width, height, 
                                       args.output, args.show_steps)
    elif args.mode == 'visits':
        print("Creating visits heatmap...")
        output_file = f"{args.output}_visits.pdf" if args.output else None
        create_heatmap_visualization(grid, data, width, height, output_file, 'visits', args.colorbar_max, args.colorbar_min, args.log_scale)
    elif args.mode == 'stops':
        print("Creating stops heatmap...")
        output_file = f"{args.output}_stops.pdf" if args.output else None
        create_heatmap_visualization(grid, data, width, height, output_file, 'stops', args.colorbar_max, args.colorbar_min, args.log_scale)
    else:  # both
        print("Creating both heatmaps...")
        if args.output:
            visits_file = f"{args.output}_visits.pdf"
            stops_file = f"{args.output}_stops.pdf"
            create_heatmap_visualization(grid, data, width, height, visits_file, 'visits', args.colorbar_max, args.colorbar_min, args.log_scale)
            create_heatmap_visualization(grid, data, width, height, stops_file, 'stops', args.colorbar_max, args.colorbar_min, args.log_scale)
        else:
            print("Showing visits heatmap...")
            create_heatmap_visualization(grid, data, width, height, None, 'visits', args.colorbar_max, args.colorbar_min, args.log_scale)
            print("Showing stops heatmap...")
            create_heatmap_visualization(grid, data, width, height, None, 'stops', args.colorbar_max, args.colorbar_min, args.log_scale)

if __name__ == "__main__":
    main()