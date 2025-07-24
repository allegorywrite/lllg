#!/home/initial/aist_ws/lg_lacam/venv/bin/python

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import argparse
import re
from typing import List, Tuple, Dict
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestNeighbors

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

def create_heatmap_only(grid: np.ndarray, heatmap: np.ndarray, width: int, height: int, 
                        output_file: str = None, heatmap_type: str = 'visits', max_value: int = None, min_value: int = None, log_scale: bool = False):
    """Create heatmap visualization showing only the heatmap."""
    import matplotlib.colors as colors
    import matplotlib.cm as cm
    
    # Get the maximum and minimum visit count for scaling
    max_visits = np.max(heatmap)
    min_visits = np.min(heatmap)
    
    # Use provided max_value or calculated max_visits
    color_max = max_value if max_value is not None else max_visits
    # Use provided min_value or calculated min_visits
    color_min = min_value if min_value is not None else min_visits
    
    if color_max > 0:
        # Create mask for obstacles
        obstacle_mask = (grid == 1)
        
        # 備え付けの深緑→赤（赤が多い方が大きい値）のカラーマップ
        normal_visit_cmap = plt.get_cmap('RdYlGn_r')
        
        # Set up normalization (linear or log scale)
        if log_scale and color_max > 1:
            from matplotlib.colors import LogNorm
            # For log scale, replace zeros with a very small value to show as deep green
            log_min = max(0.1, color_min) if color_min > 0 else 0.1
            heatmap_display = np.where(heatmap == 0, log_min, heatmap)  # Use log_min instead of masking
            visit_norm = LogNorm(vmin=log_min, vmax=color_max)
        else:
            heatmap_display = heatmap
            visit_norm = colors.Normalize(vmin=color_min, vmax=color_max)
        
        # Create figure with heatmap only
        fig, ax = plt.subplots(figsize=(8, 8))
        
        im = ax.imshow(heatmap_display, cmap=normal_visit_cmap, norm=visit_norm, origin='upper')
        
        # Overlay obstacles in black
        ax.imshow(np.where(obstacle_mask, 1, np.nan), cmap='gray', origin='upper', alpha=1.0)
        
        # Set plot properties
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(height - 0.5, -0.5)  # Flip y-axis to match coordinate system
        # Set fixed aspect ratio for projection graphs
        ax.set_aspect('equal')  # Height is 2 times width (横長にする)
        
        # Remove all axes, ticks, labels, and title
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.axis('off')
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"{heatmap_type.capitalize()} heatmap saved to {output_file}")
        else:
            plt.show()
        
        plt.close(fig)
        return im, visit_norm, normal_visit_cmap
    else:
        # No visits recorded or max_value is 0, just show obstacles in black
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(grid, cmap='gray_r', origin='upper')
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(height - 0.5, -0.5)
        # Set fixed aspect ratio for projection graphs
        ax.set_aspect('equal')  # Height is 2 times width (横長にする)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.axis('off')
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"{heatmap_type.capitalize()} heatmap saved to {output_file}")
        else:
            plt.show()
        
        plt.close(fig)
        return None, None, None

def create_colorbar_only(im, visit_norm, normal_visit_cmap, output_file: str = None, heatmap_type: str = 'visits', 
                        max_value: int = None, min_value: int = None, log_scale: bool = False):
    """Create colorbar visualization only."""
    if im is None:
        return
    
    color_max = max_value if max_value is not None else visit_norm.vmax
    color_min = min_value if min_value is not None else visit_norm.vmin
    
    # Create figure with colorbar only
    fig = plt.figure(figsize=(1, 8))
    
    # Create colorbar
    cbar = plt.colorbar(im, cax=fig.add_axes([0.3, 0.1, 0.4, 0.8]), orientation='vertical')
    
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
        log_min = max(0.1, color_min) if color_min > 0 else 0.1
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
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"{heatmap_type.capitalize()} colorbar saved to {output_file}")
    else:
        plt.show()
    
    plt.close(fig)

def create_histogram_only(grid: np.ndarray, heatmap: np.ndarray, visit_norm, normal_visit_cmap, output_file: str = None, 
                         heatmap_type: str = 'visits', max_value: int = None, min_value: int = None, log_scale: bool = False):
    """Create histogram visualization only."""
    color_max = max_value if max_value is not None else np.max(heatmap)
    color_min = min_value if min_value is not None else np.min(heatmap)
    
    # Calculate congestion distribution
    congestion_values = heatmap[grid == 0]  # Only count free cells
    congestion_values = congestion_values[congestion_values > 0]  # Only count cells with visits/stops
    
    if len(congestion_values) > 0:
        # Create figure with histogram only
        fig, ax = plt.subplots(figsize=(10, 2))
        
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
        
        ax.bar(bin_centers, hist_display, width=bin_edges[1]-bin_edges[0], 
               color=bar_colors_final, edgecolor='white', linewidth=0.5, alpha=1.0)
        
        # Set spectrum plot properties
        ax.set_xlim(max(1, color_min), color_max)
        if log_scale:
            ax.set_yscale('log')
            ax.set_ylim(0.1, np.max(hist) * 2)  # Keep 0.1 for log scale to show zero values
            # Set specific ticks and labels for better readability
            y_max = int(np.max(hist))
            # Show clean intervals: 0, 1, 100, 10000
            tick_positions = [0.1, 1]
            # Add 100 and 10000 if data supports it
            if y_max >= 10:
                tick_positions.append(10)
            if y_max >= 1000:
                tick_positions.append(1000)
            ax.set_yticks(tick_positions)
            # Custom labels: 0.1 -> '0', others as integers
            tick_labels = ['0'] + [str(int(tick)) for tick in tick_positions[1:]]
            ax.set_yticklabels(tick_labels)
        else:
            ax.set_ylim(0, np.max(hist_display) * 1.1)
        ax.set_xlabel('Number of Visits')
        ax.set_ylabel(ylabel)
        
        # Set x-axis ticks to show integer values starting from 1
        x_min = max(1, color_min)
        if color_max - x_min <= 20:
            # For small ranges, show all integer ticks
            x_ticks = list(range(x_min, color_max + 1))
        else:
            # For larger ranges, show nice round numbers
            x_ticks = [x_min]  # Always include minimum
            
            # Add nice round numbers based on range
            if color_max <= 50:
                # For moderate ranges, use multiples of 5 or 10
                step = 5 if color_max <= 100 else 10
                current = ((x_min // step) + 1) * step
                while current < color_max:
                    x_ticks.append(current)
                    current += step
            elif color_max <= 200:
                # Use multiples of 10
                step = 10
                current = ((x_min // step) + 1) * step
                while current < color_max:
                    x_ticks.append(current)
                    current += step
            else:
                # For large ranges, use multiples of 50 or 100
                if color_max <= 500:
                    step = 50
                elif color_max <= 1000:
                    step = 100
                else:
                    step = 200
                current = ((x_min // step) + 1) * step
                while current < color_max:
                    x_ticks.append(current)
                    current += step
            
            # Always include the maximum value
            if color_max not in x_ticks:
                x_ticks.append(color_max)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(int(tick)) for tick in x_ticks])
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, axis='x')
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"{heatmap_type.capitalize()} histogram saved to {output_file}")
        else:
            plt.show()
        
        plt.close(fig)
    else:
        # No congestion data
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No congestion data', ha='center', va='center', 
               transform=ax.transAxes, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"{heatmap_type.capitalize()} histogram saved to {output_file}")
        else:
            plt.show()
        
        plt.close(fig)

def create_original_combined_visualization(grid: np.ndarray, heatmap: np.ndarray, width: int, height: int, 
                                         heatmap_type: str = 'visits', max_value: int = None, min_value: int = None, log_scale: bool = False):
    """Create the original combined visualization for display mode."""
    
    # Original combined visualization for display mode
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
        
        # 備え付けの深緑→赤（赤が多い方が大きい値）のカラーマップ
        normal_visit_cmap = plt.get_cmap('RdYlGn_r')
        
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
                ax_spectrum.set_ylim(0.1, np.max(hist) * 2)  # Keep 0.1 for log scale to show zero values
                # Set specific ticks and labels for better readability
                y_max = int(np.max(hist))
                # Show clean intervals: 0, 1, 100, 10000
                tick_positions = [0.1, 1]
                # Add 100 and 10000 if data supports it
                if y_max >= 10:
                    tick_positions.append(10)
                if y_max >= 1000:
                    tick_positions.append(1000)
                ax_spectrum.set_yticks(tick_positions)
                # Custom labels: 0.1 -> '0', others as integers
                tick_labels = ['0'] + [str(int(tick)) for tick in tick_positions[1:]]
                ax_spectrum.set_yticklabels(tick_labels)
            else:
                ax_spectrum.set_ylim(0, np.max(hist_display) * 1.1)
            ax_spectrum.set_xlabel('Number of Visits')
            ax_spectrum.set_ylabel(ylabel)
            
            # Set x-axis ticks to show integer values starting from 1
            x_min = max(1, color_min)
            if color_max - x_min <= 20:
                # For small ranges, show all integer ticks
                x_ticks = list(range(x_min, color_max + 1))
            else:
                # For larger ranges, show reasonable number of ticks
                num_ticks = min(10, color_max - x_min + 1)
                x_ticks = [int(x_min + i * (color_max - x_min) / (num_ticks - 1)) for i in range(num_ticks)]
                if color_max not in x_ticks:
                    x_ticks.append(color_max)
            ax_spectrum.set_xticks(x_ticks)
            ax_spectrum.set_xticklabels([str(int(tick)) for tick in x_ticks])
            
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
    
    plt.show()

def create_heatmap_visualization(grid: np.ndarray, data: Dict, width: int, height: int, 
                               output_file: str = None, heatmap_type: str = 'visits', max_value: int = None, min_value: int = None, log_scale: bool = False, heatmap_interval: int = None):
    """Create heatmap visualization showing visits or stops with congestion spectrum."""
    
    max_step = max(data['trajectories'].keys()) if data['trajectories'] else 0
    
    # If heatmap_interval is specified, create separate heatmaps for each interval
    if heatmap_interval is not None:
        num_intervals = (max_step // heatmap_interval) + 1
        
        for interval_idx in range(num_intervals):
            start_step = interval_idx * heatmap_interval
            end_step = min((interval_idx + 1) * heatmap_interval - 1, max_step)
            
            # Create heatmap matrix for this interval
            heatmap = np.zeros((height, width), dtype=int)
            
            if heatmap_type == 'visits':
                # Count all visits for each cell within this interval (excluding start and goal positions)
                for step in range(start_step, end_step + 1):
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
                # Count only stops (when agent stays in same position, excluding start and goal) within this interval
                for agent_id in range(data['num_agents']):
                    prev_pos = None
                    for step in range(start_step, end_step + 1):
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
            
            # Create output filename for this interval
            if output_file:
                base_name = output_file.rsplit('.', 1)[0]
                interval_output = f"{base_name}_interval_{start_step}-{end_step}.pdf"
                
                # Create separate visualizations for this interval
                heatmap_file = f"{base_name}_interval_{start_step}-{end_step}_heatmap.pdf"
                im, visit_norm, normal_visit_cmap = create_heatmap_only(grid, heatmap, width, height, 
                                                                       heatmap_file, heatmap_type, max_value, min_value, log_scale)
                
                # Create colorbar only
                if im is not None:
                    colorbar_file = f"{base_name}_interval_{start_step}-{end_step}_colorbar.pdf"
                    create_colorbar_only(im, visit_norm, normal_visit_cmap, colorbar_file, heatmap_type, 
                                       max_value, min_value, log_scale)
                    
                    # Create histogram only
                    histogram_file = f"{base_name}_interval_{start_step}-{end_step}_histogram.pdf"
                    create_histogram_only(grid, heatmap, visit_norm, normal_visit_cmap, histogram_file, 
                                        heatmap_type, max_value, min_value, log_scale)
            else:
                # Display mode - show each interval separately
                print(f"Showing {heatmap_type} heatmap for interval {start_step}-{end_step}...")
                # Use the original combined visualization code for display mode
                create_original_combined_visualization(grid, heatmap, width, height, heatmap_type, max_value, min_value, log_scale)
        
        return
    
    # Original behavior when heatmap_interval is not specified
    # Create heatmap matrix
    heatmap = np.zeros((height, width), dtype=int)
    
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
    
    # Create separate visualizations if output_file is provided
    if output_file:
        # Create base filename without extension
        base_name = output_file.rsplit('.', 1)[0]
        
        # Create heatmap only
        heatmap_file = f"{base_name}_heatmap.pdf"
        im, visit_norm, normal_visit_cmap = create_heatmap_only(grid, heatmap, width, height, 
                                                               heatmap_file, heatmap_type, max_value, min_value, log_scale)
        
        # Create colorbar only
        if im is not None:
            colorbar_file = f"{base_name}_colorbar.pdf"
            create_colorbar_only(im, visit_norm, normal_visit_cmap, colorbar_file, heatmap_type, 
                               max_value, min_value, log_scale)
            
            # Create histogram only
            histogram_file = f"{base_name}_histogram.pdf"
            create_histogram_only(grid, heatmap, visit_norm, normal_visit_cmap, histogram_file, 
                                heatmap_type, max_value, min_value, log_scale)
        
        return
    
    # Original combined visualization for display mode
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
                ax_spectrum.set_ylim(0.1, np.max(hist) * 2)  # Keep 0.1 for log scale to show zero values
                # Set specific ticks and labels for better readability
                y_max = int(np.max(hist))
                # Show clean intervals: 0, 1, 100, 10000
                tick_positions = [0.1, 1]
                # Add 100 and 10000 if data supports it
                if y_max >= 10:
                    tick_positions.append(10)
                if y_max >= 1000:
                    tick_positions.append(1000)
                ax_spectrum.set_yticks(tick_positions)
                # Custom labels: 0.1 -> '0', others as integers
                tick_labels = ['0'] + [str(int(tick)) for tick in tick_positions[1:]]
                ax_spectrum.set_yticklabels(tick_labels)
            else:
                ax_spectrum.set_ylim(0, np.max(hist_display) * 1.1)
            ax_spectrum.set_xlabel('Number of Visits')
            ax_spectrum.set_ylabel(ylabel)
            
            # Set x-axis ticks to show integer values starting from 1
            x_min = max(1, color_min)
            if color_max - x_min <= 20:
                # For small ranges, show all integer ticks
                x_ticks = list(range(x_min, color_max + 1))
            else:
                # For larger ranges, show reasonable number of ticks
                num_ticks = min(10, color_max - x_min + 1)
                x_ticks = [int(x_min + i * (color_max - x_min) / (num_ticks - 1)) for i in range(num_ticks)]
                if color_max not in x_ticks:
                    x_ticks.append(color_max)
            ax_spectrum.set_xticks(x_ticks)
            ax_spectrum.set_xticklabels([str(int(tick)) for tick in x_ticks])
            
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
    # Set fixed aspect ratio for projection graphs
    ax.set_aspect(2.0)  # Height is 2 times width (横長にする)
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

def create_3d_trajectory_visualization(grid: np.ndarray, data: Dict, width: int, height: int, 
                                      output_file: str = None, show_steps: bool = False, 
                                      projection_max_density: int = None, projection_min_density: int = None,
                                      trajectory_max_density: int = None, trajectory_min_density: int = None, 
                                      makespan: int = None, log_scale: bool = False):
    """Create 3D visualization of agent trajectories in spatio-temporal space (time, x, y)."""
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get maximum timestep from data
    data_max_step = max(data['trajectories'].keys()) if data['trajectories'] else 0
    # Use makespan if specified, otherwise use calculated max_step
    if makespan is not None:
        max_step = makespan  # Use makespan as the dimension controller
        print(f"Using makespan: {makespan}, data has {data_max_step} steps")
    else:
        max_step = data_max_step
        print(f"Using data max_step: {max_step}")
    print("max_step: ", max_step)
    
    # Pre-calculate density table for efficient lookup
    radius = 2.0
    
    # Initialize density table: [time][x][y] = density_count
    density_table = {}
    for t in range(max_step + 1):
        density_table[t] = {}
        for x in range(width):
            density_table[t][x] = {}
            for y in range(height):
                density_table[t][x][y] = 0
    
    # Populate density table by incrementing counts for all trajectory points
    print("Building density table...")
    for agent_id in range(data['num_agents']):
        for step in range(min(max_step + 1, data_max_step + 1)):  # Only iterate through available data
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
    
    # Function to get density from pre-calculated table
    def get_density(t, x, y):
        if (t in density_table and x in density_table[t] and y in density_table[t][x]):
            return density_table[t][x][y]
        return 0
    
    # Calculate global density statistics for consistent color mapping
    all_densities = []
    for agent_id in range(data['num_agents']):
        for step in range(min(max_step + 1, data_max_step + 1)):  # Only iterate through available data
            if step in data['trajectories'] and agent_id in data['trajectories'][step]:
                pos = data['trajectories'][step][agent_id]
                density = get_density(step, pos[0], pos[1])
                all_densities.append(density)
    
    # # Apply user-specified min/max values or use calculated values
    # if projection_max_density is None:
    #     projection_max_density = max(all_densities) if all_densities else 1
    # if projection_min_density is None:
    #     projection_min_density = min(all_densities) if all_densities else 0

    if trajectory_max_density is None:
        trajectory_max_density = max(all_densities) if all_densities else 1
    if trajectory_min_density is None:
        trajectory_min_density = min(all_densities) if all_densities else 0
    
    # Create 2D density heatmaps by projecting 3D density onto different planes
    create_projection_heatmaps(grid, density_table, width, height, max_step, output_file, 
                              projection_max_density, projection_min_density, log_scale)
    
    # Create 2D trajectory projections with density-based coloring
    create_trajectory_projections(grid, data, width, height, max_step, output_file, 
                                 density_table, trajectory_max_density, trajectory_min_density, log_scale, data_max_step)

def create_projection_heatmaps(grid: np.ndarray, density_table: dict, width: int, height: int, 
                              max_step: int, output_file: str = None, 
                              projection_max_density: int = None, projection_min_density: int = None, log_scale: bool = False):
    """Create 2D heatmaps by projecting 3D density onto different planes."""
    
    # Project 3D density onto different planes
    projections = {}
    
    # # 1. XY-plane projection (sum across time dimension)
    # xy_projection = np.zeros((height, width), dtype=int)
    # for t in range(max_step + 1):
    #     for x in range(width):
    #         for y in range(height):
    #             if t in density_table and x in density_table[t] and y in density_table[t][x]:
    #                 xy_projection[y, x] += density_table[t][x][y]
    # projections['xy'] = xy_projection
    
    # 2. XZ-plane projection (sum across Y dimension)
    xz_projection = np.zeros((max_step + 1, width), dtype=int)
    for t in range(max_step + 1):
        for x in range(width):
            for y in range(height):
                if t in density_table and x in density_table[t] and y in density_table[t][x]:
                    xz_projection[t, x] += density_table[t][x][y]
    projections['xz'] = xz_projection
    
    # 3. YZ-plane projection (sum across X dimension)
    yz_projection = np.zeros((max_step + 1, height), dtype=int)
    for t in range(max_step + 1):
        for x in range(width):
            for y in range(height):
                if t in density_table and x in density_table[t] and y in density_table[t][x]:
                    yz_projection[t, y] += density_table[t][x][y]
    projections['yz'] = yz_projection
    
    # Find global min/max for consistent coloring across all projections
    # Only use global min/max if user didn't specify colorbar limits
    if projection_max_density is None or projection_min_density is None:
        all_densities = []
        for proj in projections.values():
            all_densities.extend(proj.flatten())
        
        projection_max = max(all_densities) if all_densities else 1
        projection_min = min(all_densities) if all_densities else 0
    else:
        projection_max = projection_max_density
        projection_min = projection_min_density
    
    # Create heatmaps for each projection
    projection_configs = {
        # 'xy': {
        #     'data': xy_projection,
        #     'title': 'XY-plane Projection (Time-Aggregated)',
        #     'xlabel': 'X Coordinate',
        #     'ylabel': 'Y Coordinate',
        #     'suffix': '_xy_projection',
        #     'obstacles': grid  # Only XY plane has obstacles
        # },
        # 'xz': {
        #     'data': xz_projection.T,  # Transpose to put time on x-axis
        #     'title': 'XZ-plane Projection (Y-Aggregated)',
        #     'xlabel': 'Time Step',
        #     'ylabel': '',  # Remove y-axis label
        #     'suffix': '_xz_projection',
        #     'obstacles': None,
        #     'invert_y': False,  # Invert y-axis for image coordinate system
        #     'hide_y_axis': True  # Hide y-axis ticks and labels
        # },
        'yz': {
            'data': yz_projection.T,  # Transpose to put time on x-axis
            'title': 'YZ-plane Projection (X-Aggregated)',
            'xlabel': '',
            'ylabel': '',  # Remove y-axis label
            'suffix': '_yz_projection',
            'obstacles': None,
            'invert_y': False,  # Invert y-axis for image coordinate system
            'hide_y_axis': True  # Hide y-axis ticks and labels
        }
    }
    
    for plane, config in projection_configs.items():
        # Use individual projection min/max if user didn't specify limits
        if projection_max_density is None or projection_min_density is None:
            proj_data = config['data']
            proj_max = projection_max_density if projection_max_density is not None else np.max(proj_data)
            proj_min = projection_min_density if projection_min_density is not None else np.min(proj_data)
        else:
            proj_max = projection_max_density
            proj_min = projection_min_density
        
        print("proj_max:", proj_max, " proj_min:", proj_min)
        
        create_single_projection_heatmap(
            config['data'], config['title'], config['xlabel'], config['ylabel'],
            config['obstacles'], output_file, config['suffix'],
            proj_max, proj_min, log_scale, config.get('invert_y', False),
            config.get('hide_y_axis', False)
        )

def create_single_projection_heatmap(projection_data: np.ndarray, title: str, xlabel: str, ylabel: str,
                                   obstacles: np.ndarray = None, output_file: str = None, suffix: str = '',
                                   proj_max_density: int = None, proj_min_density: int = None, log_scale: bool = False, invert_y: bool = False, hide_y_axis: bool = False):
    """Create a single projection heatmap."""
    
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    
    # Use same colormap as other heatmaps
    heatmap_cmap = plt.get_cmap('RdYlGn_r')
    
    # Set up normalization (linear or log scale)
    if log_scale and proj_max_density > 1:
        from matplotlib.colors import LogNorm
        log_min = max(0.1, proj_min_density) if proj_min_density > 0 else 0.1
        heatmap_display = np.where(projection_data == 0, log_min, projection_data)
        density_norm = LogNorm(vmin=log_min, vmax=proj_max_density)
    else:
        heatmap_display = projection_data
        density_norm = mcolors.Normalize(vmin=proj_min_density, vmax=proj_max_density)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Create heatmap
    im = ax.imshow(heatmap_display, cmap=heatmap_cmap, norm=density_norm, origin='upper')
    
    # Overlay obstacles (only for XY plane)
    if obstacles is not None:
        obstacle_mask = (obstacles == 1)
        ax.imshow(np.where(obstacle_mask, 1, np.nan), cmap='gray', origin='upper', alpha=1.0)
    
    # Set plot properties
    ax.set_xlim(-0.5, projection_data.shape[1] - 0.5)
    if invert_y:
        # For inverted y-axis, set limits normally and then invert
        ax.set_ylim(-0.5, projection_data.shape[0] - 0.5)
    else:
        ax.set_ylim(projection_data.shape[0] - 0.5, -0.5)
    # Set fixed aspect ratio for projection graphs
    ax.set_aspect(2.0)  # Height is 2 times width (横長にする)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # ax.set_title(title)
    
    # Hide y-axis ticks and labels if requested
    if hide_y_axis:
        ax.set_yticks([])
        ax.set_yticklabels([])
    
    # Store colorbar data for separate export
    colorbar_data = (im, density_norm, proj_max_density, proj_min_density, log_scale)
    
    plt.tight_layout()
    
    # Save heatmap
    if output_file:
        base_name = output_file.rsplit('_3d.pdf', 1)[0] if '_3d.pdf' in output_file else output_file.rsplit('.pdf', 1)[0]
        heatmap_output = f"{base_name}{suffix}.pdf"
        plt.savefig(heatmap_output, dpi=300, bbox_inches='tight')
        print(f"Projection heatmap saved to {heatmap_output}")
        
        # Save colorbar as separate PDF
        colorbar_output = f"{base_name}{suffix}_colorbar.pdf"
        create_projection_colorbar_only(colorbar_data, colorbar_output)
    else:
        plt.show()
    
    plt.close(fig)

def create_trajectory_projections(grid: np.ndarray, data: Dict, width: int, height: int, 
                                 max_step: int, output_file: str = None, density_table: dict = None,
                                 trajectory_max_density: int = None, trajectory_min_density: int = None, 
                                 log_scale: bool = False, data_max_step: int = None):
    """Create 2D trajectory projections onto different planes with density-based coloring."""
    
    # Set up heatmap-style colormap and normalization (same as 3D visualization)
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    heatmap_cmap = plt.get_cmap('RdYlGn_r')
    
    # Set up normalization (linear or log scale)
    if log_scale and trajectory_max_density > 1:
        from matplotlib.colors import LogNorm
        log_min = max(0.1, trajectory_min_density) if trajectory_min_density > 0 else 0.1
        density_norm = LogNorm(vmin=log_min, vmax=trajectory_max_density)
    else:
        density_norm = mcolors.Normalize(vmin=trajectory_min_density, vmax=trajectory_max_density)
    
    # Function to get density from pre-calculated table
    def get_density(t, x, y):
        if (density_table and t in density_table and x in density_table[t] and y in density_table[t][x]):
            return density_table[t][x][y]
        return 0
    
    # Projection configurations
    projection_configs = {
        # 'xy': {
        #     'title': 'XY-plane Trajectory Projection (Time-Aggregated)',
        #     'xlabel': 'X Coordinate',
        #     'ylabel': 'Y Coordinate',
        #     'suffix': '_xy_trajectory',
        #     'obstacles': grid,
        #     'xlim': (0, width - 1),
        #     'ylim': (-2, height + 1),
        #     'invert_y': True
        # },
        # 'xz': {
        #     'title': 'XZ-plane Trajectory Projection (Y-Aggregated)',
        #     'xlabel': 'Time Step',
        #     'ylabel': '',  # Remove y-axis label
        #     'suffix': '_xz_trajectory',
        #     'obstacles': None,
        #     'xlim': (0, max_step),
        #     'ylim': (-2, width + 1),
        #     'invert_y': True,  # Invert y-axis for image coordinate system
        #     'hide_y_axis': True  # Hide y-axis ticks and labels
        # },
        'yz': {
            'title': 'YZ-plane Trajectory Projection (X-Aggregated)',
            'xlabel': 'Time Step',
            'ylabel': '',  # Remove y-axis label
            'suffix': '_yz_trajectory',
            'obstacles': None,
            'xlim': (0, max_step),
            'ylim': (-2, height + 1),
            'invert_y': True,  # Invert y-axis for image coordinate system
            'hide_y_axis': True  # Hide y-axis ticks and labels
        }
    }
    
    for plane, config in projection_configs.items():
        create_single_trajectory_projection(
            data, plane, config['title'], config['xlabel'], config['ylabel'],
            config['obstacles'], output_file, config['suffix'],
            config['xlim'], config['ylim'], config['invert_y'],
            width, height, max_step, heatmap_cmap, density_norm, get_density, log_scale,
            config.get('hide_y_axis', False), data_max_step
        )

def create_single_trajectory_projection(data: Dict, plane: str, title: str, xlabel: str, ylabel: str,
                                       obstacles: np.ndarray = None, output_file: str = None, suffix: str = '',
                                       xlim: tuple = None, ylim: tuple = None, invert_y: bool = False,
                                       width: int = None, height: int = None, max_step: int = None, 
                                       heatmap_cmap = None, density_norm = None, get_density = None, log_scale: bool = False,
                                       hide_y_axis: bool = False, data_max_step: int = None):
    """Create a single trajectory projection with density-based coloring."""
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Draw obstacles if applicable (only for XY plane)
    if obstacles is not None:
        ax.imshow(obstacles, cmap='gray_r', origin='upper', alpha=0.3)
    
    # Draw agent trajectories with density-based coloring
    for agent_id in range(data['num_agents']):
        
        # Collect trajectory points and calculate densities for this agent
        trajectory_points = []
        trajectory_densities = []
        trajectory_steps = []
        
        for step in range(min(max_step + 1, data_max_step + 1) if data_max_step is not None else max_step + 1):
            if step in data['trajectories'] and agent_id in data['trajectories'][step]:
                pos = data['trajectories'][step][agent_id]
                
                # Get density for this point
                density = get_density(step, pos[0], pos[1]) if get_density else 0
                
                # Project to appropriate plane
                if plane == 'xy':
                    # XY plane: (x, y) coordinates
                    trajectory_points.append((pos[0], pos[1]))
                elif plane == 'xz':
                    # XZ plane: (time, x) coordinates (time on x-axis)
                    trajectory_points.append((step, pos[0]))
                elif plane == 'yz':
                    # YZ plane: (time, y) coordinates (time on x-axis)
                    trajectory_points.append((step, pos[1]))
                
                trajectory_densities.append(density)
                trajectory_steps.append(step)
        
        if trajectory_points:
            # Extract x and y coordinates
            x_coords = [p[0] for p in trajectory_points]
            y_coords = [p[1] for p in trajectory_points]
            
            # Find goal reached step for this agent
            goal_reached_step = None
            if agent_id < len(data['goals']):
                goal_pos = data['goals'][agent_id]
                for step in range(min(max_step + 1, data_max_step + 1) if data_max_step is not None else max_step + 1):
                    if (step in data['trajectories'] and agent_id in data['trajectories'][step] and 
                        data['trajectories'][step][agent_id] == goal_pos):
                        goal_reached_step = step
                        break
            
            # Draw trajectory segments with density-based coloring
            for i in range(len(trajectory_points) - 1):
                current_step = trajectory_steps[i]
                
                # Determine alpha based on goal reached
                if goal_reached_step is not None and current_step > goal_reached_step:
                    alpha = 0.0  # After goal - invisible (same as 3D)
                else:
                    alpha = 0.8  # Before goal or never reached goal
                
                if alpha > 0:
                    # Get density for color mapping
                    density = trajectory_densities[i]
                    
                    # Handle log scale density mapping
                    if log_scale and density_norm.vmax > 1:
                        log_min = max(0.1, density_norm.vmin) if density_norm.vmin > 0 else 0.1
                        display_density = max(log_min, density) if density > 0 else log_min
                    else:
                        display_density = density
                    
                    # Use heatmap colormap for consistent coloring
                    segment_color = heatmap_cmap(density_norm(display_density))
                    
                    # Adjust alpha based on density (make green/low-density trajectories more transparent)
                    # Use the normalized density value to calculate alpha
                    normalized_density = density_norm(display_density)
                    # Scale alpha: green (low density) becomes more transparent, red (high density) stays opaque
                    density_alpha = max(0.1, min(1.0, 0.0 + 1.0 * normalized_density))
                    final_alpha = alpha * density_alpha
                    
                    # Draw individual segment
                    ax.plot([x_coords[i], x_coords[i+1]], [y_coords[i], y_coords[i+1]], 
                           color=segment_color, linewidth=2, alpha=final_alpha)
    
    # Colorbar will be saved as separate PDF
    # Create colorbar data for separate export
    colorbar_data = None
    if heatmap_cmap and density_norm:
        import matplotlib.cm as cm
        sm = cm.ScalarMappable(norm=density_norm, cmap=heatmap_cmap)
        sm.set_array([])
        colorbar_data = (sm, heatmap_cmap, density_norm, log_scale)
    
    # Set plot properties
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if invert_y:
        ax.invert_yaxis()
    
    # Set fixed aspect ratio for projection graphs
    ax.set_aspect(2.0)  # Height is 2 times width (横長にする)
    # ax.set_xlabel(xlabel)
    # ax.set_ylabel(ylabel)
    # ax.set_title(title)
    
    # Remove graph frame (spines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Remove y-axis ticks and labels completely
    ax.set_yticks([])
    ax.set_yticklabels([])
    
    # Remove grid completely
    ax.grid(False)
    
    # Add info text
    info_text = f"Agents: {data['num_agents']}\nSteps: {max_step}"
    if width and height:
        info_text += f"\nGrid: {width}x{height}"
    # ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
    #        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save trajectory projection
    if output_file:
        base_name = output_file.rsplit('_3d.pdf', 1)[0] if '_3d.pdf' in output_file else output_file.rsplit('.pdf', 1)[0]
        trajectory_output = f"{base_name}{suffix}.pdf"
        plt.savefig(trajectory_output, dpi=300, bbox_inches='tight')
        print(f"Trajectory projection saved to {trajectory_output}")
        
        # Save colorbar as separate PDF
        if colorbar_data:
            colorbar_output = f"{base_name}{suffix}_colorbar.pdf"
            create_trajectory_colorbar_only(colorbar_data, colorbar_output)
    else:
        plt.show()
    
    plt.close(fig)

def create_projection_colorbar_only(colorbar_data, output_file):
    """Create colorbar-only visualization for projection heatmaps."""
    im, density_norm, proj_max_density, proj_min_density, log_scale = colorbar_data
    
    # Create figure with colorbar only
    fig = plt.figure(figsize=(1, 8))
    
    # Create colorbar
    cbar = plt.colorbar(im, cax=fig.add_axes([0.3, 0.1, 0.4, 0.8]), orientation='vertical')
    cbar.set_label('Projected Density', rotation=270, labelpad=20)
    
    # Set colorbar ticks (same logic as before)
    if log_scale and proj_max_density > 1:
        import math
        log_min = max(0.1, proj_min_density) if proj_min_density > 0 else 0.1
        max_power = int(math.log10(proj_max_density))
        min_power = int(math.log10(max(1, proj_min_density))) if proj_min_density > 0 else 0
        tick_values = [1] + [10**i for i in range(max(1, min_power), max_power + 1) if 10**i <= proj_max_density]
        if proj_max_density not in tick_values:
            tick_values.append(proj_max_density)
        cbar.set_ticks([log_min] + tick_values[1:])
        cbar.set_ticklabels([str(proj_min_density)] + [str(v) for v in tick_values[1:]])
    elif proj_max_density - proj_min_density <= 10:
        tick_values = list(range(proj_min_density if not log_scale else max(1, proj_min_density), proj_max_density + 1))
        cbar.set_ticks(tick_values)
    else:
        if log_scale:
            tick_values = [max(1, proj_min_density), proj_min_density + (proj_max_density - proj_min_density) // 4, proj_min_density + (proj_max_density - proj_min_density) // 2, proj_min_density + 3 * (proj_max_density - proj_min_density) // 4, proj_max_density]
        else:
            tick_values = [proj_min_density, proj_min_density + (proj_max_density - proj_min_density) // 4, proj_min_density + (proj_max_density - proj_min_density) // 2, proj_min_density + 3 * (proj_max_density - proj_min_density) // 4, proj_max_density]
        cbar.set_ticks(tick_values)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Projection colorbar saved to {output_file}")
    else:
        plt.show()
    
    plt.close(fig)

def create_trajectory_colorbar_only(colorbar_data, output_file):
    """Create colorbar-only visualization for trajectory projections."""
    sm, heatmap_cmap, density_norm, log_scale = colorbar_data
    
    # Create figure with colorbar only
    fig = plt.figure(figsize=(1, 8))
    
    # Create colorbar
    cbar = plt.colorbar(sm, cax=fig.add_axes([0.3, 0.1, 0.4, 0.8]), orientation='vertical')
    cbar.set_label('Density (Number of Nearby Trajectories)', rotation=270, labelpad=20)
    
    # Use custom formatter to avoid scientific notation
    if log_scale:
        # Set specific tick values and labels
        
        # Get the current colorbar limits
        vmin, vmax = density_norm.vmin, density_norm.vmax
        
        # Create reasonable tick values for log scale
        if vmax > vmin:
            # Generate tick values that are nice round numbers
            tick_values = []
            current = max(1, int(vmin))
            while current <= vmax:
                tick_values.append(current)
                if current < 10:
                    current += 1
                elif current < 100:
                    current += 10
                else:
                    current += 100
            
            # Add the max value if it's not already included
            if vmax not in tick_values:
                tick_values.append(int(vmax))
            
            # Set the ticks and labels
            cbar.set_ticks(tick_values)
            cbar.set_ticklabels([str(int(v)) for v in tick_values])
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Trajectory colorbar saved to {output_file}")
    else:
        plt.show()
    
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='Visualize MAPF agent data')
    parser.add_argument('--result', '-r', required=True, 
                       help='Path to result.txt file')
    parser.add_argument('--map', '-m', required=True, 
                       help='Path to map file')
    parser.add_argument('--output', '-o', 
                       help='Output PDF file prefix (if not specified, display on screen)')
    parser.add_argument('--mode', choices=['trajectory', 'trajectory3d', 'visits', 'stops', 'both'], default='both',
                       help='Visualization mode: trajectory, trajectory3d, visits heatmap, stops heatmap, or both heatmaps')
    parser.add_argument('--show-steps', action='store_true',
                       help='Show step numbers along trajectories (trajectory mode only)')
    parser.add_argument('--heatmap-max', type=int,
                       help='Set the maximum value for the colorbar in heatmap modes (e.g., --heatmap-max 100)')
    parser.add_argument('--heatmap-min', type=int,
                       help='Set the minimum value for the colorbar in heatmap modes (e.g., --heatmap-min 0)')
    parser.add_argument('--projection-max-density', type=int,
                       help='Set the maximum density value for projection heatmaps (e.g., --projection-max-density 50)')
    parser.add_argument('--projection-min-density', type=int,
                       help='Set the minimum density value for projection heatmaps (e.g., --projection-min-density 0)')
    parser.add_argument('--trajectory-max-density', type=int,
                       help='Set the maximum density value for trajectory visualizations (e.g., --trajectory-max-density 30)')
    parser.add_argument('--trajectory-min-density', type=int,
                       help='Set the minimum density value for trajectory visualizations (e.g., --trajectory-min-density 0)')
    parser.add_argument('--makespan', type=int,
                       help='Set the maximum time step for trajectory3d visualization (e.g., --makespan 100)')
    parser.add_argument('--log-scale', action='store_true',
                       help='Use logarithmic scale for heatmap and spectrum visualization')
    parser.add_argument('--heatmap-interval', type=int,
                       help='Create separate heatmaps for each time interval (e.g., --heatmap-interval 50)')
    
    args = parser.parse_args()
    
    # Parse input files
    print("Parsing map file...")
    grid, width, height = parse_map_file(args.map)
    
    print("Parsing result file...")
    data = parse_result_file(args.result)

    # args.log_scale = True
    
    print(f"Found {data['num_agents']} agents with {len(data['trajectories'])} trajectory steps")
    
    # Create visualization
    if args.mode == 'trajectory':
        print("Creating trajectory visualization...")
        create_trajectory_visualization(grid, data, width, height, 
                                       args.output, args.show_steps)
    elif args.mode == 'trajectory3d':
        print("Creating 3D trajectory visualization...")
        output_file = f"{args.output}_3d.pdf" if args.output else None
        create_3d_trajectory_visualization(grid, data, width, height, 
                                          output_file, args.show_steps, 
                                          args.projection_max_density, args.projection_min_density, 
                                          args.trajectory_max_density, args.trajectory_min_density, 
                                          args.makespan, args.log_scale)
    elif args.mode == 'visits':
        print("Creating visits heatmap...")
        output_file = f"{args.output}_visits.pdf" if args.output else None
        create_heatmap_visualization(grid, data, width, height, output_file, 'visits', args.heatmap_max, args.heatmap_min, args.log_scale, getattr(args, 'heatmap_interval', None))
    elif args.mode == 'stops':
        print("Creating stops heatmap...")
        output_file = f"{args.output}_stops.pdf" if args.output else None
        create_heatmap_visualization(grid, data, width, height, output_file, 'stops', args.heatmap_max, args.heatmap_min, args.log_scale, getattr(args, 'heatmap_interval', None))
    else:  # both
        print("Creating both heatmaps...")
        if args.output:
            visits_file = f"{args.output}_visits.pdf"
            stops_file = f"{args.output}_stops.pdf"
            create_heatmap_visualization(grid, data, width, height, visits_file, 'visits', args.heatmap_max, args.heatmap_min, args.log_scale, getattr(args, 'heatmap_interval', None))
            create_heatmap_visualization(grid, data, width, height, stops_file, 'stops', args.heatmap_max, args.heatmap_min, args.log_scale, getattr(args, 'heatmap_interval', None))
        else:
            print("Showing visits heatmap...")
            create_heatmap_visualization(grid, data, width, height, None, 'visits', args.heatmap_max, args.heatmap_min, args.log_scale, getattr(args, 'heatmap_interval', None))
            print("Showing stops heatmap...")
            create_heatmap_visualization(grid, data, width, height, None, 'stops', args.heatmap_max, args.heatmap_min, args.log_scale, getattr(args, 'heatmap_interval', None))

if __name__ == "__main__":
    main()