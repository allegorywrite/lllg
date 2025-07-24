#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
from typing import Dict
from .base_utils import build_density_table, get_density

def create_3d_debug_visualization(grid: np.ndarray, data: Dict, width: int, height: int, 
                                 output_file: str = None, show_steps: bool = False, 
                                 max_value: int = None, min_value: int = None, log_scale: bool = False):
    """Create 3D visualization of agent trajectories in spatio-temporal space (debug mode)."""
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    print("Building density table...")
    density_table, max_step = build_density_table(data, width, height)
    print(f"max_step: {max_step}")
    
    # Calculate global density statistics for consistent color mapping
    all_densities = []
    for agent_id in range(data['num_agents']):
        for step in range(max_step + 1):
            if step in data['trajectories'] and agent_id in data['trajectories'][step]:
                pos = data['trajectories'][step][agent_id]
                density = get_density(density_table, step, pos[0], pos[1])
                all_densities.append(density)
    
    # Apply user-specified min/max values or use calculated values
    global_max_density = max_value if max_value is not None else max(all_densities) if all_densities else 1
    global_min_density = min_value if min_value is not None else min(all_densities) if all_densities else 0
    
    # Draw agent trajectories in 3D with density-based coloring
    density_norm, heatmap_cmap = draw_3d_agent_trajectories(
        ax, data, width, height, max_step, density_table, 
        global_max_density, global_min_density, log_scale)
    
    # Add colorbar for density visualization
    if global_max_density > global_min_density:
        # Create a scalar mappable for the colorbar using the same normalization
        sm = cm.ScalarMappable(norm=density_norm, cmap=heatmap_cmap)
        sm.set_array([])
        
        # Add colorbar
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=20)
        cbar.set_label('Density (Number of Nearby Trajectories)', rotation=270, labelpad=20)
        
        # Set colorbar ticks
        if log_scale and global_max_density > 1:
            import math
            max_power = int(math.log10(global_max_density))
            min_power = int(math.log10(max(1, global_min_density))) if global_min_density > 0 else 0
            tick_values = [1] + [10**i for i in range(max(1, min_power), max_power + 1) if 10**i <= global_max_density]
            if global_max_density not in tick_values:
                tick_values.append(global_max_density)
            log_min = max(0.1, global_min_density) if global_min_density > 0 else 0.1
            cbar.set_ticks([log_min] + tick_values[1:])
            cbar.set_ticklabels([str(global_min_density)] + [str(v) for v in tick_values[1:]])
        elif global_max_density - global_min_density <= 10:
            tick_values = list(range(global_min_density if not log_scale else max(1, global_min_density), global_max_density + 1))
            cbar.set_ticks(tick_values)
        else:
            if log_scale:
                tick_values = [max(1, global_min_density), global_min_density + (global_max_density - global_min_density) // 4, global_min_density + (global_max_density - global_min_density) // 2, global_min_density + 3 * (global_max_density - global_min_density) // 4, global_max_density]
            else:
                tick_values = [global_min_density, global_min_density + (global_max_density - global_min_density) // 4, global_min_density + (global_max_density - global_min_density) // 2, global_min_density + 3 * (global_max_density - global_min_density) // 4, global_max_density]
            cbar.set_ticks(tick_values)
    
    # Set plot properties
    ax.set_xlim(0, max_step)
    ax.set_ylim(0, width - 1)
    ax.set_zlim(0, height - 1)
    
    # Invert z-axis to match coordinate system
    ax.invert_zaxis()
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('X Coordinate')
    ax.set_zlabel('Y Coordinate')
    ax.set_title('3D Spatio-Temporal Agent Trajectories (Debug Mode)')
    
    # Remove grid and background
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Make panes transparent
    ax.xaxis.pane.set_alpha(0)
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0)
    
    # Remove grid lines
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    
    # Add info text
    info_text = f"Agents: {data['num_agents']}\\nSteps: {max_step}\\nGrid: {width}x{height}"
    ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set viewing angle for x-z plane view
    ax.view_init(elev=-90, azim=-90)
    
    # Disable perspective projection (use orthographic projection)
    ax.set_proj_type('ortho')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"3D trajectory visualization saved to {output_file}")
    else:
        plt.show()

def draw_3d_agent_trajectories(ax, data: Dict, width: int, height: int, max_step: int, 
                              density_table: dict, global_max_density: int, global_min_density: int, 
                              log_scale: bool = False):
    """Draw agent trajectories in 3D with density-based coloring."""
    
    # Set up heatmap-style colormap (same as in heatmap visualization)
    heatmap_cmap = plt.get_cmap('RdYlGn_r')  # Same as heatmap
    
    # Set up normalization (linear or log scale)
    if log_scale and global_max_density > 1:
        from matplotlib.colors import LogNorm
        # For log scale, ensure minimum is at least 0.1
        log_min = max(0.1, global_min_density) if global_min_density > 0 else 0.1
        density_norm = LogNorm(vmin=log_min, vmax=global_max_density)
    else:
        density_norm = mcolors.Normalize(vmin=global_min_density, vmax=global_max_density)
    
    # Draw agent trajectories in 3D (time, x, y) with density-based coloring
    for agent_id in range(data['num_agents']):
        # Collect all positions for this agent with timestamps
        agent_path_t = []
        agent_path_x = []
        agent_path_y = []
        
        for step in range(max_step + 1):
            if step in data['trajectories'] and agent_id in data['trajectories'][step]:
                pos = data['trajectories'][step][agent_id]
                agent_path_t.append(step)
                agent_path_x.append(pos[0])
                agent_path_y.append(pos[1])
        
        if agent_path_t:
            # Calculate density for each point in the trajectory using pre-calculated table
            densities = []
            for i in range(len(agent_path_t)):
                density = get_density(density_table, agent_path_t[i], agent_path_x[i], agent_path_y[i])
                densities.append(density)
            
            # Find the timestep when agent reaches goal
            goal_reached_step = None
            if agent_id < len(data['goals']):
                goal_pos = data['goals'][agent_id]
                for step in range(max_step + 1):
                    if (step in data['trajectories'] and agent_id in data['trajectories'][step] and 
                        data['trajectories'][step][agent_id] == goal_pos):
                        goal_reached_step = step
                        break
            
            # Draw trajectory segments with density-based coloring
            for i in range(len(agent_path_t) - 1):
                # Determine alpha based on goal reached
                if goal_reached_step is not None and agent_path_t[i] > goal_reached_step:
                    alpha = 0.0  # After goal
                else:
                    alpha = 0.8  # Before goal or never reached goal
                
                if alpha > 0:  # Only draw if visible
                    # Handle log scale density mapping
                    if log_scale and global_max_density > 1:
                        # For log scale, replace zeros with log_min
                        log_min = max(0.1, global_min_density) if global_min_density > 0 else 0.1
                        display_density = max(log_min, densities[i]) if densities[i] > 0 else log_min
                    else:
                        display_density = densities[i]
                    
                    # Use normalization for consistent coloring
                    segment_color = heatmap_cmap(density_norm(display_density))
                    
                    # Draw individual segment
                    ax.plot([agent_path_t[i], agent_path_t[i+1]], 
                           [agent_path_x[i], agent_path_x[i+1]], 
                           [agent_path_y[i], agent_path_y[i+1]], 
                           color=segment_color, linewidth=2, alpha=alpha)
    
    return density_norm, heatmap_cmap