#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
from .base_utils import build_density_table, get_density

def create_2d_projection_visualization(grid: np.ndarray, data: Dict, width: int, height: int, 
                                      output_file: str = None, 
                                      projection_max: int = None, projection_min: int = None, 
                                      trajectory_max: int = None, trajectory_min: int = None, 
                                      log_scale: bool = False):
    """Create 2D projection heatmaps from 3D spatio-temporal density data."""
    
    print("Building density table...")
    density_table, max_step = build_density_table(data, width, height)
    
    # Calculate global density statistics for consistent color mapping
    all_densities = []
    for agent_id in range(data['num_agents']):
        for step in range(max_step + 1):
            if step in data['trajectories'] and agent_id in data['trajectories'][step]:
                pos = data['trajectories'][step][agent_id]
                density = get_density(density_table, step, pos[0], pos[1])
                all_densities.append(density)
    
    # Apply user-specified min/max values or use calculated values
    projection_max_density = projection_max if projection_max is not None else max(all_densities) if all_densities else 1
    projection_min_density = projection_min if projection_min is not None else min(all_densities) if all_densities else 0
    trajectory_max_density = trajectory_max if trajectory_max is not None else max(all_densities) if all_densities else 1
    trajectory_min_density = trajectory_min if trajectory_min is not None else min(all_densities) if all_densities else 0

    print("projection_max_density:", projection_max_density)
    print("projection_min_density:", projection_min_density)
    
    # Create 2D density heatmaps by projecting 3D density onto different planes
    create_projection_heatmaps(grid, density_table, width, height, max_step, output_file, 
                              projection_max_density, projection_min_density, log_scale)
    
    # # Create 2D trajectory projections with density-based coloring
    # create_trajectory_projections(grid, data, width, height, max_step, output_file, 
    #                              density_table, trajectory_max_density, trajectory_min_density, log_scale)

def create_projection_heatmaps(grid: np.ndarray, density_table: dict, width: int, height: int, 
                              max_step: int, output_file: str = None, 
                              max_value: int = None, min_value: int = None, log_scale: bool = False):
    """Create 2D heatmaps by projecting 3D density onto different planes."""
    
    # Project 3D density onto different planes
    projections = {}
    
    # XZ-plane projection (sum across Y dimension)
    xz_projection = np.zeros((max_step + 1, width), dtype=int)
    for t in range(max_step + 1):
        for x in range(width):
            for y in range(height):
                if t in density_table and x in density_table[t] and y in density_table[t][x]:
                    xz_projection[t, x] += density_table[t][x][y]
    projections['xz'] = xz_projection
    
    # YZ-plane projection (sum across X dimension)
    yz_projection = np.zeros((max_step + 1, height), dtype=int)
    for t in range(max_step + 1):
        for x in range(width):
            for y in range(height):
                if t in density_table and x in density_table[t] and y in density_table[t][x]:
                    yz_projection[t, y] += density_table[t][x][y]
    projections['yz'] = yz_projection
    
    # Find global min/max for consistent coloring across all projections
    if max_value is None or min_value is None:
        all_densities = []
        for proj in projections.values():
            all_densities.extend(proj.flatten())
        
        projection_max = max(all_densities) if all_densities else 1
        projection_min = min(all_densities) if all_densities else 0
    else:
        projection_max = max_value
        projection_min = min_value
    
    # Create heatmaps for each projection
    projection_configs = {
        'xz': {
            'data': xz_projection.T,  # Transpose to put time on x-axis
            'title': 'XZ-plane Projection (Y-Aggregated)',
            'xlabel': 'Time Step',
            'ylabel': '',  # Remove y-axis label
            'suffix': '_xz_projection',
            'hide_y_axis': True  # Hide y-axis ticks and labels
        },
        'yz': {
            'data': yz_projection.T,  # Transpose to put time on x-axis
            'title': 'YZ-plane Projection (X-Aggregated)',
            'xlabel': 'Time Step',
            'ylabel': '',  # Remove y-axis label
            'suffix': '_yz_projection',
            'hide_y_axis': True  # Hide y-axis ticks and labels
        }
    }
    
    for plane, config in projection_configs.items():
        proj_data = config['data']
        
        # Create individual projection min/max
        proj_max = np.max(proj_data) if np.max(proj_data) > 0 else projection_max
        proj_min = np.min(proj_data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Set up colormap and normalization
        heatmap_cmap = plt.get_cmap('RdYlGn_r')
        
        if log_scale and proj_max > 1:
            from matplotlib.colors import LogNorm
            log_min = max(0.1, proj_min) if proj_min > 0 else 0.1
            display_data = np.where(proj_data == 0, log_min, proj_data)
            norm = LogNorm(vmin=log_min, vmax=proj_max)
        else:
            display_data = proj_data
            from matplotlib.colors import Normalize
            norm = Normalize(vmin=proj_min, vmax=proj_max)
        
        # Create heatmap
        im = ax.imshow(display_data, cmap=heatmap_cmap, norm=norm, origin='upper', aspect='auto')
        
        # Set plot properties
        ax.set_xlabel(config['xlabel'])
        if not config.get('hide_y_axis', False):
            ax.set_ylabel(config['ylabel'])
        else:
            ax.set_yticks([])
            ax.set_ylabel('')
        
        ax.set_title(config['title'])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Density', rotation=270, labelpad=20)
        
        plt.tight_layout()
        
        if output_file:
            projection_file = f"{output_file.rsplit('.', 1)[0]}{config['suffix']}.pdf"
            plt.savefig(projection_file, dpi=300, bbox_inches='tight')
            print(f"Projection saved to {projection_file}")
        else:
            plt.show()
        
        plt.close(fig)

def create_trajectory_projections(grid: np.ndarray, data: Dict, width: int, height: int, 
                                 max_step: int, output_file: str = None, density_table: dict = None,
                                 global_max_density: int = None, global_min_density: int = None, 
                                 log_scale: bool = False):
    """Create 2D trajectory projections onto different planes with density-based coloring and heatmap colorbar."""
    
    # Set up heatmap-style colormap and normalization
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    heatmap_cmap = plt.get_cmap('RdYlGn_r')
    
    # Set up normalization (linear or log scale)
    if log_scale and global_max_density > 1:
        from matplotlib.colors import LogNorm
        log_min = max(0.1, global_min_density) if global_min_density > 0 else 0.1
        density_norm = LogNorm(vmin=log_min, vmax=global_max_density)
    else:
        density_norm = mcolors.Normalize(vmin=global_min_density, vmax=global_max_density)
    
    # Create projections
    projection_configs = {
        'xz': {
            'xlabel': 'Time Step',
            'ylabel': 'X Coordinate', 
            'title': 'XZ-plane Trajectory Projection (Density-based Coloring)',
            'suffix': '_xz_trajectory'
        },
        'yz': {
            'xlabel': 'Time Step',
            'ylabel': 'Y Coordinate',
            'title': 'YZ-plane Trajectory Projection (Density-based Coloring)', 
            'suffix': '_yz_trajectory'
        }
    }
    
    for plane, config in projection_configs.items():
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Draw agent trajectories with density-based coloring
        for agent_id in range(data['num_agents']):
            # Collect all positions for this agent
            agent_path_t = []
            agent_path_coord = []
            
            for step in range(max_step + 1):
                if step in data['trajectories'] and agent_id in data['trajectories'][step]:
                    pos = data['trajectories'][step][agent_id]
                    agent_path_t.append(step)
                    if plane == 'xz':
                        agent_path_coord.append(pos[0])  # X coordinate
                    else:  # yz
                        agent_path_coord.append(pos[1])  # Y coordinate
            
            if agent_path_t and len(agent_path_t) > 1:
                # Calculate density for each point in the trajectory
                densities = []
                if density_table:
                    for i in range(len(agent_path_t)):
                        step = agent_path_t[i]
                        pos = data['trajectories'][step][agent_id]
                        density = get_density(density_table, step, pos[0], pos[1])
                        densities.append(density)
                else:
                    densities = [1] * len(agent_path_t)  # Default density if no table
                
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
                        alpha = 0.2  # After goal - very faded
                    else:
                        alpha = 0.8  # Before goal or never reached goal
                    
                    if alpha > 0:  # Only draw if visible
                        # Handle log scale density mapping
                        if log_scale and global_max_density > 1:
                            # For log scale, replace zeros with log_min
                            display_density = max(log_min, densities[i]) if densities[i] > 0 else log_min
                        else:
                            display_density = densities[i]
                        
                        # Use normalization for consistent coloring
                        segment_color = heatmap_cmap(density_norm(display_density))
                        
                        # Draw individual segment
                        ax.plot([agent_path_t[i], agent_path_t[i+1]], 
                               [agent_path_coord[i], agent_path_coord[i+1]], 
                               color=segment_color, linewidth=2, alpha=alpha)
                
                # Mark start and end positions with special markers
                if densities:
                    # Start position
                    start_density = densities[0] if densities else 1
                    if log_scale and global_max_density > 1:
                        start_display = max(log_min, start_density) if start_density > 0 else log_min
                    else:
                        start_display = start_density
                    start_color = heatmap_cmap(density_norm(start_display))
                    ax.plot(agent_path_t[0], agent_path_coord[0], 'o', color=start_color, 
                           markersize=8, markeredgecolor='black', markeredgewidth=2)
                    
                    # End position
                    end_density = densities[-1] if densities else 1
                    if log_scale and global_max_density > 1:
                        end_display = max(log_min, end_density) if end_density > 0 else log_min
                    else:
                        end_display = end_density
                    end_color = heatmap_cmap(density_norm(end_display))
                    ax.plot(agent_path_t[-1], agent_path_coord[-1], 's', color=end_color, 
                           markersize=8, markeredgecolor='black', markeredgewidth=2)
        
        # Add colorbar for density visualization
        if global_max_density > global_min_density:
            # Create a scalar mappable for the colorbar using the same normalization
            sm = cm.ScalarMappable(norm=density_norm, cmap=heatmap_cmap)
            sm.set_array([])
            
            # Add colorbar
            cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=20)
            cbar.set_label('Density (Number of Nearby Trajectories)', rotation=270, labelpad=20)
            
            # Set colorbar ticks (same logic as 3d_debug)
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
        ax.set_xlabel(config['xlabel'])
        ax.set_ylabel(config['ylabel'])
        ax.set_title(config['title'])
        ax.grid(True, alpha=0.3)
        
        # Add info text
        info_text = f"Agents: {data['num_agents']}\\nSteps: {max_step}\\nDensity range: {global_min_density}-{global_max_density}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if output_file:
            trajectory_file = f"{output_file.rsplit('.', 1)[0]}{config['suffix']}.pdf"
            plt.savefig(trajectory_file, dpi=300, bbox_inches='tight')
            print(f"Trajectory projection saved to {trajectory_file}")
        else:
            plt.show()
        
        plt.close(fig)