#!/home/initial/aist_ws/lg_lacam/venv/bin/python

import argparse
import sys
import os

# Add the scripts directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from visualization.base_utils import parse_map_file, parse_result_file, setup_matplotlib_style
from visualization.projection_2d import create_2d_projection_visualization
from visualization.debug_3d import create_3d_debug_visualization

def main():
    parser = argparse.ArgumentParser(description='Visualize MAPF agent data (Modular)')
    parser.add_argument('--result', '-r', required=True, 
                       help='Path to result.txt file')
    parser.add_argument('--map', '-m', required=True, 
                       help='Path to map file')
    parser.add_argument('--output', '-o', 
                       help='Output PDF file prefix (if not specified, display on screen)')
    parser.add_argument('--mode', choices=['2d_projection', '3d_debug'], default='2d_projection',
                       help='Visualization mode: 2d_projection or 3d_debug')
    parser.add_argument('--show-steps', action='store_true',
                       help='Show step numbers along trajectories (3d_debug mode only)')
    parser.add_argument('--projection-max-density', type=int,
                       help='Set the maximum density for 2D projection heatmaps (e.g., --projection-max-density 30)')
    parser.add_argument('--projection-min-density', type=int,
                       help='Set the minimum density for 2D projection heatmaps (e.g., --projection-min-density 0)')
    parser.add_argument('--trajectory-max-density', type=int,
                       help='Set the maximum density for trajectory projections (e.g., --trajectory-max-density 20)')
    parser.add_argument('--trajectory-min-density', type=int,
                       help='Set the minimum density for trajectory projections (e.g., --trajectory-min-density 0)')
    parser.add_argument('--debug-max-density', type=int,
                       help='Set the maximum density for 3D debug mode (e.g., --debug-max-density 15)')
    parser.add_argument('--debug-min-density', type=int,
                       help='Set the minimum density for 3D debug mode (e.g., --debug-min-density 0)')
    parser.add_argument('--log-scale', action='store_true',
                       help='Use logarithmic scale for heatmap and spectrum visualization')
    
    args = parser.parse_args()
    
    # Set up matplotlib style
    setup_matplotlib_style()
    
    # Parse input files
    print("Parsing map file...")
    grid, width, height = parse_map_file(args.map)
    
    print("Parsing result file...")
    data = parse_result_file(args.result)

    # Force log scale for better visualization (as in original)
    # args.log_scale = True
    
    print(f"Found {data['num_agents']} agents with {len(data['trajectories'])} trajectory steps")
    
    # Create visualization based on mode
    if args.mode == '2d_projection':
        print("Creating 2D projection visualization...")
        output_file = f"{args.output}_2d_projection.pdf" if args.output else None
        create_2d_projection_visualization(
            grid, data, width, height, output_file,
            args.projection_max_density, args.projection_min_density,
            args.trajectory_max_density, args.trajectory_min_density,
            args.log_scale
        )
    
    elif args.mode == '3d_debug':
        print("Creating 3D debug trajectory visualization...")
        output_file = f"{args.output}_3d_debug.pdf" if args.output else None
        create_3d_debug_visualization(
            grid, data, width, height, output_file, args.show_steps, 
            args.debug_max_density, args.debug_min_density, args.log_scale
        )

if __name__ == "__main__":
    main()