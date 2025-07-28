#!/home/initial/aist_ws/lg_lacam/venv/bin/python

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from collections import defaultdict

# Set style for publication-quality plots
plt.rcParams.update({
    'font.size': 24,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'serif'],
    'axes.linewidth': 2.5,
    'axes.edgecolor': 'black',
    'axes.facecolor': 'white',
    # 'axes.grid': True,
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
    'ytick.labelsize': 40,
    'legend.frameon': True,
    'legend.fancybox': False,
    'legend.edgecolor': 'black',
    'legend.facecolor': 'white',
    'legend.framealpha': 1.0,
    'legend.fontsize': 24,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

def create_scenario_bar_chart(csv_file, methods=None, output_file=None, normalize=False, metric='flow_time_ratio', each_map=False):
    """
    Create bar charts comparing different methods for each scenario.
    
    Args:
        csv_file: Path to the benchmark_results_all.csv file
        methods: List of methods to include in the chart (default: all available methods)
        output_file: Output file path for the chart (default: show plot)
        normalize: If True, normalize by lacam's metric value (default: False)  
        metric: Metric to compare ('flow_time_ratio' or 'runtime') (default: 'flow_time_ratio')
    """
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Get available methods if not specified
    if methods is None:
        methods = df['algorithm'].unique().tolist()
        methods = [m for m in methods if m != 'algorithm']  # Remove header if present
    
    # Filter data for specified methods
    df_filtered = df[df['algorithm'].isin(methods)]
    
    # Group by scenario and agents (and optionally by map)
    scenario_data = defaultdict(dict)
    
    for _, row in df_filtered.iterrows():
        scenario = row['scenario']
        algorithm = row['algorithm']
        agents = row['agents']
        map_name = row['map']
        metric_value = row[metric]

        if pd.notna(metric_value):
            if each_map:
                # Group by map, scenario, and agents
                scenario_agents_key = f"{map_name}_{scenario}_{agents}agents"
            else:
                # Group by scenario and agents (ignoring map names)
                scenario_agents_key = f"{scenario}_{agents}agents"
            
            if scenario_agents_key not in scenario_data:
                scenario_data[scenario_agents_key] = {}
            if algorithm not in scenario_data[scenario_agents_key]:
                scenario_data[scenario_agents_key][algorithm] = []
            scenario_data[scenario_agents_key][algorithm].append(metric_value)
    
    # Calculate average metric value for each scenario and algorithm
    avg_data = {}
    for scenario in scenario_data:
        avg_data[scenario] = {}
        for algorithm in scenario_data[scenario]:
            avg_data[scenario][algorithm] = np.mean(scenario_data[scenario][algorithm])
    
    # Normalize metric value by lacam's value for each scenario if requested
    if normalize:
        normalized_avg_data = {}
        for scenario in avg_data:
            normalized_avg_data[scenario] = {}
            lacam_ratio = avg_data[scenario].get('lacam', 1.0)  # Use 1.0 as fallback if lacam not found
            for algorithm in avg_data[scenario]:
                normalized_avg_data[scenario][algorithm] = avg_data[scenario][algorithm] / lacam_ratio
        
        avg_data = normalized_avg_data
    
    # Filter scenarios that have data for all specified methods
    filtered_avg_data = {}
    for scenario in avg_data:
        if all(method in avg_data[scenario] for method in methods):
            filtered_avg_data[scenario] = avg_data[scenario]
    
    # Print statistics for each method when normalize is enabled
    if normalize:
        print("\n=== Statistics for each method ===")
        method_values = {method: [] for method in methods}
        
        # Collect all values for each method
        for scenario in filtered_avg_data:
            for method in methods:
                if method in filtered_avg_data[scenario]:
                    method_values[method].append(filtered_avg_data[scenario][method])
        
        # Calculate and print statistics
        for method in methods:
            if method_values[method]:
                values = method_values[method]
                mean_val = np.mean(values)
                max_val = np.max(values)
                min_val = np.min(values)
                std_val = np.std(values)
                count = len(values)
                
                print(f"{method}:")
                print(f"  Count: {count}")
                print(f"  Mean:  {mean_val:.4f}")
                print(f"  Max:   {max_val:.4f}")
                print(f"  Min:   {min_val:.4f}")
                print(f"  Std:   {std_val:.4f}")
                print()
    
    # Sort scenarios by sorting criteria
    if normalize:
        # When normalized, sort by lg_lacam ratio in ascending order
        scenario_lg_ratios = {}
        for scenario in filtered_avg_data:
            if filtered_avg_data[scenario] and 'lg_lacam' in filtered_avg_data[scenario]:
                scenario_lg_ratios[scenario] = filtered_avg_data[scenario]['lg_lacam']
            else:
                scenario_lg_ratios[scenario] = float('inf')  # Put scenarios without lg_lacam at the end
        scenarios = sorted(filtered_avg_data.keys(), key=lambda x: scenario_lg_ratios[x], reverse=False)
    else:
        if metric == 'runtime':
            # For runtime, sort by lg_lacam runtime values in descending order (larger runtime first)
            scenario_lg_ratios = {}
            for scenario in filtered_avg_data:
                if filtered_avg_data[scenario] and 'lg_lacam' in filtered_avg_data[scenario]:
                    scenario_lg_ratios[scenario] = filtered_avg_data[scenario]['lg_lacam']
                else:
                    scenario_lg_ratios[scenario] = 0  # Put scenarios without lg_lacam at the end
            scenarios = sorted(filtered_avg_data.keys(), key=lambda x: scenario_lg_ratios[x], reverse=False)
        else:
            # Sort scenarios by maximum metric value (ascending order)
            scenario_max_ratios = {}
            for scenario in filtered_avg_data:
                if filtered_avg_data[scenario]:
                    scenario_max_ratios[scenario] = max(filtered_avg_data[scenario].values())
                else:
                    scenario_max_ratios[scenario] = 0
            scenarios = sorted(filtered_avg_data.keys(), key=lambda x: scenario_max_ratios[x], reverse=False)
    avg_data = filtered_avg_data
    n_scenarios = len(scenarios)
    
    if n_scenarios == 0:
        print("No data found for the specified methods.")
        return

    if each_map:
        # Group scenarios by map for subplot creation
        map_scenarios = defaultdict(list)
        for scenario in scenarios:
            if each_map and '_' in scenario:
                map_name = scenario.split('_')[0]
                map_scenarios[map_name].append(scenario)
            else:
                map_scenarios['all'].append(scenario)
        
        # Create subplots for each map
        n_maps = len(map_scenarios)
        fig, axes = plt.subplots(n_maps, 1, figsize=(10, 6 * n_maps), squeeze=False)
        axes = axes.flatten()
    else:
        # Set up the plot (make it taller but reasonable size)
        # fig, ax = plt.subplots(figsize=(max(12, n_scenarios * 0.8), 12))
        fig, ax = plt.subplots(figsize=(30, 3))
        axes = [ax]
        map_scenarios = {'all': scenarios}
    
    # Set width of bars and positions (no gaps between scenarios)
    bar_width = 1.0
    
    # Fixed colors for each method to maintain consistency
    method_colors = {
        'lg_lacam': '#0000FF',     # Blue
        'gg_lacam': '#ff66c4',     # Pink
        'lacam': '#008000',        # Green
        'lg&gg_lacam': '#FDD017',  # Yellow

        # 'lacam3': '#FF4500',
        # 'lacam3_lns': '#FF8C00',
        # 'lg_lacam_lns': '#4169E1',
        'lacam3': '#BB011B',
        'lacam3_lns': '#FC9325',
        'lg_lacam_lns': '#16ACEA',

        # # pale
        # 'lg_lacam':  '#FAC22B',  # Yellow
        # 'gg_lacam': '#603E95',     # Blue
        # 'lacam': '#D7255D',     # Red
        # 'lg&gg_lacam': '#009DA1',        # Green
        'lns2': '#9b59b6'          # Purple

        # 'lg_lacam':  '#FAC22B',  # Yellow
        # 'gg_lacam': '#603E95',     # Blue
        # 'lacam': '#D7255D',     # Red
        # 'lg&gg_lacam': '#009DA1',        # Green
        # 'lns2': '#9b59b6'          # Purple

        # simple
        # 'lg_lacam':  '#0000FF',     # Blue
        # 'gg_lacam': '#FF0000',     # Red
        # 'lacam': '#008000',     # Green
        # 'lg&gg_lacam':  '#FDD017',  # Yellow
        # 'lns2': '#9b59b6'          # Purple

        # 'lg_lacam': '#E8D71E',     # Yellow
        # 'gg_lacam': '#D71B3B',   # Red
        # 'lacam': '#4203C9',        # Dark Blue
        # 'lg&gg_lacam': '#16ACEA',     # Sky Blue
        # 'lns2': '#9b59b6'          # Purple

        # # iconic
        # 'lg_lacam':'#ED1C16',   # Red
        # 'gg_lacam': '#25D366',        # Green
        # 'lacam': '#FF9900',     # Yellow
        # 'lg&gg_lacam': '#0085C3',     # Blue
        # 'lns2': '#833AB4'          # Purple
    }
    
    # Plot each map's data on its corresponding subplot
    for map_idx, (map_name, map_scenario_list) in enumerate(map_scenarios.items()):
        ax = axes[map_idx] if each_map else axes[0]
        
        # Set positions for this map's scenarios
        positions = np.arange(len(map_scenario_list))
        
        # Prepare data for plotting - sort by flow_time_ratio in ascending order for each scenario
        for i, scenario in enumerate(map_scenario_list):
            # Get all method-value pairs for this scenario
            method_value_pairs = []
            for method in methods:
                if method in avg_data[scenario]:
                    method_value_pairs.append((method, avg_data[scenario][method]))
            
            # Plot overlapping bars for this scenario
            for j, (method, value) in enumerate(method_value_pairs):
                color = method_colors.get(method, '#333333')  # Default color if method not in predefined colors
                
                # All bars at the same position (completely overlapping)
                bar_position = positions[i]
                
                # Increase bar width slightly for bars that come to front (to hide gaps)
                current_bar_width = bar_width + j * 0.2 + 0.3

                if (method == "lg_lacam"):
                    continue
                
                # Plot bar with incrementally increasing width
                bar = ax.bar(bar_position, value, current_bar_width, 
                           label=method if i == 0 and map_idx == 0 else "", 
                           color=color, alpha=1.0, edgecolor='none')
        
        # Add line plot for lg_lacam values when normalized (before getting legend handles)
        if 'lg_lacam' in methods:
            lg_lacam_values = []
            for scenario in map_scenario_list:
                if 'lg_lacam' in avg_data[scenario]:
                    lg_lacam_values.append(avg_data[scenario]['lg_lacam'])
                else:
                    lg_lacam_values.append(None)
            
            # Plot line graph
            ax.plot(positions, lg_lacam_values, color='blue', linewidth=5, linestyle='-', marker='none', 
                   label='LG' if map_idx == 0 else "")
        
        # Set x-axis labels for this subplot
        n_map_scenarios = len(map_scenario_list)
        if each_map:
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.tick_params(axis='x', which='both', bottom=False, top=False)
        else:
            tick_positions = [i for i in range(0, n_map_scenarios, max(1, n_map_scenarios // 10))]
            if n_map_scenarios - 1 not in tick_positions and n_map_scenarios > 10:
                tick_positions.append(n_map_scenarios - 1)
            
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_positions)
        
        # Remove margins on left and right to align bars with plot edges
        ax.set_xlim(-0.5, n_map_scenarios)
        
        # Set y-axis to logarithmic scale
        ax.set_yscale('log')
        
        # Set custom y-axis ticks for simple integer display
        from matplotlib.ticker import FixedLocator, FixedFormatter
        if normalize:
            # When normalized, show ticks around 1.0
            ax.yaxis.set_major_locator(FixedLocator([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]))
            ax.yaxis.set_major_formatter(FixedFormatter(['0.5', '1.0', '1.5', '2.0', '2.5', '3.0']))
        else:
            if metric == 'runtime':
                # For runtime, show different scale
                ax.yaxis.set_major_locator(FixedLocator([0.1, 1.0, 10, 100]))
                ax.yaxis.set_major_formatter(FixedFormatter(['0.1', '1.0', '10', '100']))
            else:
                # For flow_time_ratio
                ax.yaxis.set_major_locator(FixedLocator([2, 3, 4, 6]))
                ax.yaxis.set_major_formatter(FixedFormatter(['2', '3', '4', '6']))
        
        # Add grid for better readability
        if each_map:
            ax.grid(axis='y', alpha=0.3, linestyle='-')
            ax.grid(axis='x', alpha=0)  # Disable x-axis grid
        else:
            ax.grid(True, alpha=0.3, linestyle='-')
        
        # Add title for each subplot when using each_map
        if each_map:
            ax.set_title(f'Map: {map_name}', fontsize=20, fontweight='bold')
    
    # Create separate legend file with custom labels (use first subplot for legend)
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    # Map method names to display names
    method_display_names = {
        'lacam': 'LaCAM',
        'lacam3': 'LaCAM3 (initial)',
        'lacam3_lns': 'LaCAM3 (30s)',
        'gg_lacam': 'GG', 
        'lg_lacam': 'LG',
        'lg_lacam_lns': 'LG+LNS (30s)',
        'lg&gg_lacam': 'LG+GG',
        'lns2': 'LNS2'
    }
    
    # Create legend figure
    legend_fig = plt.figure(figsize=(8, 1))
    legend_ax = legend_fig.add_subplot(111)
    legend_ax.axis('off')
    
    # Order legend according to methods order
    ordered_handles = []
    ordered_labels = []
    for method in methods:
        display_name = method_display_names.get(method, method)
        if method in by_label:
            ordered_handles.append(by_label[method])
            ordered_labels.append(display_name)
        # If this is lg_lacam and LG line plot exists, add it right after lg_lacam
        if method == 'lg_lacam' and 'LG' in by_label:
            ordered_handles.append(by_label['LG'])
            ordered_labels.append('LG')
    
    # Create horizontal legend without frame with custom labels
    legend = legend_ax.legend(ordered_handles, ordered_labels, 
                             loc='center', ncol=len(ordered_handles), 
                             frameon=False, fontsize=24)
    
    # Save legend as separate PDF
    if output_file:
        legend_output = output_file.replace('.png', '_legend.pdf').replace('.pdf', '_legend.pdf')
        legend_fig.savefig(legend_output, bbox_inches='tight', pad_inches=0.1)
        print(f"Legend saved to {legend_output}")
    
    plt.close(legend_fig)
    
    # Adjust layout to prevent label cutoff
    plt.subplots_adjust(bottom=0.15, left=0.1, right=0.85, top=0.95)
    
    # Save or show the plot
    if output_file:
        if each_map:
            # Save each map as separate PDF
            for map_idx, (map_name, map_scenario_list) in enumerate(map_scenarios.items()):
                # Create individual figure for this map
                individual_fig, individual_ax = plt.subplots(figsize=(3.5, 10))
                
                # Set positions for this map's scenarios
                positions = np.arange(len(map_scenario_list))
                
                # Plot data for this map
                for i, scenario in enumerate(map_scenario_list):
                    method_value_pairs = []
                    for method in methods:
                        if method in avg_data[scenario]:
                            method_value_pairs.append((method, avg_data[scenario][method]))
                    
                    for j, (method, value) in enumerate(method_value_pairs):
                        color = method_colors.get(method, '#333333')
                        bar_position = positions[i]
                        current_bar_width = bar_width + j * 0.1 + 0.2

                        if (method == "lg_lacam"):
                            continue
                        
                        bar = individual_ax.bar(bar_position, value, current_bar_width, 
                                   label=method if i == 0 else "", 
                                   color=color, alpha=1.0, edgecolor='none')
                
                # Add line plot for lg_lacam
                if 'lg_lacam' in methods:
                    lg_lacam_values = []
                    for scenario in map_scenario_list:
                        if 'lg_lacam' in avg_data[scenario]:
                            lg_lacam_values.append(avg_data[scenario]['lg_lacam'])
                        else:
                            lg_lacam_values.append(None)
                    
                    individual_ax.plot(positions, lg_lacam_values, color='blue', linewidth=10, 
                                     linestyle='-', marker='none', label='LG')
                
                # Configure individual subplot
                n_map_scenarios = len(map_scenario_list)
                tick_positions = [i for i in range(0, n_map_scenarios, max(1, n_map_scenarios // 10))]
                if n_map_scenarios - 1 not in tick_positions and n_map_scenarios > 10:
                    tick_positions.append(n_map_scenarios - 1)
                
                # individual_ax.set_xticks(tick_positions)
                # individual_ax.set_xticklabels(tick_positions)
                individual_ax.set_xlim(-0.5, n_map_scenarios)
                
                # Calculate y-axis upper limit with margin
                max_value = max([max(avg_data[scenario].values()) for scenario in map_scenario_list])
                min_value = min([min(avg_data[scenario].values()) for scenario in map_scenario_list])
                margin = 0.1  # 10% margin
                upper_limit = max_value + (max_value-min_value) * margin
                individual_ax.set_ylim(bottom=1.0, top=upper_limit)
                # individual_ax.set_yscale('log')
                
                # Set y-axis ticks
                if normalize:
                    individual_ax.yaxis.set_major_locator(FixedLocator([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]))
                    individual_ax.yaxis.set_major_formatter(FixedFormatter(['0.5', '1.0', '1.5', '2.0', '2.5', '3.0']))
                else:
                    if metric == 'runtime':
                        individual_ax.yaxis.set_major_locator(FixedLocator([0.1, 1.0, 10, 100]))
                        individual_ax.yaxis.set_major_formatter(FixedFormatter(['0.1', '1.0', '10', '100']))
                    else:
                        pass
                        # individual_ax.yaxis.set_major_locator(FixedLocator([1.0, 1.01, 1.02]))
                        # individual_ax.yaxis.set_major_formatter(FixedFormatter(['1.0', '1.01', '1.02']))
                
                individual_ax.grid(axis='y', alpha=0.3, linestyle='-')
                # individual_ax.set_title(f'Map: {map_name}', fontsize=20, fontweight='bold')

                individual_ax.set_xticks([])
                individual_ax.set_xticklabels([])
                individual_ax.tick_params(axis='x', which='both', bottom=False, top=False)
                
                # Save individual map PDF
                map_output = output_file.replace('.pdf', f'_{map_name}.pdf').replace('.png', f'_{map_name}.pdf')
                individual_fig.savefig(map_output, dpi=150, bbox_inches='tight')
                print(f"Map {map_name} chart saved to {map_output}")
                plt.close(individual_fig)
        else:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Chart saved to {output_file}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Create bar charts comparing methods by scenario')
    parser.add_argument('--csv', '-c', 
                       default='benchmark_results/benchmark_results_all.csv',
                       help='Path to the CSV file (default: benchmark_results/benchmark_results_all.csv)')
    parser.add_argument('--methods', '-m', nargs='+',
                       help='Methods to include in the chart (default: all methods)')
    parser.add_argument('--output', '-o',
                       help='Output file path (default: show plot)')
    parser.add_argument('--normalize', '-n', action='store_true',
                       help='Normalize metric values by lacam values')
    parser.add_argument('--metric', '-t', choices=['flow_time_ratio', 'runtime'], 
                       default='flow_time_ratio',
                       help='Metric to compare (default: flow_time_ratio)')
    parser.add_argument('--each_map', action='store_true',
                       help='Create separate charts for each map')
    
    args = parser.parse_args()
    
    # Default methods if not specified
    if args.methods is None:
        # args.methods = ['lacam', 'gg_lacam', 'lg_lacam', 'lg&gg_lacam']
        # args.methods = ['lg&gg_lacam', 'lg_lacam', 'gg_lacam', 'lacam']
        args.methods = ['lacam3', 'lacam3_lns', 'lg_lacam', 'lg_lacam_lns']
        # args.methods = ['lg_lacam', 'lg_lacam_lns', 'lacam3', 'lacam3_lns']
        # args.methods = ['lg_lacam', 'gg_lacam', 'lacam']
    
    create_scenario_bar_chart(args.csv, args.methods, args.output, args.normalize, args.metric, args.each_map)

if __name__ == "__main__":
    main()