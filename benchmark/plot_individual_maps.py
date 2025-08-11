#!/usr/bin/env python3
"""
Individual Map Benchmark Plotting Script

This script generates separate Runtime vs Flow time/Lower bound ratio plots 
for each of the 6 maps individually, formatted for publication.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Optional
import json

# Set style for publication-quality plots
plt.rcParams.update({
    'font.size': 24,
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
    'legend.fontsize': 24,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

# Define publication-quality color palette and markers
ALGORITHM_STYLES = {
    "lg_lacam": {
        "color": "#0000FF",     # Blue
        "marker": "*",          # Star
        "linestyle": "-",       # Solid line
        "linewidth": 5,         # Line thickness
        "markersize": 28,       # Data point size
        "markeredgewidth": 3,   # Data point border thickness
        "label": "LG"
    },
    "lg_lacam_8": {
        "color": "#1E90FF",     # Dodger Blue
        "marker": "*",
        "linestyle": "-",
        "linewidth": 5,
        "markersize": 24,
        "markeredgewidth": 3,
        "label": "LG (Window:8)"
    },
    "lg_lacam_15": {
        "color": "#00BFFF",     # Deep Sky Blue
        "marker": "*",
        "linestyle": "-",
        "linewidth": 5,
        "markersize": 24,
        "markeredgewidth": 3,
        "label": "LG (Window:15)"
    },
    "lg_lacam_20": {
        "color": "#87CEFA",     # Light Sky Blue
        "marker": "*",
        "linestyle": "-",
        "linewidth": 5,
        "markersize": 24,
        "markeredgewidth": 3,
        "label": "LG (Window:20)"
    },
    "gg_lacam": {
        "color": "#ff66c4",     # pink
        "marker": "o",          # Diamond
        "linestyle": "-",       # Solid line
        "linewidth": 5,         # Line thickness
        "markersize": 18,       # Data point size
        "markeredgewidth": 3,   # Data point border thickness
        "label": "GG"
    },
    "lg&gg_lacam": {
        "color": "#FDD017",     # Yellow
        "marker": "X",          # Plus (filled)
        "linestyle": "-",       # Solid line
        "linewidth": 5,         # Line thickness
        "markersize": 20,       # Data point size
        "markeredgewidth": 3,   # Data point border thickness
        "label": "LG+GG"
    },
    "lacam": {
        "color": "#008000",     # Green
        "marker": "^",          # Triangle
        "linestyle": "-",       # Solid line
        "linewidth": 5,         # Line thickness
        "markersize": 16,       # Data point size
        "markeredgewidth": 3,   # Data point border thickness
        "label": "LaCAM"
    },
    "eecbs_f": {
        "color": "#FF0000",     # Red
        "marker": "^",          # Triangle
        "linestyle": "-.",      # Dash-dot line
        "linewidth": 5,         # Line thickness
        "markersize": 16,       # Data point size
        "markeredgewidth": 3,   # Data point border thickness
        "label": "EECBS-f"
    }, 
    "lns2": {
        "color": "#800080",     # Purple
        "marker": "D",          # Diamond
        "linestyle": "-",       # Dotted line
        "linewidth": 5,         # Line thickness
        "markersize": 14,       # Data point size
        "markeredgewidth": 3,   # Data point border thickness
        "label": "LNS2"
    },
    "lns": {
        "color": "#800080",     # Purple
        "marker": "D",          # Diamond
        "linestyle": ":",       # Dotted line
        "linewidth": 5,         # Line thickness
        "markersize": 14,       # Data point size
        "markeredgewidth": 3,   # Data point border thickness
        "label": "LNS"
    },
    "lacam3": {
        "color": "#FF4500",     # Orange Red
        "marker": "s",          # Square
        "linestyle": "-",       # Solid line
        "linewidth": 5,         # Line thickness
        "markersize": 16,       # Data point size
        "markeredgewidth": 3,   # Data point border thickness
        "label": "LaCAM3"
    },
    "lacam3_lns": {
        "color": "#FF8C00",     # Dark Orange
        "marker": "s",          # Square
        "linestyle": "-",      # Dashed line
        "linewidth": 5,         # Line thickness
        "markersize": 16,       # Data point size
        "markeredgewidth": 3,   # Data point border thickness
        "label": "LaCAM3-LNS"
    },
    "lg_lacam_lns": {
        "color": "#4169E1",     # Royal Blue
        "marker": "*",          # Star
        "linestyle": "-",      # Dashed line
        "linewidth": 5,         # Line thickness
        "markersize": 28,       # Data point size
        "markeredgewidth": 3,   # Data point border thickness
        "label": "LG-LNS"
    }
}

ALGORITHMS = {
    "lg_lacam": "LG (Proposed)",
    "lg_lacam_8": "LG (Window:8)",
    "lg_lacam_15": "LG (Window:15)",
    "lg_lacam_20": "LG (Window:20)",
    "lg&gg_lacam": "LG+GG",
    "lacam": "LaCAM",
    "eecbs_f": "EECBS-f", 
    "lns2": "LNS2",
    "lacam3": "LaCAM3",
    "lacam3_lns": "LaCAM3-LNS",
    "lg_lacam_lns": "LG-LNS"
}

MAPS = {
    "Paris_1_256.map": "Paris_1_256",
    "empty-48-48.map": "empty-48-48", 
    "ost003d.map": "ost003d",
    "random-64-64-20.map": "random-64-64-20",
    "room-64-64-8.map": "room-64-64-8",
    "warehouse-20-40-10-2-2.map": "warehouse-20-40-10-2-2"
}

class IndividualMapPlotter:
    def __init__(self, results_file: str, output_dir: str = "individual_plots"):
        self.results_file = Path(results_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load results
        self.df = self.load_results()
        
    def load_results(self) -> pd.DataFrame:
        """Load benchmark results from CSV or JSON file(s)."""
        
        # Check if the file is a combined file or if we need to load multiple timespan files
        if self.results_file.exists():
            # Single file provided
            if self.results_file.suffix == '.csv':
                df = pd.read_csv(self.results_file)
            elif self.results_file.suffix == '.json':
                with open(self.results_file, 'r') as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
            else:
                raise ValueError(f"Unsupported file format: {self.results_file.suffix}")
        else:
            # Try to load multiple timespan files
            base_name = self.results_file.stem
            base_dir = self.results_file.parent
            
            # Check if there are timespan-specific files
            timespan_files = []
            for file_path in base_dir.glob(f"{base_name}_timespan_*.csv"):
                timespan_files.append(file_path)
            
            if not timespan_files:
                # Try JSON files
                for file_path in base_dir.glob(f"{base_name}_timespan_*.json"):
                    timespan_files.append(file_path)
            
            if not timespan_files:
                raise FileNotFoundError(f"No results files found for {self.results_file}")
            
            # Load and combine all timespan files
            dfs = []
            for file_path in sorted(timespan_files):
                print(f"Loading timespan file: {file_path}")
                if file_path.suffix == '.csv':
                    timespan_df = pd.read_csv(file_path)
                elif file_path.suffix == '.json':
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    timespan_df = pd.DataFrame(data)
                else:
                    continue
                
                dfs.append(timespan_df)
            
            if not dfs:
                raise ValueError("No valid timespan files found")
            
            # Combine all dataframes
            df = pd.concat(dfs, ignore_index=True)
        
        # Filter successful runs only
        df = df[df['success'] == True].copy()
        
        # Calculate flow time ratio if not already present
        if 'flow_time_ratio' not in df.columns or df['flow_time_ratio'].isna().all():
            df['flow_time_ratio'] = df['flow_time'] / df['lower_bound']
        
        return df
    
    def _add_lns_init_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract comp_time_init and soc_init from result files when LNS is enabled."""
        import yaml
        import re
        
        # Add columns if they don't exist
        if 'comp_time_init' not in df.columns:
            df['comp_time_init'] = None
        if 'soc_init' not in df.columns:
            df['soc_init'] = None
        
        # Check if we have a config file to determine LNS setting
        config_path = Path("config_all_maps_lns.yaml")
        if not config_path.exists():
            config_path = Path("config.yaml")
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                lns_enabled = config.get('settings', {}).get('lns', False)
            except:
                lns_enabled = False
        else:
            lns_enabled = False
        
        if not lns_enabled:
            return df
        
        # Extract data for each row
        for idx, row in df.iterrows():
            try:
                # Reconstruct result file path based on the pattern used in benchmark
                algorithm = row['algorithm']
                map_name = Path(row['map']).stem  # Remove .map extension
                agents = row['agents']
                scenario = Path(row['scenario']).stem  # Remove .scen extension
                
                result_file = Path(f"../build/result_{algorithm}_{map_name}_{agents}_{scenario}.txt")
                
                if result_file.exists():
                    with open(result_file, 'r') as f:
                        content = f.read()
                    
                    # Extract comp_time_init
                    match_time = re.search(r'comp_time_init=([0-9.]+)', content)
                    if match_time:
                        df.loc[idx, 'comp_time_init'] = float(match_time.group(1))
                        
                    # Extract soc_init
                    match_soc = re.search(r'soc_init=(\d+)', content)
                    if match_soc:
                        df.loc[idx, 'soc_init'] = int(match_soc.group(1))
                        
            except Exception as e:
                print(f"Warning: Could not extract LNS init data for row {idx}: {e}")
                continue
        
        return df
    
    def plot_lns_scatter(self, map_file: str, save_plots: bool = True):
        """Create scatter plot for LNS mode with comp_time_init vs Flow Time/LB."""
        
        map_data = self.df[self.df['map'] == map_file]
        if map_data.empty:
            print(f"No data found for map: {map_file}")
            return None
        
        # Check if comp_time_init data is available
        if 'comp_time_init' not in map_data.columns or map_data['comp_time_init'].isna().all():
            print(f"No comp_time_init data found for map: {map_file}")
            return None
            
        # Handle both .map and non-.map formats
        if map_file.endswith('.map'):
            map_name = MAPS.get(map_file, map_file.replace('.map', ''))
        else:
            map_name = MAPS.get(map_file + '.map', map_file)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # First collect all data by algorithm and group by agent count
        algorithm_data = {}
        for alg_id in sorted(map_data['algorithm'].unique()):
            alg_data = map_data[map_data['algorithm'] == alg_id]
            # Filter out rows with missing comp_time_init
            alg_data = alg_data[alg_data['comp_time_init'].notna()]
            
            if alg_data.empty:
                continue
            
            # Group by agent count and take mean
            agg_dict = {
                'comp_time_init': 'mean',
                'flow_time_ratio': 'mean'
            }
            grouped = alg_data.groupby('agents').agg(agg_dict).reset_index().sort_values('agents')
            
            if not grouped.empty:
                algorithm_data[alg_id] = grouped
        
        # Connect same agent counts across algorithms with gray dotted lines
        if len(algorithm_data) >= 2:
            all_agents = set()
            for data in algorithm_data.values():
                all_agents.update(data['agents'].values)
            
            for agents in sorted(all_agents):
                points = []
                for alg_id in sorted(algorithm_data.keys()):
                    alg_data = algorithm_data[alg_id]
                    agent_data = alg_data[alg_data['agents'] == agents]
                    if not agent_data.empty:
                        points.append((agent_data['comp_time_init'].iloc[0], agent_data['flow_time_ratio'].iloc[0]))
                
                if len(points) >= 2:
                    points.sort()  # Sort by comp_time_init
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    ax.plot(x_coords, y_coords, color='gray', linestyle=':', 
                           linewidth=1, alpha=0.7, zorder=0)
        
        # Plot each algorithm with agent count series connected by lines
        for alg_id in sorted(map_data['algorithm'].unique()):
            if alg_id not in algorithm_data:
                continue
                
            grouped = algorithm_data[alg_id]
            alg_name = ALGORITHMS.get(alg_id, alg_id)
            style = ALGORITHM_STYLES.get(alg_id, {"color": "black", "marker": "o", "linestyle": "-", "linewidth": 3})
            
            # Plot with agent count series connected
            ax.plot(grouped['comp_time_init'], grouped['flow_time_ratio'], 
                   marker=style['marker'], linestyle=style['linestyle'],
                   color=style['color'], linewidth=style.get('linewidth', 3),
                   markersize=style.get('markersize', 12), 
                   markerfacecolor='white', markeredgecolor=style['color'],
                   markeredgewidth=style.get('markeredgewidth', 2), 
                   label=style.get('label', alg_name), zorder=2)
            
            # Add agent count labels only for LaCAM
            if alg_id == 'lacam':
                for _, row in grouped.iterrows():
                    agents = int(row["agents"])
                    if agents == 1000:
                        label_text = 'agents: 1000'
                    else:
                        label_text = str(agents)
                    ax.annotate(label_text, 
                              (row['comp_time_init'], row['flow_time_ratio']),
                              xytext=(-24, 8), textcoords='offset points',
                              fontsize=18, ha='left', va='bottom', zorder=3)
        
        # Formatting
        ax.set_xlabel('comp_time_init (ms)', fontweight='bold', fontsize=22)
        ax.set_ylabel('Flow Time / LB', fontweight='bold', fontsize=22)
        ax.set_title(f'{map_name} Map - LNS Scatter Plot', fontweight='bold', fontsize=24, y=1.05)
        
        # Ensure all spines are visible with proper thickness
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(3.0)
            spine.set_edgecolor('black')
        
        # Grid styling
        ax.grid(True)
        
        # Legend
        ax.legend(loc='best', frameon=True, edgecolor='black', fancybox=False)
        
        # Set reasonable axis limits
        if not map_data.empty:
            comp_time_values = map_data['comp_time_init'].dropna()
            flow_ratio_values = map_data['flow_time_ratio'].dropna()
            
            if len(comp_time_values) > 0 and len(flow_ratio_values) > 0:
                # Add margins
                x_min, x_max = comp_time_values.min(), comp_time_values.max()
                y_min, y_max = flow_ratio_values.min(), flow_ratio_values.max()
                
                x_margin = (x_max - x_min) * 0.1
                y_margin = (y_max - y_min) * 0.1
                
                ax.set_xlim(left=max(0, x_min - x_margin), right=x_max + x_margin)
                ax.set_ylim(bottom=max(1.0, y_min - y_margin), top=y_max + y_margin)
        
        # Adjust layout
        plt.tight_layout()
        
        if save_plots:
            # Save with map-specific filename
            map_clean = map_file.replace('.map', '').replace('-', '_')
            output_file = self.output_dir / f"lns_scatter_{map_clean}.pdf"
            plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='black')
            print(f"Saved LNS scatter plot: {output_file}")
            plt.close()
        else:
            plt.show()
        
        return fig

    def plot_lns_sub(self, map_file: str, save_plots: bool = True):
        """Create runtime vs flow time ratio plot for LNS sub-mode with multiple timespan support."""
        
        map_data = self.df[self.df['map'] == map_file]
        if map_data.empty:
            print(f"No data found for map: {map_file}")
            return None

        if 'comp_time_init' not in map_data.columns or map_data['comp_time_init'].isna().all() or \
           'flow_time_init' not in map_data.columns or map_data['flow_time_init'].isna().all():
            print(f"No LNS init data found for map: {map_file}")
            return None

        map_name = MAPS.get(map_file, map_file.replace('.map', ''))
        
        # Create a separate plot for each agent count
        agent_counts = sorted(map_data['agents'].unique())
        figs = {}

        for agents in agent_counts:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            agent_data = map_data[map_data['agents'] == agents]

            for alg_id in sorted(agent_data['algorithm'].unique()):
                alg_data = agent_data[agent_data['algorithm'] == alg_id]
                if alg_data.empty:
                    continue

                style = ALGORITHM_STYLES.get(alg_id, {"color": "black", "marker": "o", "linestyle": "-"})
                
                # Check if we have timespan data
                if 'timespan' in alg_data.columns and alg_data['timespan'].notna().any():
                    # Group by timespan and calculate means
                    timespan_data = alg_data.groupby('timespan').agg({
                        'comp_time_init': 'mean',
                        'flow_time_init': 'mean',
                        'runtime': 'mean',
                        'flow_time_ratio': 'mean',
                        'lower_bound': 'mean'
                    }).reset_index().sort_values('timespan')
                    
                    if timespan_data.empty:
                        continue
                    
                    # Create trend line: comp_time_init/1000 -> runtime(t1) -> runtime(t2) -> ...
                    x_coords = []
                    y_coords = []
                    
                    # First point: comp_time_init/1000 and flow_time_init/lower_bound
                    first_row = timespan_data.iloc[0]
                    x_coords.append(first_row['comp_time_init'] / 1000)
                    y_coords.append(first_row['flow_time_init'] / first_row['lower_bound'])
                    
                    # Subsequent points: runtime(timespan) and flow_time_ratio
                    for _, row in timespan_data.iterrows():
                        x_coords.append(row['runtime'])
                        y_coords.append(row['flow_time_ratio'])
                    
                    # Plot the trend line
                    ax.plot(x_coords, y_coords, 
                            marker=style['marker'], linestyle=style['linestyle'], 
                            color=style['color'], linewidth=style.get('linewidth', 3),
                            markersize=style.get('markersize', 24),
                            markerfacecolor='white', markeredgecolor=style['color'],
                            markeredgewidth=style.get('markeredgewidth', 3),
                            label=style.get('label', alg_id))
                    
                    # Add fill_between area for uncertainty
                    # Calculate min/max for each timespan
                    timespan_bounds = alg_data.groupby('timespan').agg({
                        'comp_time_init': ['min', 'max'],
                        'flow_time_init': ['min', 'max'],
                        'runtime': ['min', 'max'],
                        'flow_time_ratio': ['min', 'max'],
                        'lower_bound': 'mean'
                    }).reset_index()
                    
                    # Flatten column names
                    timespan_bounds.columns = ['timespan'] + ['_'.join(col) if col[1] else col[0] for col in timespan_bounds.columns[1:]]
                    timespan_bounds = timespan_bounds.sort_values('timespan')
                    
                    if len(timespan_bounds) > 0:
                        x_fill_lower = []
                        x_fill_upper = []
                        y_fill_lower = []
                        y_fill_upper = []
                        
                        # First point bounds
                        first_bounds = timespan_bounds.iloc[0]
                        x_fill_lower.append(first_bounds['comp_time_init_min'] / 1000)
                        x_fill_upper.append(first_bounds['comp_time_init_max'] / 1000)
                        y_fill_lower.append(first_bounds['flow_time_init_min'] / first_bounds['lower_bound_mean'])
                        y_fill_upper.append(first_bounds['flow_time_init_max'] / first_bounds['lower_bound_mean'])
                        
                        # Subsequent points bounds
                        for _, row in timespan_bounds.iterrows():
                            x_fill_lower.append(row['runtime_min'])
                            x_fill_upper.append(row['runtime_max'])
                            y_fill_lower.append(row['flow_time_ratio_min'])
                            y_fill_upper.append(row['flow_time_ratio_max'])
                        
                        # Create fill_between using the trend line with bounds
                        ax.fill_between(x_coords, y_fill_lower, y_fill_upper,
                                        color=style['color'], alpha=0.1, linewidth=0)
                
                else:
                    # Fallback to original behavior if no timespan data
                    # Calculate initial and final points (mean)
                    runtime_init_mean = alg_data['comp_time_init'].mean() / 1000 # Convert to sec
                    flow_time_init_ratio_mean = alg_data['flow_time_init'].mean() / alg_data['lower_bound'].mean()
                    runtime_final_mean = alg_data['runtime'].mean()
                    flow_time_final_ratio_mean = alg_data['flow_time_ratio'].mean()

                    # Calculate initial and final points (min/max for fill_between)
                    runtime_init_min = alg_data['comp_time_init'].min() / 1000
                    runtime_init_max = alg_data['comp_time_init'].max() / 1000
                    flow_time_init_ratio_min = alg_data['flow_time_init'].min() / alg_data['lower_bound'].mean()
                    flow_time_init_ratio_max = alg_data['flow_time_init'].max() / alg_data['lower_bound'].mean()
                    runtime_final_min = alg_data['runtime'].min()
                    runtime_final_max = alg_data['runtime'].max()
                    flow_time_final_ratio_min = alg_data['flow_time_ratio'].min()
                    flow_time_final_ratio_max = alg_data['flow_time_ratio'].max()

                    # Plotting the line between init and final (mean values)
                    ax.plot([runtime_init_mean, runtime_final_mean], [flow_time_init_ratio_mean, flow_time_final_ratio_mean],
                            marker=style['marker'], linestyle=style['linestyle'], color=style['color'],
                            linewidth=style.get('linewidth', 3), markersize=style.get('markersize', 12),
                            markerfacecolor='white', markeredgecolor=style['color'],
                            markeredgewidth=style.get('markeredgewidth', 3),
                            label=style.get('label', alg_id))

                    # Plotting the fill_between area
                    ax.fill_between([runtime_init_mean, runtime_final_mean], 
                                    [flow_time_init_ratio_min, flow_time_final_ratio_min], 
                                    [flow_time_init_ratio_max, flow_time_final_ratio_max], 
                                    color=style['color'], alpha=0.1, linewidth=0)

            # Formatting
            ax.set_xlabel('Runtime (sec)', fontweight='bold', fontsize=22)
            ax.set_ylabel('Flow Time / LB', fontweight='bold', fontsize=22)
            # ax.set_title(f'{map_name} Map - {agents} Agents', fontweight='bold', fontsize=24, y=1.05)
            ax.grid(True)
            # Set x-axis lower limit to 0
            ax.set_xlim(left=0)
            
            # Set y-axis limits with margin
            if not agent_data.empty:
                # Collect all y-values from the plot data
                all_y_values = []
                for alg_id in sorted(agent_data['algorithm'].unique()):
                    alg_data = agent_data[agent_data['algorithm'] == alg_id]
                    if alg_data.empty:
                        continue
                    
                    if 'timespan' in alg_data.columns and alg_data['timespan'].notna().any():
                        # For timespan data, include all flow_time_ratio values
                        all_y_values.extend(alg_data['flow_time_ratio'].tolist())
                        # Also include flow_time_init/lower_bound values
                        flow_time_init_ratios = alg_data['flow_time_init'] / alg_data['lower_bound']
                        all_y_values.extend(flow_time_init_ratios.tolist())
                    else:
                        # For non-timespan data, include flow_time_ratio and flow_time_init ratio
                        all_y_values.extend(alg_data['flow_time_ratio'].tolist())
                        flow_time_init_ratios = alg_data['flow_time_init'] / alg_data['lower_bound']
                        all_y_values.extend(flow_time_init_ratios.tolist())
                
                if all_y_values:
                    y_min = min(all_y_values)
                    y_max = max(all_y_values)
                    y_margin = (y_max - y_min) * 0.3  # 10% margin
                    ax.set_ylim(bottom=y_min - y_margin/4, top=y_max + y_margin/4)
            
            # ax.legend(loc='best', frameon=True, edgecolor='black', fancybox=False)
            plt.tight_layout()

            if save_plots:
                map_clean = map_file.replace('.map', '').replace('-', '_')
                output_file = self.output_dir / f"lns_sub_{map_clean}_{agents}_agents.pdf"
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                print(f"Saved LNS sub plot: {output_file}")
                plt.close(fig)
            else:
                plt.show()
            
            figs[agents] = fig
        
        # Generate legend as separate PDF
        if save_plots:
            self.save_legend_as_pdf()
        
        return figs
    
    def plot_histogram(self, map_file: str, save_plots: bool = True):
        """Create histogram plots for a single map (flowtime/LB vs scenario number), one per agent count."""
        
        map_data = self.df[self.df['map'] == map_file]
        if map_data.empty:
            print(f"No data found for map: {map_file}")
            return None
            
        map_name = MAPS.get(map_file, map_file.replace('.map', ''))
        
        # Group by scenario to get scenario numbers
        if 'scenario' not in map_data.columns:
            print(f"No scenario information found for map: {map_file}")
            return None
        
        # Get unique agent counts and sort them
        agent_counts = sorted(map_data['agents'].unique())
        n_agents = len(agent_counts)
        
        if n_agents == 0:
            print(f"No agent data found for map: {map_file}")
            return None
        
        # Create subplots for each agent count
        fig, axes = plt.subplots(1, n_agents, figsize=(4*n_agents, 8))
        if n_agents == 1:
            axes = [axes]  # Make it iterable for single subplot
        
        figures = []
        for agent_idx, agents in enumerate(agent_counts):
            ax = axes[agent_idx]
            
            # Filter data for this agent count
            agent_data = map_data[map_data['agents'] == agents]
            
            # Get unique scenarios and sort them
            scenarios = sorted(agent_data['scenario'].unique())
            
            if len(scenarios) == 0:
                print(f"No scenarios found for map: {map_file}, agents: {agents}")
                continue
            
            # Collect data by algorithm and scenario
            algorithm_data = {}
            for alg_id in sorted(agent_data['algorithm'].unique()):
                alg_data = agent_data[agent_data['algorithm'] == alg_id]
                scenario_values = {}
                
                for _, row in alg_data.iterrows():
                    scenario = row['scenario']
                    flow_ratio = row['flow_time_ratio']
                    if scenario not in scenario_values:
                        scenario_values[scenario] = []
                    scenario_values[scenario].append(flow_ratio)
                
                # Take mean if multiple runs per scenario
                for scenario in scenario_values:
                    scenario_values[scenario] = np.mean(scenario_values[scenario])
                
                algorithm_data[alg_id] = scenario_values
            
            # Set up bar positions - use same position for all algorithms (complete overlap)
            n_algorithms = len(algorithm_data)
            if n_algorithms == 0:
                continue
                
            bar_width = 1.0  # Full width with no gaps between scenarios
            x_positions = np.arange(len(scenarios))
            
            # Plot bars for each algorithm, sorted by flow_time_ratio (smallest first)
            for scenario_idx, scenario in enumerate(scenarios):
                # Get all algorithms that have data for this scenario
                scenario_algorithms = []
                for alg_id, scenario_values in algorithm_data.items():
                    if scenario in scenario_values:
                        scenario_algorithms.append((alg_id, scenario_values[scenario]))
                
                # Sort by flow_time_ratio (descending) so largest values are plotted first (behind)
                scenario_algorithms.sort(key=lambda x: x[1], reverse=True)
                
                # Plot bars for this scenario - all at same x position for complete overlap
                for bar_idx, (alg_id, flow_ratio) in enumerate(scenario_algorithms):
                    style = ALGORITHM_STYLES.get(alg_id, {"color": "black", "label": alg_id})
                    x_pos = x_positions[scenario_idx]  # Same position for all algorithms
                    
                    ax.bar(x_pos, flow_ratio, bar_width, 
                          color=style['color'], alpha=0.8, 
                          label=style.get('label', alg_id) if scenario_idx == 0 else "",
                          edgecolor='black', linewidth=1)
            
            # Formatting for each subplot
            # ax.set_xlabel('Scenario Number', fontweight='bold', fontsize=18)  # Removed
            if agent_idx == 0:  # Only leftmost subplot gets y-label
                ax.set_ylabel('Flow Time / LB', fontweight='bold', fontsize=18)
            ax.set_title(f'{agents} Agents', fontweight='bold', fontsize=20)
            
            # Remove x-axis labels and ticks
            ax.set_xticks([])
            ax.set_xticklabels([])
            
            # Ensure all spines are visible with proper thickness
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(2.0)
                spine.set_edgecolor('black')
            
            # Grid styling
            ax.grid(True, axis='y')
            
            # Set y-axis limits with minimum of 1.0
            y_min = 1.0
            if len(agent_data) > 0:
                y_max = agent_data['flow_time_ratio'].max() * 1.1  # Add 10% margin
                ax.set_ylim(bottom=y_min, top=y_max)
            
            # Legend only for the first subplot to avoid repetition
            if agent_idx == 0:
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), loc='best', frameon=True, 
                         edgecolor='black', fancybox=False, fontsize=14)
        
        # Overall title
        fig.suptitle(f'{map_name} Map - Histogram', fontweight='bold', fontsize=24, y=0.95)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)  # Make more room for suptitle
        
        if save_plots:
            # Save with map-specific filename
            map_clean = map_file.replace('.map', '').replace('-', '_')
            output_file = self.output_dir / f"histogram_{map_clean}.pdf"
            plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='black')
            print(f"Saved histogram plot: {output_file}")
            plt.close()
        else:
            plt.show()
        
        return fig

    def plot_individual_map(self, map_file: str, save_plots: bool = True, markers_by_agents: bool = False, lns_line: bool = False):
        """Create runtime vs flow time ratio plot for a single map."""
        
        map_data = self.df[self.df['map'] == map_file]
        if map_data.empty:
            print(f"No data found for map: {map_file}")
            return None
            
        map_name = MAPS.get(map_file, map_file.replace('.map', ''))
        
        # Create figure with proper size for publication (square for 1:1 aspect ratio)
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # Define markers for agent counts (if markers_by_agents is True)
        if markers_by_agents:
            # agent_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', '|', '_']
            agent_markers = ['X', 'o', '^', 's', '*', '<', '>', 'p', '*', 'h', 'H', '+', 'x', '|', '_']
            agent_markers_size = [22, 22, 22, 20, 32]
            # Get all unique agent counts across all algorithms
            all_agent_counts = sorted(set(map_data['agents'].unique()))
        
        # First collect all data by algorithm
        algorithm_data = {}
        algorithm_bounds = {}
        for alg_id in sorted(map_data['algorithm'].unique()):
            alg_data = map_data[map_data['algorithm'] == alg_id]
            # Group by agent count and take mean
            agg_dict = {
                'runtime': 'mean',
                'flow_time_ratio': 'mean'
            }
            # Add comp_time_init if available
            if 'comp_time_init' in alg_data.columns:
                agg_dict['comp_time_init'] = 'mean'
            
            grouped = alg_data.groupby('agents').agg(agg_dict).reset_index().sort_values('agents')
            
            # Also calculate min/max bounds for flow_time_ratio
            bounds_dict = {
                'runtime': 'mean',
                'flow_time_ratio': ['min', 'max']
            }
            bounds = alg_data.groupby('agents').agg(bounds_dict).reset_index().sort_values('agents')
            # Flatten column names for bounds
            bounds.columns = ['agents', 'runtime', 'flow_time_ratio_min', 'flow_time_ratio_max']
            
            if not grouped.empty:
                algorithm_data[alg_id] = grouped
                algorithm_bounds[alg_id] = bounds
        
        # Collect points for later connection (after axis scaling is set)
        agent_points_data = {}
        if len(algorithm_data) >= 2:
            all_agents = set()
            for data in algorithm_data.values():
                all_agents.update(data['agents'].values)
            
            for agents in sorted(all_agents):
                points_with_alg = []
                for alg_id in sorted(algorithm_data.keys()):
                    alg_data = algorithm_data[alg_id]
                    agent_data = alg_data[alg_data['agents'] == agents]
                    if not agent_data.empty:
                        point = (agent_data['runtime'].iloc[0], agent_data['flow_time_ratio'].iloc[0])
                        points_with_alg.append((point, alg_id))
                
                if len(points_with_alg) >= 2:
                    agent_points_data[agents] = points_with_alg
        
        # Plot each algorithm with agent count series connected by lines
        for alg_id in sorted(map_data['algorithm'].unique()):
            if alg_id not in algorithm_data:
                continue
                
            grouped = algorithm_data[alg_id]
            bounds = algorithm_bounds[alg_id]
            alg_name = ALGORITHMS.get(alg_id, alg_id)
            style = ALGORITHM_STYLES.get(alg_id, {"color": "black", "marker": "o", "linestyle": "-", "linewidth": 3})
            
            # Plot fill_between for min/max bounds
            ax.fill_between(bounds['runtime'], bounds['flow_time_ratio_min'], bounds['flow_time_ratio_max'],
                           color=style['color'], alpha=0.2, linewidth=0)
            
            # Use consistent marker size for all methods
            base_markersize = 20  # Fixed size for all methods
            marker_sizes = [base_markersize] * len(grouped)
            
            # Plot points individually with varying sizes
            for i, (_, row) in enumerate(grouped.iterrows()):
                # Choose marker based on option
                if markers_by_agents:
                    # Use agent count to determine marker
                    agent_idx = all_agent_counts.index(row['agents'])
                    marker = agent_markers[agent_idx % len(agent_markers)]
                    marker_size = agent_markers_size[agent_idx % len(agent_markers)]
                else:
                    marker = style['marker']
                    marker_size = base_markersize
                
                ax.scatter(row['runtime'], row['flow_time_ratio'],
                          marker=marker, s=marker_size**2,  # s expects area (size squared)
                          color=style['color'], 
                          facecolor='white', edgecolor=style['color'],
                          linewidth=style.get('markeredgewidth', 2),
                          zorder=2)
            
            # Plot connecting lines (without markers to avoid duplication)
            ax.plot(grouped['runtime'], grouped['flow_time_ratio'], 
                   linestyle=style['linestyle'],
                   color=style['color'], linewidth=style.get('linewidth', 3),
                   label=style.get('label', alg_name), zorder=1)
            
            # Add agent count labels only for LaCAM
            if alg_id == 'lacam':
                for _, row in grouped.iterrows():
                    agents = int(row["agents"])
                    if agents == 1000:
                        label_text = 'agents: 1000'
                    else:
                        label_text = str(agents)
                    
                    # Add comp_time_init if available
                    if 'comp_time_init' in grouped.columns and pd.notna(row.get('comp_time_init')):
                        comp_time_init = row['comp_time_init']
                        label_text += f' (init: {comp_time_init:.1f})'
                    
                    ax.annotate(label_text, 
                              (row['runtime'], row['flow_time_ratio']),
                              xytext=(20, -10), textcoords='offset points',
                              fontsize=32, ha='left', va='bottom', zorder=3)
        
        # Formatting
        ax.set_xlabel('runtime (sec)', fontweight='bold', fontsize=22)
        ax.set_ylabel('flow time / LB', fontweight='bold', fontsize=22)
        # ax.set_title(f'{map_name} Map', fontweight='bold', fontsize=24, y=1.05)
        
        # Ensure all spines are visible with proper thickness
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(3.0)
            spine.set_edgecolor('black')
        
        # Grid styling
        ax.grid(True)
        
        # Set tick label font size
        ax.tick_params(axis='both', which='major', labelsize=32)
        
        # Legend
        # ax.legend(loc='best', frameon=True, edgecolor='black', fancybox=False)
        
        # Set log scale for x-axis (runtime)
        # ax.set_xscale('log')
        
        # Set equal aspect ratio (1:1)
        # ax.set_aspect('equal', adjustable='box')
        
        # Set reasonable axis limits based on grouped mean values
        if algorithm_data:
            # Get min and max from grouped mean values (not all individual data points)
            all_flow_ratios = []
            all_runtimes = []
            for grouped in algorithm_data.values():
                all_flow_ratios.extend(grouped['flow_time_ratio'].tolist())
                all_runtimes.extend(grouped['runtime'].tolist())
            
            if all_flow_ratios and all_runtimes:
                # Calculate proper limits with margins
                y_min = min(all_flow_ratios)
                y_max = max(all_flow_ratios)
                x_min = min(all_runtimes)
                x_max = max(all_runtimes)
                
                # Add margins: 10% for y-axis, log-friendly margins for x-axis
                y_margin = (y_max - y_min) * 0.1
                # y_bottom = max(1.0, y_min - y_margin)
                y_bottom = y_min - y_margin
                y_top = y_max + y_margin
                
                # For log scale, use multiplicative margins
                # x_margin_factor = 0.5  # 50% margin on each side for log scale
                # x_left = x_min / (1 + x_margin_factor)
                # x_right = x_max * (1 + x_margin_factor)
                # x_left = x_min - x_margin_factor
                # x_left = -0.5
                # x_right = x_max + x_margin_factor

                x_margin = (x_max - x_min) * 0.1
                # x_bottom = max(1.0, x_min - x_margin)
                x_bottom = - x_margin
                x_top = x_max + x_margin
                
                ax.set_xlim(left=x_bottom, right=x_top)
                ax.set_ylim(bottom=y_bottom, top=y_top)
        
        # Connect same agent counts across algorithms with gray dotted lines (after axis scaling)
        if agent_points_data:
            import math
            
            # Get axis limits for normalization
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            
            # Calculate agent count range for line thickness scaling
            all_agent_counts = list(agent_points_data.keys())
            min_agents = min(all_agent_counts) if all_agent_counts else 1
            max_agents = max(all_agent_counts) if all_agent_counts else 1
            
            def polar_angle_normalized(point):
                x, y = point
                # Normalize coordinates to [0,1] based on axis limits
                x_norm = (x - x_min) / (x_max - x_min) if x_max > x_min else 0
                y_norm = (y - y_min) / (y_max - y_min) if y_max > y_min else 0
                return math.atan2(y_norm, x_norm)
            
            for agents, points_with_alg in agent_points_data.items():
                # Sort points by normalized polar angle
                points_with_angles_alg = [(p, alg_id, polar_angle_normalized(p)) for p, alg_id in points_with_alg]
                points_with_angles_alg.sort(key=lambda x: x[2])
                points_sorted = [p[0] for p in points_with_angles_alg]
                
                # Calculate line thickness and color based on agent count (same scaling as marker size)
                base_linewidth = 2.0
                if max_agents > min_agents:
                    # line_thickness = base_linewidth * (1.0 + 2.0 * (agents - min_agents) / (max_agents - min_agents))
                    line_thickness = base_linewidth
                    # Color interpolation from gray (0.5) to black (0.0)
                    # color_intensity = 0.5 * (1.0 - (agents - min_agents) / (max_agents - min_agents))
                    color_intensity = 0.5
                    line_color = (color_intensity, color_intensity, color_intensity)
                else:
                    line_thickness = base_linewidth
                    line_color = 'gray'

                # # Connect consecutive points in polar angle order
                # for i in range(len(points_sorted) - 1):
                #     x_coords = [points_sorted[i][0], points_sorted[i+1][0]]
                #     y_coords = [points_sorted[i][1], points_sorted[i+1][1]]
                #     ax.plot(x_coords, y_coords, color=line_color, linestyle=':', 
                #            linewidth=line_thickness, alpha=0.7, zorder=0)
        
        # Connect same method groups with lines for each agent count (if lns_line is enabled)
        if lns_line:
            # Define method groups
            method_groups = [
                ("lacam3", "lacam3_lns"),
                ("lg_lacam", "lg_lacam_lns")
            ]
            
            for base_method, lns_method in method_groups:
                if base_method in algorithm_data and lns_method in algorithm_data:
                    base_data = algorithm_data[base_method]
                    lns_data = algorithm_data[lns_method]
                    
                    # Get the color from the base method's style
                    base_style = ALGORITHM_STYLES.get(base_method, {"color": "black"})
                    line_color = base_style['color']
                    
                    # Find common agent counts
                    base_agents = set(base_data['agents'].values)
                    lns_agents = set(lns_data['agents'].values)
                    common_agents = base_agents.intersection(lns_agents)
                    
                    for agents in common_agents:
                        # Get points for this agent count
                        base_point_data = base_data[base_data['agents'] == agents]
                        lns_point_data = lns_data[lns_data['agents'] == agents]
                        
                        if not base_point_data.empty and not lns_point_data.empty:
                            base_point = (base_point_data['runtime'].iloc[0], base_point_data['flow_time_ratio'].iloc[0])
                            lns_point = (lns_point_data['runtime'].iloc[0], lns_point_data['flow_time_ratio'].iloc[0])
                            
                            # Draw line connecting the two points
                            x_coords = [base_point[0], lns_point[0]]
                            y_coords = [base_point[1], lns_point[1]]
                            
                            # Use the base method's color for the connection line
                            ax.plot(x_coords, y_coords, color=line_color, linestyle='--', 
                                   linewidth=2, alpha=0.8, zorder=1)
        
        # Adjust layout
        plt.tight_layout()
        
        if save_plots:
            # Save with map-specific filename
            # map_clean = map_file.replace('.map', '').replace('-', '_')
            map_clean = map_file.replace('.map', '')
            output_file = self.output_dir / f"time_flowtime_{map_clean}.pdf"
            plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='black')
            print(f"Saved plot: {output_file}")
            plt.close()
        else:
            plt.show()
        
        return fig
    
    def plot_all_individual_maps(self, save_plots: bool = True, histogram_mode: bool = False, lns_mode: bool = False, lns_sub_mode: bool = False, markers_by_agents: bool = False, lns_line: bool = False):
        """Generate individual plots for all available maps."""
        
        available_maps = self.df['map'].unique()
        if lns_mode:
            plot_type = "LNS scatter"
        elif histogram_mode:
            plot_type = "histogram"
        elif lns_sub_mode:
            plot_type = "LNS sub"
        else:
            plot_type = "runtime vs flow ratio"
        print(f"Generating {plot_type} plots for {len(available_maps)} maps...")
        
        figures = {}
        for map_file in sorted(available_maps):
            print(f"Processing map: {MAPS.get(map_file, map_file)}")
            if lns_mode:
                fig = self.plot_lns_scatter(map_file, save_plots)
            elif histogram_mode:
                fig = self.plot_histogram(map_file, save_plots)
            elif lns_sub_mode:
                fig = self.plot_lns_sub(map_file, save_plots)
            else:
                fig = self.plot_individual_map(map_file, save_plots, markers_by_agents, lns_line)
            if fig:
                figures[map_file] = fig
        
        print(f"Generated {len(figures)} individual map plots")
        return figures
    
    def generate_summary_table_by_map(self):
        """Generate a summary table of results grouped by map."""
        
        # Group by map and agents, calculate statistics
        summary = self.df.groupby(['map', 'agents']).agg({
            'runtime': ['mean', 'std', 'count'],
            'flow_time_ratio': ['mean', 'std'],
            'success': 'count'
        }).round(4)
        
        # Flatten column names
        summary.columns = ['_'.join(col).strip() for col in summary.columns]
        
        # Save to CSV
        summary_file = self.output_dir / "summary_by_map.csv"
        summary.to_csv(summary_file)
        print(f"Summary table saved to {summary_file}")
        
        return summary
    
    def save_legend_as_pdf(self, save_plots: bool = True):
        """Generate and save the legend as a separate PDF file."""
        
        # Create a temporary figure just for the legend
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        
        # Get unique algorithms from the data
        available_algorithms = set(self.df['algorithm'].unique())
        
        # Define desired order for legend
        legend_order = ['lacam', 'gg_lacam', 'lg_lacam', 'lg&gg_lacam', 'lns2']
        
        # Create legend entries for each algorithm present in the data
        legend_handles = []
        legend_labels = []
        
        # First add algorithms in the specified order
        for alg_id in legend_order:
            if alg_id in available_algorithms:
                style = ALGORITHM_STYLES.get(alg_id, {"color": "black", "marker": "o", "linestyle": "-"})
                
                # Create a line2D object for the legend
                from matplotlib.lines import Line2D
                legend_line = Line2D([0], [0], 
                                   color=style['color'], 
                                   marker=style['marker'],
                                   linestyle=style['linestyle'],
                                   linewidth=style.get('linewidth', 5),
                                   markersize=style.get('markersize', 24),
                                   markerfacecolor='white',
                                   markeredgecolor=style['color'],
                                   markeredgewidth=style.get('markeredgewidth', 3))
                
                legend_handles.append(legend_line)
                legend_labels.append(style.get('label', alg_id))
        
        # Add any remaining algorithms not in the specified order
        for alg_id in sorted(available_algorithms - set(legend_order)):
            style = ALGORITHM_STYLES.get(alg_id, {"color": "black", "marker": "o", "linestyle": "-"})
            
            # Create a line2D object for the legend
            from matplotlib.lines import Line2D
            legend_line = Line2D([0], [0], 
                               color=style['color'], 
                               marker=style['marker'],
                               linestyle=style['linestyle'],
                               linewidth=style.get('linewidth', 5),
                               markersize=style.get('markersize', 24),
                               markerfacecolor='white',
                               markeredgecolor=style['color'],
                               markeredgewidth=style.get('markeredgewidth', 3))
            
            legend_handles.append(legend_line)
            legend_labels.append(style.get('label', alg_id))
        
        # Remove the axes
        ax.remove()
        
        # Create legend - arrange all items horizontally in a single row, no frame
        figlegend = plt.figlegend(legend_handles, legend_labels, 
                                 loc='center', ncol=len(legend_labels), frameon=False, 
                                 fontsize=16)
        
        if save_plots:
            # Save legend as PDF
            output_file = self.output_dir / "legend.pdf"
            plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='black')
            print(f"Saved legend: {output_file}")
            plt.close()
        else:
            plt.show()
        
        return fig

def main():
    parser = argparse.ArgumentParser(description="Generate individual map benchmark plots")
    parser.add_argument("results_file", 
                       help="Path to benchmark results CSV or JSON file, or base name for timespan files")
    parser.add_argument("--output-dir", default="individual_plots",
                       help="Output directory for plots")
    parser.add_argument("--map", 
                       help="Generate plot for specific map only")
    parser.add_argument("--histogram", action="store_true",
                       help="Generate histogram plot (flowtime/LB vs scenario number)")
    parser.add_argument("--lns", action="store_true",
                       help="Generate LNS scatter plot (comp_time_init vs Flow Time/LB)")
    parser.add_argument("--lns-sub", action="store_true",
                       help="Generate LNS sub plot (runtime vs flow_time_ratio)")
    parser.add_argument("--legend", action="store_true",
                       help="Generate legend as separate PDF file")
    parser.add_argument("--markers-by-agents", action="store_true",
                       help="Use different markers for agent counts instead of methods")
    parser.add_argument("--lns-line", action="store_true",
                       help="Connect same method groups (lacam3/lacam3_lns and lg_lacam/lg_lacam_lns) with lines for each agent count")
    
    args = parser.parse_args()
    
    results_path = Path(args.results_file)
    
    # Check if the file exists directly, or if timespan files exist
    if not results_path.exists():
        # Check if timespan files exist
        timespan_files = list(results_path.parent.glob(f"{results_path.stem}_timespan_*.csv"))
        if not timespan_files:
            timespan_files = list(results_path.parent.glob(f"{results_path.stem}_timespan_*.json"))
        
        if not timespan_files:
            print(f"Results file not found: {args.results_file}")
            print(f"Also checked for timespan files: {results_path.parent}/{results_path.stem}_timespan_*.csv")
            return 1
    
    plotter = IndividualMapPlotter(args.results_file, args.output_dir)
    
    if args.legend:
        # Generate legend only
        plotter.save_legend_as_pdf()
    elif args.map:
        # Plot specific map
        if args.lns:
            plotter.plot_lns_scatter(args.map)
        elif args.histogram:
            plotter.plot_histogram(args.map)
        elif args.lns_sub:
            plotter.plot_lns_sub(args.map)
        else:
            plotter.plot_individual_map(args.map, markers_by_agents=args.markers_by_agents, lns_line=args.lns_line)
    else:
        # Plot all maps
        plotter.plot_all_individual_maps(histogram_mode=args.histogram, lns_mode=args.lns, lns_sub_mode=args.lns_sub, markers_by_agents=args.markers_by_agents, lns_line=args.lns_line)
        if not args.histogram and not args.lns and not args.lns_sub:
            plotter.generate_summary_table_by_map()
    
    # Always generate legend when plotting (unless only legend was requested)
    if not args.legend:
        plotter.save_legend_as_pdf()
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())