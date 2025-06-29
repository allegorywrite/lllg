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

# Define publication-quality color palette and markers
ALGORITHM_STYLES = {
    "lg_lacam": {
        "color": "#0000FF",     # Blue
        "marker": "*",          # Star
        "linestyle": "-",       # Solid line
        "linewidth": 5,         # Line thickness
        "markersize": 24,       # Data point size
        "markeredgewidth": 3,   # Data point border thickness
        "label": "LG-LaCAM"
    },
    "lg_lacam_8": {
        "color": "#1E90FF",     # Dodger Blue
        "marker": "*",
        "linestyle": "-",
        "linewidth": 5,
        "markersize": 24,
        "markeredgewidth": 3,
        "label": "LG-LaCAM (Window:8)"
    },
    "lg_lacam_15": {
        "color": "#00BFFF",     # Deep Sky Blue
        "marker": "*",
        "linestyle": "-",
        "linewidth": 5,
        "markersize": 24,
        "markeredgewidth": 3,
        "label": "LG-LaCAM (Window:15)"
    },
    "lg_lacam_20": {
        "color": "#87CEFA",     # Light Sky Blue
        "marker": "*",
        "linestyle": "-",
        "linewidth": 5,
        "markersize": 24,
        "markeredgewidth": 3,
        "label": "LG-LaCAM (Window:20)"
    },
    "gg_lacam": {
        "color": "#ff66c4",     # pink
        "marker": "*",          # Star
        "linestyle": "-",       # Solid line
        "linewidth": 5,         # Line thickness
        "markersize": 24,       # Data point size
        "markeredgewidth": 3,   # Data point border thickness
        "label": "GG-LaCAM"
    },
    "lg&gg_lacam": {
        "color": "#FDD017",     # Yellow
        "marker": "*",          # Star
        "linestyle": "-",       # Solid line
        "linewidth": 5,         # Line thickness
        "markersize": 24,       # Data point size
        "markeredgewidth": 3,   # Data point border thickness
        "label": "LG&GG-LaCAM"
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
        "linestyle": ":",       # Dotted line
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
    }
}

ALGORITHMS = {
    "lg_lacam": "LG-LaCAM (Proposed)",
    "lg_lacam_8": "LG-LaCAM (Window:8)",
    "lg_lacam_15": "LG-LaCAM (Window:15)",
    "lg_lacam_20": "LG-LaCAM (Window:20)",
    "lg&gg_lacam": "LG&GG-LaCAM",
    "lacam": "LaCAM",
    "eecbs_f": "EECBS-f", 
    "lns2": "LNS2"
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
        """Load benchmark results from CSV or JSON file."""
        if self.results_file.suffix == '.csv':
            df = pd.read_csv(self.results_file)
        elif self.results_file.suffix == '.json':
            with open(self.results_file, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported file format: {self.results_file.suffix}")
        
        # Filter successful runs only
        df = df[df['success'] == True].copy()
        
        # Calculate flow time ratio if not already present
        if 'flow_time_ratio' not in df.columns or df['flow_time_ratio'].isna().all():
            df['flow_time_ratio'] = df['flow_time'] / df['lower_bound']
        
        return df
    
    def plot_individual_map(self, map_file: str, save_plots: bool = True):
        """Create runtime vs flow time ratio plot for a single map."""
        
        map_data = self.df[self.df['map'] == map_file]
        if map_data.empty:
            print(f"No data found for map: {map_file}")
            return None
            
        map_name = MAPS.get(map_file, map_file.replace('.map', ''))
        
        # Create figure with proper size for publication (square for 1:1 aspect ratio)
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 8))
        
        # First collect all data by algorithm
        algorithm_data = {}
        for alg_id in sorted(map_data['algorithm'].unique()):
            alg_data = map_data[map_data['algorithm'] == alg_id]
            # Group by agent count and take mean
            grouped = alg_data.groupby('agents').agg({
                'runtime': 'mean',
                'flow_time_ratio': 'mean'
            }).reset_index().sort_values('agents')
            
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
                        points.append((agent_data['runtime'].iloc[0], agent_data['flow_time_ratio'].iloc[0]))
                
                if len(points) >= 2:
                    points.sort()  # Sort by runtime
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
            ax.plot(grouped['runtime'], grouped['flow_time_ratio'], 
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
                              (row['runtime'], row['flow_time_ratio']),
                              xytext=(-24, 8), textcoords='offset points',
                              fontsize=18, ha='left', va='bottom', zorder=3)
        
        # Formatting
        ax.set_xlabel('runtime (sec)', fontweight='bold', fontsize=22)
        ax.set_ylabel('flow time / LB', fontweight='bold', fontsize=22)
        ax.set_title(f'{map_name} Map', fontweight='bold', fontsize=24, y=1.05)
        
        # Ensure all spines are visible with proper thickness
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(3.0)
            spine.set_edgecolor('black')
        
        # Grid styling
        ax.grid(True)
        
        # Legend
        ax.legend(loc='best', frameon=True, edgecolor='black', fancybox=False)
        
        # Set log scale for x-axis (runtime)
        # ax.set_xscale('log')
        
        # Set equal aspect ratio (1:1)
        # ax.set_aspect('equal', adjustable='box')
        
        # Set reasonable axis limits to include all data points
        if not map_data.empty:
            # Get min and max from all actual data points (not just grouped means)
            all_flow_ratios = map_data['flow_time_ratio'].tolist()
            all_runtimes = map_data['runtime'].tolist()
            
            if all_flow_ratios and all_runtimes:
                # Calculate proper limits with margins
                y_min = min(all_flow_ratios)
                y_max = max(all_flow_ratios)
                x_min = min(all_runtimes)
                x_max = max(all_runtimes)
                
                # Add margins: 10% for y-axis, log-friendly margins for x-axis
                y_margin = (y_max - y_min) * 0.1
                y_bottom = max(1.0, y_min - y_margin)
                y_top = y_max + y_margin
                
                # For log scale, use multiplicative margins
                x_margin_factor = 0.5  # 50% margin on each side for log scale
                # x_left = x_min / (1 + x_margin_factor)
                # x_right = x_max * (1 + x_margin_factor)
                x_left = x_min - x_margin_factor
                x_right = x_max + x_margin_factor
                
                ax.set_xlim(left=x_left, right=x_right)
                ax.set_ylim(bottom=y_bottom, top=y_top)
        
        # Adjust layout
        plt.tight_layout()
        
        if save_plots:
            # Save with map-specific filename
            map_clean = map_file.replace('.map', '').replace('-', '_')
            output_file = self.output_dir / f"runtime_vs_flow_ratio_{map_clean}.pdf"
            plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='black')
            print(f"Saved plot: {output_file}")
            plt.close()
        else:
            plt.show()
        
        return fig
    
    def plot_all_individual_maps(self, save_plots: bool = True):
        """Generate individual plots for all available maps."""
        
        available_maps = self.df['map'].unique()
        print(f"Generating plots for {len(available_maps)} maps...")
        
        figures = {}
        for map_file in sorted(available_maps):
            print(f"Processing map: {MAPS.get(map_file, map_file)}")
            fig = self.plot_individual_map(map_file, save_plots)
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

def main():
    parser = argparse.ArgumentParser(description="Generate individual map benchmark plots")
    parser.add_argument("results_file", 
                       help="Path to benchmark results CSV or JSON file")
    parser.add_argument("--output-dir", default="individual_plots",
                       help="Output directory for plots")
    parser.add_argument("--map", 
                       help="Generate plot for specific map only")
    
    args = parser.parse_args()
    
    if not Path(args.results_file).exists():
        print(f"Results file not found: {args.results_file}")
        return 1
    
    plotter = IndividualMapPlotter(args.results_file, args.output_dir)
    
    if args.map:
        # Plot specific map
        plotter.plot_individual_map(args.map)
    else:
        # Plot all maps
        plotter.plot_all_individual_maps()
        plotter.generate_summary_table_by_map()
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())