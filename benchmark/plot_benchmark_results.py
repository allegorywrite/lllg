#!/usr/bin/env python3
"""
MAPF Benchmark Results Plotting Script

This script generates runtime vs Flow time/Lower bound ratio plots for comparing
different MAPF algorithms across multiple maps and agent counts.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Optional
import json

# Set style for publication-quality plots
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'serif'],
    'axes.linewidth': 1.0,
    'axes.edgecolor': 'black',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.linewidth': 0.5,
    'grid.color': 'gray',
    'grid.alpha': 0.3,
    'figure.facecolor': 'white',
    'figure.edgecolor': 'black',
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'legend.frameon': True,
    'legend.fancybox': False,
    'legend.edgecolor': 'black',
    'legend.facecolor': 'white',
    'legend.framealpha': 1.0
})

# Define publication-quality color palette and markers
ALGORITHM_STYLES = {
    "lg_lacam": {"color": "#000000", "marker": "o", "linestyle": "-", "linewidth": 2},
    "eecbs_f": {"color": "#FF0000", "marker": "s", "linestyle": "--", "linewidth": 2}, 
    "lns2": {"color": "#0000FF", "marker": "^", "linestyle": "-.", "linewidth": 2},
    "lacam_plus": {"color": "#008000", "marker": "D", "linestyle": ":", "linewidth": 2}
}

ALGORITHMS = {
    "lg_lacam": "LG-LaCAM (Proposed)",
    "eecbs_f": "EECBS-f", 
    "lns2": "LNS2",
    "lacam_plus": "LaCAM+"
}

MAPS = {
    "Paris_1_256.map": "Paris",
    "empty-48-48.map": "Empty", 
    "ost003d.map": "OST003D",
    "random-64-64-20.map": "Random",
    "room-64-64-8.map": "Room",
    "warehouse-20-40-10-2-2.map": "Warehouse"
}

class BenchmarkPlotter:
    def __init__(self, results_file: str, output_dir: str = "plots"):
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
    
    def plot_runtime_vs_flow_ratio_by_map(self, save_plots: bool = True):
        """Create runtime vs flow time ratio plots for each map."""
        
        maps = self.df['map'].unique()
        
        # Create subplots for all maps
        n_maps = len(maps)
        n_cols = 3
        n_rows = (n_maps + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, map_file in enumerate(maps):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            map_data = self.df[self.df['map'] == map_file]
            map_name = MAPS.get(map_file, map_file.replace('.map', ''))
            
            # Plot each algorithm with agent count series connected by lines
            for alg_id in sorted(map_data['algorithm'].unique()):
                alg_data = map_data[map_data['algorithm'] == alg_id]
                alg_name = ALGORITHMS.get(alg_id, alg_id)
                style = ALGORITHM_STYLES.get(alg_id, {"color": "black", "marker": "o", "linestyle": "-", "linewidth": 2})
                
                # Group by agent count and take mean
                grouped = alg_data.groupby('agents').agg({
                    'runtime': 'mean',
                    'flow_time_ratio': 'mean'
                }).reset_index().sort_values('agents')
                
                # Plot with agent count series connected
                ax.plot(grouped['runtime'], grouped['flow_time_ratio'], 
                       marker=style['marker'], linestyle=style['linestyle'],
                       color=style['color'], linewidth=style['linewidth'],
                       markersize=8, markerfacecolor='white', markeredgecolor=style['color'],
                       markeredgewidth=2, label=alg_name)
                
                # Add agent count labels
                for _, row in grouped.iterrows():
                    ax.annotate(f'{int(row["agents"])}', 
                              (row['runtime'], row['flow_time_ratio']),
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=10, ha='left')
            
            ax.set_xlabel('Runtime (seconds)', fontweight='bold')
            ax.set_ylabel('Flow Time / Lower Bound', fontweight='bold')
            ax.set_title(f'{map_name}', fontweight='bold', fontsize=14)
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            ax.spines['top'].set_linewidth(1.0)
            ax.spines['right'].set_linewidth(1.0)
            
            if i == 0:  # Only show legend on first subplot
                ax.legend(loc='best', frameon=True, edgecolor='black')
            
            # Set reasonable axis limits
            if not map_data.empty:
                ax.set_xlim(left=0)
                ax.set_ylim(bottom=1.0)  # Ratio should be >= 1
        
        # Hide empty subplots
        for i in range(len(maps), n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if save_plots:
            output_file = self.output_dir / "runtime_vs_flow_ratio_by_map.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='black')
            print(f"Saved plot: {output_file}")
        
        if save_plots:
            plt.close()
        else:
            plt.show()
        return fig
    
    def plot_runtime_vs_flow_ratio_by_agents(self, save_plots: bool = True):
        """Create runtime vs flow time ratio plots for each agent count."""
        
        agent_counts = sorted(self.df['agents'].unique())
        
        # Create subplots for different agent counts
        n_agents = len(agent_counts)
        n_cols = 3
        n_rows = (n_agents + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, agent_count in enumerate(agent_counts):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            agent_data = self.df[self.df['agents'] == agent_count]
            
            # Plot each algorithm
            for alg_id in agent_data['algorithm'].unique():
                alg_data = agent_data[agent_data['algorithm'] == alg_id]
                alg_name = ALGORITHMS.get(alg_id, alg_id)
                
                # Scatter plot with some jitter
                x = alg_data['runtime']
                y = alg_data['flow_time_ratio']
                
                ax.scatter(x, y, label=alg_name, alpha=0.7, s=30)
            
            ax.set_xlabel('Runtime (seconds)')
            ax.set_ylabel('Flow Time / Lower Bound')
            ax.set_title(f'{agent_count} Agents')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Set reasonable axis limits
            if not agent_data.empty:
                ax.set_xlim(left=0)
                ax.set_ylim(bottom=1.0)
        
        # Hide empty subplots
        for i in range(len(agent_counts), n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if save_plots:
            output_file = self.output_dir / "runtime_vs_flow_ratio_by_agents.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved plot: {output_file}")
        
        if save_plots:
            plt.close()
        else:
            plt.show()
        return fig
    
    def plot_algorithm_comparison(self, save_plots: bool = True):
        """Create a comprehensive comparison plot of all algorithms."""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Runtime vs Flow Ratio (all data)
        ax1 = axes[0, 0]
        for alg_id in sorted(self.df['algorithm'].unique()):
            alg_data = self.df[self.df['algorithm'] == alg_id]
            alg_name = ALGORITHMS.get(alg_id, alg_id)
            style = ALGORITHM_STYLES.get(alg_id, {"color": "black", "marker": "o"})
            
            ax1.scatter(alg_data['runtime'], alg_data['flow_time_ratio'], 
                       label=alg_name, alpha=0.7, s=30, 
                       color=style['color'], marker=style['marker'], 
                       edgecolors='black', linewidth=0.5)
        
        ax1.set_xlabel('Runtime (seconds)', fontweight='bold')
        ax1.set_ylabel('Flow Time / Lower Bound', fontweight='bold')
        ax1.set_title('Runtime vs Flow Time Ratio', fontweight='bold', fontsize=14)
        ax1.spines['top'].set_visible(True)
        ax1.spines['right'].set_visible(True)
        ax1.legend(frameon=True, edgecolor='black')
        ax1.set_xlim(left=0)
        ax1.set_ylim(bottom=1.0)
        
        # Plot 2: Average runtime by agent count
        ax2 = axes[0, 1]
        agent_summary = self.df.groupby(['algorithm', 'agents'])['runtime'].mean().reset_index()
        
        for alg_id in sorted(agent_summary['algorithm'].unique()):
            alg_data = agent_summary[agent_summary['algorithm'] == alg_id]
            alg_name = ALGORITHMS.get(alg_id, alg_id)
            style = ALGORITHM_STYLES.get(alg_id, {"color": "black", "marker": "o", "linestyle": "-", "linewidth": 2})
            
            ax2.plot(alg_data['agents'], alg_data['runtime'], 
                    marker=style['marker'], linestyle=style['linestyle'],
                    color=style['color'], linewidth=style['linewidth'],
                    markersize=8, markerfacecolor='white', markeredgecolor=style['color'],
                    markeredgewidth=2, label=alg_name)
        
        ax2.set_xlabel('Number of Agents', fontweight='bold')
        ax2.set_ylabel('Average Runtime (seconds)', fontweight='bold')
        ax2.set_title('Scalability: Runtime vs Agent Count', fontweight='bold', fontsize=14)
        ax2.spines['top'].set_visible(True)
        ax2.spines['right'].set_visible(True)
        ax2.legend(frameon=True, edgecolor='black')
        
        # Plot 3: Average flow time ratio by agent count
        ax3 = axes[1, 0]
        flow_summary = self.df.groupby(['algorithm', 'agents'])['flow_time_ratio'].mean().reset_index()
        
        for alg_id in sorted(flow_summary['algorithm'].unique()):
            alg_data = flow_summary[flow_summary['algorithm'] == alg_id]
            alg_name = ALGORITHMS.get(alg_id, alg_id)
            style = ALGORITHM_STYLES.get(alg_id, {"color": "black", "marker": "o", "linestyle": "-", "linewidth": 2})
            
            ax3.plot(alg_data['agents'], alg_data['flow_time_ratio'], 
                    marker=style['marker'], linestyle=style['linestyle'],
                    color=style['color'], linewidth=style['linewidth'],
                    markersize=8, markerfacecolor='white', markeredgecolor=style['color'],
                    markeredgewidth=2, label=alg_name)
        
        ax3.set_xlabel('Number of Agents', fontweight='bold')
        ax3.set_ylabel('Average Flow Time / Lower Bound', fontweight='bold')
        ax3.set_title('Solution Quality vs Agent Count', fontweight='bold', fontsize=14)
        ax3.spines['top'].set_visible(True)
        ax3.spines['right'].set_visible(True)
        ax3.legend(frameon=True, edgecolor='black')
        ax3.set_ylim(bottom=1.0)
        
        # Plot 4: Success rate by agent count
        ax4 = axes[1, 1]
        success_summary = self.df.groupby(['algorithm', 'agents']).size().reset_index(name='total_runs')
        
        for alg_id in sorted(success_summary['algorithm'].unique()):
            alg_data = success_summary[success_summary['algorithm'] == alg_id]
            alg_name = ALGORITHMS.get(alg_id, alg_id)
            style = ALGORITHM_STYLES.get(alg_id, {"color": "black", "marker": "o", "linestyle": "-", "linewidth": 2})
            
            # Calculate success rate (assuming equal number of experiments per algorithm)
            success_rate = alg_data['total_runs'] / alg_data['total_runs'].max() * 100
            
            ax4.plot(alg_data['agents'], success_rate, 
                    marker=style['marker'], linestyle=style['linestyle'],
                    color=style['color'], linewidth=style['linewidth'],
                    markersize=8, markerfacecolor='white', markeredgecolor=style['color'],
                    markeredgewidth=2, label=alg_name)
        
        ax4.set_xlabel('Number of Agents', fontweight='bold')
        ax4.set_ylabel('Success Rate (%)', fontweight='bold')
        ax4.set_title('Success Rate vs Agent Count', fontweight='bold', fontsize=14)
        ax4.spines['top'].set_visible(True)
        ax4.spines['right'].set_visible(True)
        ax4.legend(frameon=True, edgecolor='black')
        ax4.set_ylim(0, 105)
        
        plt.tight_layout()
        
        if save_plots:
            output_file = self.output_dir / "algorithm_comparison.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='black')
            print(f"Saved plot: {output_file}")
        
        if save_plots:
            plt.close()
        else:
            plt.show()
        return fig
    
    def generate_summary_table(self):
        """Generate a summary table of benchmark results."""
        
        summary = self.df.groupby(['algorithm', 'agents']).agg({
            'runtime': ['mean', 'std', 'count'],
            'flow_time_ratio': ['mean', 'std'],
            'success': 'sum'
        }).round(3)
        
        # Flatten column names
        summary.columns = ['_'.join(col).strip() for col in summary.columns]
        
        # Save to CSV
        summary_file = self.output_dir / "benchmark_summary.csv"
        summary.to_csv(summary_file)
        print(f"Summary table saved to {summary_file}")
        
        return summary
    
    def plot_all(self):
        """Generate all plots."""
        print("Generating runtime vs flow ratio plots by map...")
        self.plot_runtime_vs_flow_ratio_by_map()
        
        print("Generating runtime vs flow ratio plots by agent count...")
        self.plot_runtime_vs_flow_ratio_by_agents()
        
        print("Generating algorithm comparison plots...")
        self.plot_algorithm_comparison()
        
        print("Generating summary table...")
        self.generate_summary_table()
        
        print(f"All plots saved to {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Plot MAPF benchmark results")
    parser.add_argument("results_file", 
                       help="Path to benchmark results CSV or JSON file")
    parser.add_argument("--output-dir", default="plots",
                       help="Output directory for plots")
    parser.add_argument("--plot-type", choices=["all", "by_map", "by_agents", "comparison"],
                       default="all", help="Type of plots to generate")
    
    args = parser.parse_args()
    
    if not Path(args.results_file).exists():
        print(f"Results file not found: {args.results_file}")
        return 1
    
    plotter = BenchmarkPlotter(args.results_file, args.output_dir)
    
    if args.plot_type == "all":
        plotter.plot_all()
    elif args.plot_type == "by_map":
        plotter.plot_runtime_vs_flow_ratio_by_map()
    elif args.plot_type == "by_agents":
        plotter.plot_runtime_vs_flow_ratio_by_agents()
    elif args.plot_type == "comparison":
        plotter.plot_algorithm_comparison()
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())