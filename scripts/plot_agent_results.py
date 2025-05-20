import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import re
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# Apply styling improvements from plot_results.py
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 18

def parse_data_file(file_path):
    """Parses a single data file to extract SOC, comp_time, and soc_lb."""
    data = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    data[key] = value
        
        soc = int(data.get('soc', float('nan')))
        comp_time = int(data.get('comp_time', float('nan')))
        soc_lb = int(data.get('soc_lb', float('nan'))) # Extract soc_lb
        return soc, comp_time, soc_lb
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
        return float('nan'), float('nan'), float('nan')

def load_data_for_agent_group(agent_dir_path):
    """Loads all data from a specific agent group directory (e.g., build/agent_100)."""
    all_data = []
    if not os.path.isdir(agent_dir_path):
        print(f"Directory not found: {agent_dir_path}")
        return pd.DataFrame()

    for filename in os.listdir(agent_dir_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(agent_dir_path, filename)
            
            # Extract info from filename: result_random-X_NXXX_TYPE.txt
            match = re.match(r"result_.*_N(\d+)_(gg_lg|gg|lg|vanilla)\.txt", filename)
            if match:
                num_agents = int(match.group(1))
                exp_type = match.group(2)
                
                soc, comp_time, soc_lb = parse_data_file(file_path) # Get soc_lb
                
                if not pd.isna(soc) and not pd.isna(comp_time) and not pd.isna(soc_lb):
                    all_data.append({
                        "num_agents": num_agents,
                        "type": exp_type,
                        "soc": soc,
                        "soc_lb": soc_lb, # Store soc_lb
                        "runtime": comp_time,
                        "source_file": filename
                    })
            else:
                print(f"Could not parse filename: {filename}")
    
    df_loaded = pd.DataFrame(all_data)
    if not df_loaded.empty:
        # Calculate normalized SOC
        # Handle cases where soc_lb might be zero or invalid to avoid division by zero
        df_loaded['soc_normalized'] = np.where(
            (df_loaded['soc_lb'] > 0), 
            df_loaded['soc'] / df_loaded['soc_lb'], 
            np.nan # Assign NaN if soc_lb is not positive
        )
        # Drop rows where normalization resulted in NaN, if any, or if soc_normalized is NaN
        df_loaded.dropna(subset=['soc_normalized'], inplace=True)
    return df_loaded

def plot_agent_data(df, num_agents_val, output_dir):
    """Plots Normalized SOC vs Runtime for a specific number of agents, colored by type."""
    if df.empty or 'soc_normalized' not in df.columns:
        print(f"No data with soc_normalized to plot for {num_agents_val} agents.")
        return

    plt.figure(figsize=(12, 8))
    
    # Define colors for consistency
    palette = {
        "vanilla": "blue",
        "lg": "orange",
        "gg": "green",
        "lg&gg": "red" # Assuming lg&gg is the correct representation from .clinerules
    }
    
    # Corrected type for lg&gg to match filename parsing
    df_plot = df.copy()
    df_plot['type'] = df_plot['type'].replace('gg_lg', 'lg&gg')

    ax = sns.scatterplot(data=df_plot, x="runtime", y="soc_normalized", hue="type", palette=palette, s=60, alpha=0.6, legend=True)
    
    # Draw centroid and confidence ellipse for each type
    for name, group in df_plot.groupby('type'):
        if len(group) < 2: # Need at least 2 points for covariance
            continue
            
        x_coords = group['runtime']
        y_coords = group['soc_normalized'] # Use normalized SOC for y-coordinates
        
        # Filter out NaN y_coords before processing
        valid_indices = ~np.isnan(y_coords)
        x_coords = x_coords[valid_indices]
        y_coords = y_coords[valid_indices]

        if len(y_coords) < 2: # Check again after NaN removal
            continue

        # Calculate centroid
        centroid_x = np.mean(x_coords)
        centroid_y = np.mean(y_coords)
        
        # Plot centroid
        ax.scatter(centroid_x, centroid_y, marker='X', s=100, color=palette[name], edgecolor='black', zorder=5) # Reduced marker size
        
        # Calculate covariance matrix for ellipse
        # Ensure there's variance in both dimensions, otherwise cov matrix might be singular or near-singular
        if np.isclose(np.std(x_coords), 0) or np.isclose(np.std(y_coords), 0):
            print(f"Skipping ellipse for {name} due to zero variance in one or both dimensions.")
            continue
            
        cov = np.cov(x_coords, y_coords)
        
        # Check if covariance matrix is valid for eigenvalue decomposition
        if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
            print(f"Skipping ellipse for {name} due to invalid covariance matrix: {cov}")
            continue

        try:
            eigenvalues, eigenvectors = np.linalg.eig(cov)
        except np.linalg.LinAlgError:
            print(f"Skipping ellipse for {name} due to LinAlgError in eigenvalue decomposition.")
            continue
        
        # Ensure eigenvalues are non-negative (can happen with floating point issues for near-zero variance)
        eigenvalues = np.maximum(eigenvalues, 0)
        
        # Get ellipse parameters
        # Order eigenvalues and eigenvectors
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:,order]
        
        # Angle of the ellipse (angle of the first eigenvector)
        angle = np.degrees(np.arctan2(*eigenvectors[:,0][::-1]))
        
        # Width and height of the ellipse (sqrt of eigenvalues)
        # Scale factor for 95% confidence interval (chi-squared with 2 dof)
        # For 1 std deviation, scale = 1. For 2 std deviations, scale = 2.
        # For 95% confidence ellipse, scale = sqrt(5.991) approx 2.448
        # Let's use 2 standard deviations for a common representation
        n_std = 2 
        width, height = 2 * n_std * np.sqrt(eigenvalues)
        
        # Create ellipse
        # Ellipse is centered at the centroid
        ellipse = Ellipse(xy=(centroid_x, centroid_y),
                          width=width, height=height,
                          angle=angle,
                          facecolor=palette[name], alpha=0.2, edgecolor=palette[name], linestyle='--')
        ax.add_patch(ellipse)

    plt.title(f"Normalized SOC (SOC/SOC_LB) vs Runtime for {num_agents_val} Agents")
    plt.xlabel("Runtime (comp_time)")
    plt.ylabel("Normalized SOC (SOC / SOC_LB)")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    
    handles, labels = ax.get_legend_handles_labels()
    unique_labels_dict = {}
    for handle, label in zip(handles, labels):
        if label not in unique_labels_dict:
            unique_labels_dict[label] = handle
    
    ax.legend(unique_labels_dict.values(), unique_labels_dict.keys(), title="Experiment Type")

    plt.tight_layout()
    plot_filename = os.path.join(output_dir, f"soc_normalized_vs_runtime_N{num_agents_val}.png") # Updated filename
    plt.savefig(plot_filename, dpi=150)
    print(f"Saved plot: {plot_filename}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot SOC vs Runtime for specified agent group directories.")
    parser.add_argument('agent_dirs', nargs='+', type=str, 
                        help="List of agent group directories (e.g., build/agent_100 build/agent_200).")
    parser.add_argument('--output_dir', type=str, default="plots_agent_specific",
                        help="Directory to save the plots.")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    for agent_dir_path in args.agent_dirs:
        print(f"\nProcessing directory: {agent_dir_path}")
        # Extract num_agents from directory name, e.g., "build/agent_100" -> 100
        dir_name_match = re.search(r"agent_(\d+)", agent_dir_path)
        if not dir_name_match:
            print(f"Could not determine number of agents from directory name: {agent_dir_path}")
            continue
        
        current_num_agents = int(dir_name_match.group(1))
        
        df_agent_group = load_data_for_agent_group(agent_dir_path)
        
        if not df_agent_group.empty:
            # Filter for the specific number of agents, though load_data_for_agent_group should already do this
            # This is more of a sanity check or if the directory contains mixed agent counts (it shouldn't based on structure)
            df_to_plot = df_agent_group[df_agent_group["num_agents"] == current_num_agents]
            if not df_to_plot.empty:
                plot_agent_data(df_to_plot, current_num_agents, args.output_dir)
            else:
                print(f"No data found for {current_num_agents} agents in {agent_dir_path} after filtering.")
        else:
            print(f"No data loaded from {agent_dir_path}.")

if __name__ == "__main__":
    main()
