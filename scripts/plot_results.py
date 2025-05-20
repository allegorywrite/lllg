import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse # Import argparse

# Apply styling improvements
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 18

def plot_summary_success_rate(df, output_dir):
    """Plots the overall success rate as a bar chart."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    success_rate_all = df.groupby(["num_agents", "gg_enabled"])["solved"].mean().reset_index()
    success_rate_all["solved"] = success_rate_all["solved"] * 100 # Convert to percentage

    plt.figure(figsize=(10, 6))
    sns.barplot(x="num_agents", y="solved", hue="gg_enabled", data=success_rate_all,
                palette={True: "blue", False: "red"}, dodge=True) # Use blue/red consistently
    plt.title("Overall Success Rate Comparison")
    plt.xlabel("Number of Agents")
    plt.ylabel("Success Rate (%)")
    plt.ylim(0, 105)
    
    # Handle legend
    handles, labels = plt.gca().get_legend_handles_labels()
    label_map = {'True': 'Global Guidance', 'False': 'Local Guidance'}
    unique_handles_labels = {label: handle for handle, label in zip(handles, labels) if label in label_map}
    if unique_handles_labels:
        plt.legend([unique_handles_labels[label] for label in unique_handles_labels],
                   [label_map[label] for label in unique_handles_labels],
                   title="Guidance Type")

    plt.grid(True, which="major", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_success_rate.png"), dpi=150)
    plt.close()

def plot_summary_boxplot(df, metric_column, y_label, title, filename_prefix, output_dir):
    """Plots overall metric distribution as a box plot for solved instances."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    solved_df = df[df["solved"] == 1].copy()
    if solved_df.empty:
        print(f"No solved instances to plot {metric_column}.")
        return

    plt.figure(figsize=(10, 6))
    sns.boxplot(x="num_agents", y=metric_column, hue="gg_enabled", data=solved_df,
                palette={True: "blue", False: "red"}, dodge=True, showfliers=False) # Set showfliers=False
    plt.title(title)
    plt.xlabel("Number of Agents")
    plt.ylabel(y_label)

    # Handle legend
    handles, labels = plt.gca().get_legend_handles_labels()
    label_map = {'True': 'Global Guidance', 'False': 'Local Guidance'}
    unique_handles_labels = {label: handle for handle, label in zip(handles, labels) if label in label_map}
    if unique_handles_labels:
        plt.legend([unique_handles_labels[label] for label in unique_handles_labels],
                   [label_map[label] for label in unique_handles_labels],
                   title="Guidance Type")

    plt.grid(True, which="major", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"summary_{filename_prefix}_boxplot.png"), dpi=150)
    plt.close()


def plot_line_per_scenario(df, metric_column, y_label, title, filename_prefix, output_dir):
    """
    Plots a line for each scenario, showing the metric against the number of agents.
    Lines are styled by gg_enabled. (Previous implementation)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df['scenario_unique_id'] = df['scenario_type'] + "_" + df['scenario_id'].astype(str)
    plt.figure(figsize=(16, 9))
    sns.lineplot(data=df, x="num_agents", y=metric_column, hue="gg_enabled",
                 style="gg_enabled", units="scenario_unique_id", estimator=None, lw=0.5,
                 palette={True: "blue", False: "red"}, markers=True, markersize=4, dashes=True)
    plt.title(title)
    plt.xlabel("Number of Agents")
    plt.ylabel(y_label)

    handles, labels = plt.gca().get_legend_handles_labels()
    label_map = {'True': 'Global Guidance', 'False': 'Local Guidance'}
    unique_handles_labels = {}
    for handle, label in zip(handles, labels):
        if label in label_map and label not in unique_handles_labels:
             unique_handles_labels[label] = handle
    if unique_handles_labels:
        plt.legend([unique_handles_labels[label] for label in unique_handles_labels],
                   [label_map[label] for label in unique_handles_labels],
                   title="Guidance Type")

    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    agent_ticks = sorted(df['num_agents'].unique())
    plt.xticks(agent_ticks)
    plt.savefig(os.path.join(output_dir, f"{filename_prefix}.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot experiment results.")
    parser.add_argument('--mode', type=str, default='summary', choices=['summary', 'per_scenario'],
                        help='Plotting mode: "summary" for overall boxplots/barplots, "per_scenario" for lines per scenario.')
    args = parser.parse_args()

    csv_file = "experiment_summary.csv"
    
    if args.mode == 'summary':
        plots_dir = "plots_summary"
    else: # per_scenario
        plots_dir = "plots_per_scenario_lines"

    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: {csv_file} not found. Please run parse_results.py first.")
        return
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        return

    if df['gg_enabled'].dtype != 'bool':
        df['gg_enabled'] = df['gg_enabled'].astype(bool)

    if args.mode == 'summary':
        print("Generating summary plots (Success Rate Bar Chart, Runtime/Cost Box Plots)...")
        # 1. Success Rate (Bar Chart)
        plot_summary_success_rate(df, plots_dir)

        # 2. Runtime (Box Plot)
        plot_summary_boxplot(df, "comp_time", "Runtime (comp_time)", 
                             "Overall Runtime Distribution (Solved Instances)", 
                             "runtime", plots_dir)
        
        # 3. Cost (SOC) (Box Plot)
        plot_summary_boxplot(df, "soc", "Sum of Costs (SOC)", 
                             "Overall Sum of Costs Distribution (Solved Instances)", 
                             "soc", plots_dir)
        print(f"Summary plots saved to '{plots_dir}' directory.")

    elif args.mode == 'per_scenario':
        print("Generating plots with one line per scenario...")
        # 1. Success Rate (Line per scenario)
        df_success = df.copy()
        df_success['success_rate_val'] = df_success['solved'] * 100
        plot_line_per_scenario(df_success,
                               metric_column="success_rate_val",
                               y_label="Success (100=Solved, 0=Failed)",
                               title="Success per Scenario vs. Number of Agents",
                               filename_prefix="success_rate_per_scenario",
                               output_dir=plots_dir)

        # Filter for solved instances for runtime and cost plots
        solved_df = df[df["solved"] == 1].copy()
        if not solved_df.empty:
            # 2. Runtime (Line per scenario)
            plot_line_per_scenario(solved_df,
                                   metric_column="comp_time",
                                   y_label="Runtime (comp_time)",
                                   title="Runtime per Scenario vs. Number of Agents (Solved Instances)",
                                   filename_prefix="runtime_per_scenario",
                                   output_dir=plots_dir)

            # 3. Cost (soc) (Line per scenario)
            plot_line_per_scenario(solved_df,
                                   metric_column="soc",
                                   y_label="Sum of Costs (SOC)",
                                   title="Sum of Costs per Scenario vs. Number of Agents (Solved Instances)",
                                   filename_prefix="soc_per_scenario",
                                   output_dir=plots_dir)
        else:
            print("No solved instances found to plot runtime or cost per scenario.")
        print(f"Line plots per scenario saved to '{plots_dir}' directory.")

if __name__ == "__main__":
    main()
