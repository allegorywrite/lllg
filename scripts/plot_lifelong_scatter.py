#!/usr/bin/env python3
"""
Parse benchmark result files and plot a scatter:
- x-axis: comp_time per step (sec/step)
- y-axis: total_completed_tasks / makespan (default)
         or comp_time per step (via --y-axis runtime)
         or makespan/simulation_time (via --y-axis makespan)

Special case for RHCR results: use simulation_time in place of makespan.

Usage:
  python benchmark/plot_lifelong_scatter.py \
      --results benchmark/results \
      --out benchmark/results/scatter_comp_time_vs_throughput.pdf \
      [--y-axis throughput|runtime|comp_time|makespan] \
      [--marker-by agents|method] \
      [--connect-same-color] \
      [--split-by-agents] \
      [--show]

The script searches recursively under --results for *.txt files, expects
simple `key=value` pairs near the top of each file, and stops parsing when
encountering a large section (e.g. starts= or goals=).

If multiple distinct `map_file` values are found in the results, separate
figures are generated per map. Output filenames are formed by inserting the
map name (basename without extension) before the file extension of `--out`.
For example, `scatter.pdf` becomes `scatter_warehouse.pdf` and
`scatter_office.pdf`.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Literal

import matplotlib.pyplot as plt

# Match the plotting style of benchmark/plot_benchmark_results.py
plt.rcParams.update({
    'font.size': 26,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'serif'],
    'axes.linewidth': 2.0,
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

# Define algorithm styles consistent with publication-quality palette
# Colors per algorithm (marker will be chosen by agents later).
# ALGORITHM_STYLES = {
#     # 全部異なる色を割り当て (tab20 palette)
#     'lifelong_lacam': {"color": "#1f77b4", "linestyle": "-", "linewidth": 2},           # Blue
#     'lifelong_lg_lacam': {"color": "#2ca02c", "linestyle": "-", "linewidth": 2},        # Green
#     'lifelong_lacam_horizon_1': {"color": "#9467bd", "linestyle": "-", "linewidth": 2}, # Purple
#     'lifelong_lacam_horizon_20': {"color": "#8c564b", "linestyle": "-", "linewidth": 2},# Brown
#     'lifelong_lg_lacam_horizon_1': {"color": "#e377c2", "linestyle": "-", "linewidth": 2},  # Pink
#     'lifelong_lg_lacam_horizon_2': {"color": "#d62728", "linestyle": "-", "linewidth": 2},  # Red
#     'lifelong_lg_lacam_horizon_3': {"color": "#bcbd22", "linestyle": "-", "linewidth": 2},  # Olive
#     'lifelong_lg_lacam_horizon_5': {"color": "#17becf", "linestyle": "-", "linewidth": 2},  # Cyan
#     'lifelong_lg_lacam_horizon_10': {"color": "#ffbb78", "linestyle": "-", "linewidth": 2}, # Light Orange
#     'lifelong_lg_lacam_horizon_20': {"color": "#7f7f7f", "linestyle": "-", "linewidth": 2}, # Gray
#     'lifelong_lacam_lns': {"color": "#bcbd22", "linestyle": "-", "linewidth": 2},  # Olive
#     'lifelong_lg_lacam_anytime': {"color": "#17becf", "linestyle": "-", "linewidth": 2}, # Cyan

#     # Extra
#     'lifelong_lg_lacam_sort': {"color": "#aec7e8", "linestyle": "-", "linewidth": 2},   # Light Blue
#     'lifelong_lg_lacam_opt': {"color": "#ffbb78", "linestyle": "-", "linewidth": 2},    # Light Orange
#     'RHCR': {"color": "#9467bd", "linestyle": "-", "linewidth": 2}, # Purple
#     'mapf_lrr2023': {"color": "#ff7f0e", "linestyle": "-", "linewidth": 2},           # Orange
#     'guided_pibt': {"color": "#ff7f0e", "linestyle": "-", "linewidth": 2},           # Orange

#     'lifelong_lacam_plan': {"color": "#2ca02c", "linestyle": "-", "linewidth": 2},        # Green
#     'lifelong_lg_lacam_plan': {"color": "#d62728", "linestyle": "-", "linewidth": 2},        # Red
#     # 'lifelong_lacam_lns_plan': {"color": "#f7b6d2", "linestyle": "-", "linewidth": 2}, # Light Pink
#     'lifelong_lg_lacam_lns_plan': {"color": "#9467bd", "linestyle": "-", "linewidth": 2}, # Purple
    
#     'lifelong_lacam_plan_hindrance': {"color": "#ff7f0e", "linestyle": "-", "linewidth": 2},           # Orange
#     'lifelong_lg_lacam_plan_hindrance': {"color": "#d62728", "linestyle": "-", "linewidth": 2},        # Red

#     'lifelong_lg_regret_lacam': {"color": "#ff7f0e", "linestyle": "-", "linewidth": 2},           # Orange
#     'lifelong_lg_regret_lacam_control': {"color": "#d62728", "linestyle": "-", "linewidth": 2},        # Red
#     'lifelong_lg_unreached_lacam': {"color": "#d62728", "linestyle": "-", "linewidth": 2},        # Red
# }

ALGORITHM_STYLES = {
    'realtime_lacam': {"color": "#8c564b", "linestyle": "-", "linewidth": 2},# Brown
    'realtime_lg_lacam': {"color": "#ff7f0e", "linestyle": "-", "linewidth": 2},           # Orange
    'lifelong_lacam': {"color": "#2ca02c", "linestyle": "-", "linewidth": 2},        # Green
    'lifelong_lg_lacam': {"color": "#1f77b4", "linestyle": "-", "linewidth": 2},           # Blue
    'lifelong_lg_lacam_control': {"color": "#e377c2", "linestyle": "-", "linewidth": 2},  # Pink
    'lifelong_lg_lacam_refine': {"color": "#d62728", "linestyle": "-", "linewidth": 2},  # Red
    'lifelong_lg_lacam_refine_lns': {"color": "#d62728", "linestyle": "-", "linewidth": 2},  # Red
    'RHCR': {"color": "#9467bd", "linestyle": "-", "linewidth": 2}, # Purple
    'mapf_lrr2023': {"color": "#e377c2", "linestyle": "-", "linewidth": 2},  # Pink
    'guided_pibt': {"color": "#ff7f0e", "linestyle": "-", "linewidth": 2},           # Orange
    'lifelong_lg_lacam_horizon_1': {"color": "#e377c2", "linestyle": "-", "linewidth": 2},  # Pink
    'lifelong_lg_lacam_horizon_2': {"color": "#d62728", "linestyle": "-", "linewidth": 2},  # Red
    'lifelong_lg_lacam_horizon_3': {"color": "#bcbd22", "linestyle": "-", "linewidth": 2},  # Olive
    'lifelong_lg_lacam_horizon_5': {"color": "#17becf", "linestyle": "-", "linewidth": 2},  # Cyan
    'lifelong_lg_lacam_horizon_10': {"color": "#ffbb78", "linestyle": "-", "linewidth": 2}, # Light Orange
    'lifelong_lg_lacam_horizon_20': {"color": "#7f7f7f", "linestyle": "-", "linewidth": 2}, # Gray
    'lifelong_lacam_lns': {"color": "#2ca02c", "linestyle": "-", "linewidth": 2},        # Green
    'lifelong_lg_lacam_lns': {"color": "#1f77b4", "linestyle": "-", "linewidth": 2},           # Blue

    'lifelong_anytime_lacam': {"color": "#2ca02c", "linestyle": "-", "linewidth": 2},        # Green
    'lifelong_lg_anytime_lacam': {"color": "#1f77b4", "linestyle": "-", "linewidth": 2},           # Blue
    'lifelong_lg_anytime_lacam_refine': {"color": "#d62728", "linestyle": "-", "linewidth": 2},  # Red
}

# Marker assignment by agents. Keys are agent counts that appear in results.
# If an agents value is not listed, a default cycle is used.
# AGENTS_TO_MARKERS = {
#     100: 'o',   # circle
#     200: 's',   # square
#     300: 'D',   # diamond
#     400: '^',   # triangle up
#     500: 'v',   # triangle down
#     600: 'P',   # plus (filled)
#     700: 'X',   # x (filled)
#     800: '*',   # star
# }
AGENTS_TO_MARKERS = {
    100: 'X',   # x (filled)
    200: 'X',   # x (filled)
    300: '^',   # triangle up
    400: 'o',   # circle
    500: 'v',   # triangle down
    600: '^',   # triangle up
    700: 'X',   # x (filled)
    800: 's',   # square
    1000: '*',   # star
}

# Per-marker sizes in points (used for line markers) and converted to area for scatter.
# You can tune each marker's size to the most legible value.
MARKER_SIZES_PT = {
    'o': 14,
    's': 16,
    'D': 13,
    '^': 15,
    'v': 16,
    'P': 17,
    'X': 17,
    '*': 21,
    'h': 16,
    'H': 16,
    '<': 16,
    '>': 16,
    'p': 18,
    '8': 18,
    'd': 17,
}

DEFAULT_MARKER_CYCLE = ['o', 's', 'D', '^', 'v', 'P', 'X', '*', 'h']

METHOD_TO_MARKERS: Dict[str, str] = {
    # Assign markers per method (algorithm). Keys should match directory names in benchmark/results/<algo>/...
    'lifelong_lacam': 'o',
    'lifelong_lacam_lns': 'D',
    'lifelong_lg_lacam': '*',
    'lifelong_lg_lacam_anytime': '^',
    'lifelong_lg_lacam_lns': 'v',
    'lifelong_lg_lacam_horizon_1': '<',
    'lifelong_lg_lacam_horizon_2': '>',
    'lifelong_lg_lacam_horizon_3': 'p',
    'lifelong_lg_lacam_horizon_5': 'H',
    'lifelong_lg_lacam_horizon_10': '8',
    'lifelong_lg_lacam_horizon_20': 'd',
    'realtime_lacam': 'X',
    'realtime_lg_lacam': '*',
    'RHCR': '^',
    'guided_pibt': 'P',
    'mapf_lrr2023': 'D',
    'lifelong_lacam_plan': 'o',
    'lifelong_lacam_plan_hindrance': 's',
    'lifelong_lg_lacam_plan': '^',
    'lifelong_lg_lacam_plan_hindrance': 'v',
    'lifelong_lg_lacam_lns_plan': 'H',
    'lifelong_lg_regret_lacam': 'p',
    'lifelong_lg_regret_lacam_control': '8',
    'lifelong_lg_unreached_lacam': 'd',
}

def _marker_for_agents(agents: Optional[int]) -> str:
    if agents is None:
        return DEFAULT_MARKER_CYCLE[0]
    if agents in AGENTS_TO_MARKERS:
        return AGENTS_TO_MARKERS[agents]
    # Fallback: pick a marker from cycle based on numeric value
    idx = abs(int(agents)) % len(DEFAULT_MARKER_CYCLE)
    return DEFAULT_MARKER_CYCLE[idx]

def _marker_for_method(algo: str) -> str:
    mk = METHOD_TO_MARKERS.get(algo)
    if mk is not None:
        return mk
    # Deterministic fallback (no randomized hash()).
    marker_cycle = ['o', 's', 'D', '^', 'v', 'P', 'X', '*', 'h', 'H', '<', '>', 'p', '8', 'd']
    idx = sum(algo.encode('utf-8')) % len(marker_cycle)
    return marker_cycle[idx]

def _marker_size_pt(marker: str, default: float = 10.0) -> float:
    return float(MARKER_SIZES_PT.get(marker, default))

def _scatter_size_from_pt(marker: str, default_pt: float = 10.0) -> float:
    # Matplotlib scatter uses area (points^2)
    ms = _marker_size_pt(marker, default_pt)
    return ms * ms

# Pretty names for legend
ALGORITHMS = {
    # 'realtime_lacam': 'Realtime LaCAM',
    'realtime_lg_lacam': 'Realtime LG LaCAM',
    'lifelong_lacam': 'LaCAM',
    'lifelong_lg_lacam_anytime': 'Lifelong LG LaCAM MC',
    'lifelong_lacam_lns': 'Lifelong LaCAM LNS',
    # 'lifelong_lacam_horizon_1': 'Lifelong LaCAM Horizon 1',
    # 'lifelong_lacam_horizon_20': 'Lifelong LaCAM Horizon 20',
    'lifelong_lg_lacam_lns': 'Lifelong LG LaCAM LNS',
    'lifelong_lg_lacam_horizon_1': 'horizon=1',
    'lifelong_lg_lacam_horizon_2': 'horizon=2',
    # 'lifelong_lg_lacam_horizon_3': 'Lifelong LG LaCAM Horizon 3',
    'lifelong_lg_lacam_horizon_5': 'horizon=5',
    'lifelong_lg_lacam_horizon_10': 'horizon=10',
    'lifelong_lg_lacam_horizon_20': 'horizon=20',
    # 'lifelong_lg_lacam': 'full horizon',
    'lifelong_lg_lacam': 'full horizon',
    'lifelong_lg_lacam_control': 'Control',
    # 'lifelong_lg_lacam_sort': 'Lifelong LG LaCAM Sorted',
    # 'lifelong_lg_lacam_opt': 'Lifelong LG LaCAM Regularized',
    'RHCR': 'RHCR',
    'guided_pibt': 'Guided-PIBT',
    'mapf_lrr2023': 'WPPL',
    'lifelong_lacam_plan': 'Lifelong LaCAM Plan',
    'lifelong_lacam_plan_hindrance': 'Lifelong LaCAM Plan Hindrance',
    'lifelong_lg_lacam_plan': 'Lifelong LG LaCAM',
    'lifelong_lacam_refine': 'Lifelong LaCAM Refine',
    'lifelong_lg_lacam_refine': 'Lifelong LG LaCAM Refine',
    'lifelong_lg_lacam_refine_lns': 'Lifelong LG LaCAM Refine LNS',
    'lifelong_lg_lacam_plan_hindrance': 'Lifelong LG LaCAM Plan Hindrance',
    # 'lifelong_lacam_lns_plan': 'Lifelong LaCAM LNS Plan',
    'lifelong_lg_lacam_lns_plan': 'Lifelong LG LaCAM + LNS',
    # 'lifelong_lg_regret_lacam': 'Lifelong LG LaCAM Regret',
    'lifelong_lg_regret_lacam_control': 'Lifelong LG LaCAM Regret',
    'lifelong_lg_unreached_lacam': 'Lifelong LG LaCAM Unreached',
    'lifelong_anytime_lacam': 'Lifelong LaCAM Anytime',
    'lifelong_lg_anytime_lacam': 'Lifelong LG LaCAM Anytime',
    'lifelong_lg_anytime_lacam_refine': 'Lifelong LG LaCAM Refine Anytime',
}


@dataclass
class ResultPoint:
    algo: str
    path: str
    comp_time: float
    total_completed_tasks: float
    makespan: float  # or simulation_time for RHCR
    agents: Optional[int] = None
    map_file: Optional[str] = None
    lb_sp_task_count: Optional[float] = None

    @property
    def throughput(self) -> float:
        if self.makespan == 0:
            return float('nan')
        return self.total_completed_tasks / self.makespan

    @property
    def throughput_upper_bound(self) -> float:
        if self.makespan == 0 or self.lb_sp_task_count is None:
            return float('nan')
        return self.lb_sp_task_count / self.makespan

    @property
    def runtime_per_step(self) -> float:
        if self.makespan == 0:
            return float('nan')
        return self.comp_time / self.makespan


def parse_header_value(line: str) -> Optional[tuple[str, str]]:
    """Parse a simple `key=value` line; return (key, value) or None."""
    line = line.strip()
    if not line or '=' not in line:
        return None
    # Ignore noisy sections
    if line.startswith('starts=') or line.startswith('goals='):
        return ('__STOP__', '')
    key, val = line.split('=', 1)
    return key.strip(), val.strip()


def is_rhcr_path(path: str) -> bool:
    parts = os.path.normpath(path).split(os.sep)
    # Expect structure like benchmark/results/<algo>/file.txt
    return any(part == 'RHCR' for part in parts)


def parse_result_file(path: str) -> Optional[ResultPoint]:
    algo_name = os.path.basename(os.path.dirname(path))  # parent dir name

    comp_time: Optional[float] = None
    total_completed_tasks: Optional[float] = None
    makespan: Optional[float] = None
    sim_time: Optional[float] = None
    agents: Optional[int] = None
    map_file: Optional[str] = None
    lb_sp_task_count: Optional[float] = None

    try:
        with open(path, 'r') as f:
            # Only read top portion; break early when big sections begin
            for idx, raw in enumerate(f):
                parsed = parse_header_value(raw)
                if parsed is None:
                    continue
                key, val = parsed
                if key == '__STOP__':
                    break
                if key == 'comp_time':
                    try:
                        # Input comp_time is in milliseconds; convert to seconds for plotting
                        comp_time = float(val) / 1000.0
                    except ValueError:
                        pass
                elif key == 'total_completed_tasks':
                    try:
                        total_completed_tasks = float(val)
                    except ValueError:
                        pass
                elif key == 'makespan':
                    try:
                        makespan = float(val)
                    except ValueError:
                        pass
                elif key == 'simulation_time':
                    try:
                        sim_time = float(val)
                    except ValueError:
                        pass
                elif key == 'agents':
                    try:
                        agents = int(float(val))
                    except ValueError:
                        pass
                elif key == 'map_file':
                    map_file = val
                elif key == 'lb_sp_task_count':
                    try:
                        lb_sp_task_count = float(val)
                    except ValueError:
                        pass
                # Early exit if we already have what we need
                if comp_time is not None and total_completed_tasks is not None and (makespan is not None or sim_time is not None):
                    # Keep reading a bit more in case ordering varies, but avoid huge files
                    if idx > 50:  # safety bound
                        break
    except (OSError, UnicodeDecodeError):
        return None

    # Choose makespan: RHCR uses simulation_time
    chosen_makespan: Optional[float]
    if is_rhcr_path(path):
        chosen_makespan = sim_time if sim_time is not None else makespan
    else:
        chosen_makespan = makespan if makespan is not None else sim_time

    if comp_time is None or total_completed_tasks is None or chosen_makespan is None:
        return None

    return ResultPoint(
        algo=algo_name,
        path=path,
        comp_time=comp_time,
        total_completed_tasks=total_completed_tasks,
        makespan=chosen_makespan,
        agents=agents,
        map_file=map_file,
        lb_sp_task_count=lb_sp_task_count,
    )


def collect_results(results_root: str) -> List[ResultPoint]:
    points: List[ResultPoint] = []
    for dirpath, _, filenames in os.walk(results_root):
        for fn in filenames:
            if not fn.endswith('.txt'):
                continue
            full = os.path.join(dirpath, fn)
            rp = parse_result_file(full)
            if rp is not None:
                points.append(rp)
    return points


def _aggregate_by_agents(points: List[ResultPoint]) -> Dict[Tuple[str, int], Tuple[float, float, Optional[float], float]]:
    """Return mean (runtime_per_step, throughput, throughput_ub, makespan) by (algo, agents)."""
    # (sum_rtps, sum_th, cnt, sum_ub, cnt_ub, sum_makespan)
    sums: Dict[Tuple[str, int], Tuple[float, float, int, float, int, float]] = {}
    for p in points:
        if p.agents is None:
            continue
        if p.makespan == 0:
            continue
        key = (p.algo, int(p.agents))
        rtps = p.runtime_per_step
        th = p.throughput
        ub = p.throughput_upper_bound
        ms = p.makespan
        ub_valid = not math.isnan(ub) and math.isfinite(ub)
        if key not in sums:
            sums[key] = (rtps, th, 1, (ub if ub_valid else 0.0), (1 if ub_valid else 0), ms)
        else:
            acc_rtps, acc_th, cnt, acc_ub, cnt_ub, acc_ms = sums[key]
            sums[key] = (
                acc_rtps + rtps,
                acc_th + th,
                cnt + 1,
                acc_ub + (ub if ub_valid else 0.0),
                cnt_ub + (1 if ub_valid else 0),
                acc_ms + ms,
            )

    means: Dict[Tuple[str, int], Tuple[float, float, Optional[float], float]] = {}
    for k, (acc_rtps, acc_th, cnt, acc_ub, cnt_ub, acc_ms) in sums.items():
        means[k] = (
            acc_rtps / cnt,
            acc_th / cnt,
            (acc_ub / cnt_ub) if cnt_ub > 0 else None,
            acc_ms / cnt,
        )
    return means


YAxisMode = Literal['throughput', 'runtime', 'comp_time', 'makespan']
LegendMode = Literal['separate', 'inline']
MarkerBy = Literal['agents', 'method']


def plot(
    points: List[ResultPoint],
    out_path: str,
    show: bool = False,
    hide_labels: bool = False,
    x_axis_mode: str = 'runtime',
    y_axis_mode: YAxisMode = 'throughput',
    hide_upper_bound: bool = False,
    annotate_rightmost_only: bool = False,
    legend_mode: LegendMode = 'separate',
    marker_by: MarkerBy = 'agents',
    connect_same_color: bool = False,
) -> None:
    if not points:
        print('[WARN] No valid result points found to plot.')
        return

    # Filter: only plot algorithms explicitly listed in ALGORITHMS
    allowed_algos = set(ALGORITHMS.keys())
    points = [p for p in points if p.algo in allowed_algos]
    if not points:
        print('[WARN] No points match ALGORITHMS filter; nothing to plot.')
        return

    # Group by algorithm for coloring/legend
    by_algo: Dict[str, List[ResultPoint]] = {}
    for p in points:
        by_algo.setdefault(p.algo, []).append(p)

    # fig, ax = plt.subplots(1, 1, figsize=(4, 8))
    fig, ax = plt.subplots(1, 1, figsize=(6, 5.9))
    # fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    algos_in_plot = [a for a in ALGORITHMS.keys() if a in by_algo]

    agent_ticks: List[int] = []
    if x_axis_mode == 'agents':
        agent_ticks = sorted({int(p.agents) for p in points if p.agents is not None})

    def _legend_out_path(plot_path: str) -> str:
        stem, ext = os.path.splitext(plot_path)
        if not ext:
            ext = '.pdf'
        return f"{stem}_legend{ext}"

    def _save_horizontal_legend(plot_path: str, handles: List[object], labels: List[str]) -> None:
        if not handles or not labels:
            return
        n = len(labels)
        fig_w = max(6.0, 2.4 * n)
        fig_h = 1.2
        fig_legend = plt.figure(figsize=(fig_w, fig_h))
        ax_legend = fig_legend.add_subplot(111)
        ax_legend.axis('off')
        ax_legend.legend(
            handles,
            labels,
            loc='center',
            ncol=n,
            frameon=False,
            handlelength=2.0,
            columnspacing=1.2,
            handletextpad=0.6,
        )
        try:
            fig_legend.tight_layout()
        except Exception:
            pass
        legend_path = _legend_out_path(plot_path)
        try:
            fig_legend.savefig(legend_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='black')
            print(f'[OK] Saved legend to: {legend_path}')
        except Exception as e:
            print(f'[ERR] Failed to save legend figure: {e}')
        plt.close(fig_legend)

    def _y_value(p: ResultPoint) -> float:
        if y_axis_mode in ('runtime', 'comp_time'):
            return p.runtime_per_step
        if y_axis_mode == 'makespan':
            return p.makespan
        return p.throughput

    # 1) Raw scatter per algorithm (faint), with marker chosen by agents/method
    any_upper_bounds = False
    for algo in (a for a in ALGORITHMS.keys() if a in by_algo):
        plist = by_algo[algo]
        style = ALGORITHM_STYLES.get(algo, {"color": "black", "linestyle": "-", "linewidth": 2})
        # Plot each point to allow agent-based markers and sizes
        for p in plist:
            mk = 'o'
            if x_axis_mode != 'agents':
                if marker_by == 'method':
                    mk = _marker_for_method(p.algo)
                else:
                    mk = _marker_for_agents(p.agents)
            
            x_val = p.runtime_per_step
            if x_axis_mode == 'agents':
                if p.agents is None:
                    continue
                x_val = float(p.agents)

            ax.scatter(
                [x_val],
                [_y_value(p)],
                alpha=0.3,
                s=_scatter_size_from_pt(mk, default_pt=9.0),
                color=style['color'],
                marker=mk,
                edgecolors=('none' if x_axis_mode == 'agents' else 'black'),
                linewidth=(0.0 if x_axis_mode == 'agents' else 0.3),
                label=None,
            )
            if (y_axis_mode == 'throughput') and (not hide_upper_bound):
                ub = p.throughput_upper_bound
                if not math.isnan(ub) and math.isfinite(ub):
                    any_upper_bounds = True
                    ax.scatter(
                        [x_val],
                        [ub],
                        alpha=0.25,
                        s=_scatter_size_from_pt(mk, default_pt=9.0),
                        facecolors='none',
                        edgecolors=style['color'],
                        marker=mk,
                        linewidth=1.0,
                        label=None,
                        zorder=2,
                    )

    # 2) Aggregate by (algo, agents). Plot aggregated points with
    #    marker chosen by agents (size per marker). Then, for each
    #    agents value, connect aggregated points left-to-right with a
    #    light gray line.
    means = _aggregate_by_agents(points)
    if means:
        # Build sorted series per algorithm for colored algorithm lines
        from collections import defaultdict
        series_by_algo = defaultdict(list)  # algo -> list of (n, x, y)
        series_ub_by_algo = defaultdict(list)  # algo -> list of (n, x, ub)
        for (algo, n), (avg_runtime_per_step, avg_throughput, avg_throughput_ub, avg_makespan) in means.items():
            if n is None:
                continue
            
            x_val = avg_runtime_per_step
            if x_axis_mode == 'agents':
                x_val = float(n)
                
            if y_axis_mode == 'throughput':
                y_val = avg_throughput
            elif y_axis_mode == 'makespan':
                y_val = avg_makespan
            else:  # runtime/comp_time
                y_val = avg_runtime_per_step
            series_by_algo[algo].append((int(n), x_val, y_val))
            if (
                (y_axis_mode == 'throughput')
                and (not hide_upper_bound)
                and avg_throughput_ub is not None
                and math.isfinite(avg_throughput_ub)
            ):
                series_ub_by_algo[algo].append((int(n), x_val, avg_throughput_ub))

        # Identify the rightmost algorithm (largest max X) if requested
        rightmost_algo = None
        if annotate_rightmost_only:
            max_x_global = -1.0
            for algo, series in series_by_algo.items():
                if not series:
                    continue
                # series items are (n, x, y)
                max_x_local = max(s[1] for s in series)
                if max_x_local > max_x_global:
                    max_x_global = max_x_local
                    rightmost_algo = algo

        # Draw algorithm lines (color-coded, no markers)
        for algo in (a for a in ALGORITHMS.keys() if a in series_by_algo):
            series = series_by_algo[algo]
            # Sort by agents (index 0) so the line follows the agent count progression
            series_sorted = sorted(series, key=lambda t: t[0])
            xs = [x for (_n, x, _y) in series_sorted]
            ys = [y for (_n, _x, y) in series_sorted]
            style = ALGORITHM_STYLES.get(algo, {"color": "black", "linestyle": "-", "linewidth": 2})
            ax.plot(
                xs,
                ys,
                linestyle=style['linestyle'],
                color=style['color'],
                linewidth=style['linewidth'],
                label=algo,
                zorder=1,
            )

        # Draw upper-bound lines (same color, dashed), when available.
        if (y_axis_mode == 'throughput') and (not hide_upper_bound):
            for algo, series in series_ub_by_algo.items():
                if len(series) < 2:
                    continue
                series_sorted = sorted(series, key=lambda t: t[0])
                xs = [x for (_n, x, _ub) in series_sorted]
                ys = [ub for (_n, _x, ub) in series_sorted]
                style = ALGORITHM_STYLES.get(algo, {"color": "black", "linestyle": "-", "linewidth": 2})
                ax.plot(
                    xs,
                    ys,
                    linestyle='--',
                    color=style['color'],
                    linewidth=max(1.0, float(style.get('linewidth', 2)) * 0.75),
                    alpha=0.8,
                    label=None,
                    zorder=1,
                )

        # Connect same-color algorithms per agent count with dashed lines.
        # This is only meaningful for x-axis=runtime (different X values per method);
        # for x-axis=agents it would produce vertical clutter.
        if connect_same_color and x_axis_mode == 'runtime':
            by_agents_by_color: Dict[int, Dict[str, List[Tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
            for (algo, n), (avg_runtime_per_step, avg_throughput, _avg_throughput_ub, avg_makespan) in means.items():
                if n is None:
                    continue
                style = ALGORITHM_STYLES.get(algo, {"color": "black"})
                color = style.get("color", "black")
                x_val = avg_runtime_per_step
                if y_axis_mode == 'throughput':
                    y_val = avg_throughput
                elif y_axis_mode == 'makespan':
                    y_val = avg_makespan
                else:  # runtime/comp_time
                    y_val = avg_runtime_per_step
                by_agents_by_color[int(n)][str(color)].append((x_val, y_val))

            for _n, by_color in by_agents_by_color.items():
                for color, xy_list in by_color.items():
                    if len(xy_list) < 2:
                        continue
                    xy_sorted = sorted(xy_list, key=lambda t: t[0])
                    xs = [t[0] for t in xy_sorted]
                    ys = [t[1] for t in xy_sorted]
                    ax.plot(
                        xs,
                        ys,
                        linestyle='--',
                        color=color,
                        linewidth=1.2,
                        alpha=0.55,
                        label=None,
                        zorder=0,
                    )

        # Plot aggregated points with per-agent/per-method markers and sizes (no legend)
        for (algo, n), (avg_runtime_per_step, avg_throughput, avg_throughput_ub, avg_makespan) in means.items():
            style = ALGORITHM_STYLES.get(algo, {"color": "black", "linestyle": "-", "linewidth": 2})
            mk = 'o'
            if x_axis_mode != 'agents':
                if marker_by == 'method':
                    mk = _marker_for_method(algo)
                else:
                    mk = _marker_for_agents(n)
            ms_pt = _marker_size_pt(mk, default=10.0)
            
            x_val = avg_runtime_per_step
            if x_axis_mode == 'agents':
                x_val = float(n)

            if y_axis_mode == 'throughput':
                y_val = avg_throughput
            elif y_axis_mode == 'makespan':
                y_val = avg_makespan
            else:  # runtime/comp_time
                y_val = avg_runtime_per_step

            marker_face_color = 'white'
            marker_edge_color = style['color']
            marker_edge_width = 2
            if x_axis_mode == 'agents':
                marker_face_color = style['color']
                marker_edge_color = style['color']
                marker_edge_width = 0.0
            
            ax.plot(
                [x_val], [y_val],
                linestyle='None',
                marker=mk,
                color=style['color'],
                markersize=ms_pt,
                markerfacecolor=marker_face_color,
                markeredgecolor=marker_edge_color,
                markeredgewidth=marker_edge_width,
                label=None,
                zorder=3,
            )
            if (y_axis_mode == 'throughput') and (not hide_upper_bound):
                if avg_throughput_ub is not None and math.isfinite(avg_throughput_ub):
                    any_upper_bounds = True
                    ax.plot(
                        [x_val], [avg_throughput_ub],
                        linestyle='None',
                        marker=mk,
                        color=style['color'],
                        markersize=ms_pt,
                        markerfacecolor='none',
                        markeredgecolor=style['color'],
                        markeredgewidth=2,
                        alpha=0.9,
                        label=None,
                        zorder=3,
                    )
            # Annotate agent counts near each aggregated point (unless hidden)
            # If x-axis is agents, annotating agents might be redundant, but keep it for consistency or valid check
            # Actually if x-axis is agents, the label "100" at x=100 is very redundant.
            should_annotate = not hide_labels
            if x_axis_mode == 'agents':
                should_annotate = False
            
            if should_annotate and annotate_rightmost_only:
                if algo != rightmost_algo:
                    should_annotate = False

            if should_annotate:
                ax.annotate(
                    f'{int(n)}',
                    (x_val, y_val),
                    # xytext=(-25, 12),
                    xytext=(-5, -30),
                    textcoords='offset points',
                    fontsize=22,
                    ha='left',
                )

        # Connect points with the same agents value from left to right (light gray)
        # Only relevant if x-axis is runtime. If x-axis is agents, points with same agents are at same X, so line is vertical?
        # Usually this line is to compare algorithms at same agent count.
        # If x-axis is agents, then "same agents" means same X coordinate. The algorithms would be stacked vertically.
        # It might be less useful or confusing to draw those connection lines if x-axis is agents.
        if x_axis_mode == 'runtime':
            series_by_n = defaultdict(list)  # n -> list of (x, y)
            for (_algo, n), (avg_runtime_per_step, avg_throughput, _avg_throughput_ub, avg_makespan) in means.items():
                if n is None:
                    continue
                if y_axis_mode == 'throughput':
                    y_val = avg_throughput
                elif y_axis_mode == 'makespan':
                    y_val = avg_makespan
                else:  # runtime/comp_time
                    y_val = avg_runtime_per_step
                series_by_n[int(n)].append((avg_runtime_per_step, y_val))
            for _n, xy_list in series_by_n.items():
                if len(xy_list) < 2:
                    continue
                xy_sorted = sorted(xy_list, key=lambda t: t[0])
                xs = [t[0] for t in xy_sorted]
                ys = [t[1] for t in xy_sorted]
                # ax.plot(
                #     xs,
                #     ys,
                #     color='#B0B0B0',  # light gray
                #     linewidth=1.0,
                #     linestyle='-',
                #     alpha=0.7,
                #     zorder=0,
                # )

    ax.set_xscale('log')
    # if x_axis_mode == 'runtime':
    #     ax.set_xlabel('runtime [sec]', fontweight='bold')
    #     # ax.set_xscale('log')
    # else:
    #     ax.set_xlabel('number of agents', fontweight='bold')
    #     # scale? Usually linear for agents.
    #     # ax.set_xscale('linear') 

    # ax.set_ylim(bottom=0.0)

    # ax.set_ylabel('throughput', fontweight='bold')
    # ax.set_title('Throughput vs Computation Time', fontweight='bold', fontsize=14)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    for side in ('top', 'right', 'bottom', 'left'):
        ax.spines[side].set_linewidth(2.0)
    
    ax.grid(True, alpha=0.3)
    if x_axis_mode == 'agents' and agent_ticks:
        ax.set_xticks([float(a) for a in agent_ticks])
        ax.set_xticklabels([str(a) for a in agent_ticks])
    if y_axis_mode == 'runtime':
        ax.set_yscale('log', nonpositive='clip')
    if not hide_labels:
        ordered_handles = []
        ordered_labels = []
        from matplotlib.lines import Line2D
        for algo in algos_in_plot:
            style = ALGORITHM_STYLES.get(algo, {"color": "black", "linestyle": "-", "linewidth": 2})
            handle_kwargs = dict(
                linestyle=style.get('linestyle', '-'),
                color=style.get('color', 'black'),
                linewidth=style.get('linewidth', 2),
            )
            if x_axis_mode == 'agents':
                handle_kwargs.update(
                    marker='o',
                    markersize=10,
                    markerfacecolor=style.get('color', 'black'),
                    markeredgecolor='none',
                    markeredgewidth=0.0,
                )
            elif marker_by == 'method':
                mk = _marker_for_method(algo)
                handle_kwargs.update(
                    marker=mk,
                    markersize=_marker_size_pt(mk, default=10.0),
                    markerfacecolor=style.get('color', 'black'),
                    markeredgecolor='black',
                    markeredgewidth=0.3,
                )
            ordered_handles.append(Line2D([0], [0], **handle_kwargs))
            ordered_labels.append(ALGORITHMS.get(algo, algo))

        if (y_axis_mode == 'throughput') and (not hide_upper_bound) and any_upper_bounds:
            ordered_handles.append(
                Line2D(
                    [0], [0],
                    linestyle='--',
                    color='black',
                    linewidth=1.5,
                )
            )
            ordered_labels.append('Upper bound (lb_sp_task_count/makespan)')

        if ordered_handles:
            if legend_mode == 'inline':
                ax.legend(handles=ordered_handles, labels=ordered_labels, frameon=False, loc='best')
            else:
                _save_horizontal_legend(out_path, ordered_handles, ordered_labels)

    plt.tight_layout()

    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir and not os.path.exists(out_dir):
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError:
            pass

    try:
        plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='black')
        print(f'[OK] Saved plot to: {out_path}')
    except Exception as e:
        print(f'[ERR] Failed to save figure: {e}')

    if show:
        try:
            plt.show()
        except Exception:
            # Environments without display may fail .show(); ignore
            pass
    plt.close()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description='Plot scatter of comp_time vs total_completed_tasks/makespan for benchmark results.')
    parser.add_argument('--results', default=os.path.join('benchmark', 'results'), help='Results root directory (default: benchmark/results)')
    parser.add_argument('--out', default=os.path.join('benchmark', 'results', 'time_throughput.pdf'), help='Output image path (when multiple maps exist, map name is appended)')
    parser.add_argument('--show', action='store_true', help='Show the plot window')
    parser.add_argument('--hide-labels', action='store_true', help='Hide agent-count annotations and legend')
    parser.add_argument('--hide-upper-bound', action='store_true', help='Hide upper bound overlay (lb_sp_task_count/makespan)')
    parser.add_argument('--x-axis', choices=['runtime', 'agents'], default='runtime', help='X-axis variable (default: runtime). runtime means comp_time per step (sec/step).')
    parser.add_argument('--marker-by', choices=['agents', 'method'], default='agents', help='Marker assignment mode (default: agents). agents: markers by agent count; method: markers by algorithm.')
    parser.add_argument('--connect-same-color', action='store_true', help='Connect same-color algorithms per agent count with dashed lines (x-axis=runtime only).')
    parser.add_argument(
        '--split-by-agents',
        '--split-agents',
        action='store_true',
        dest='split_by_agents',
        help='Generate a separate figure for each agent count (in addition to per-map splitting).',
    )
    parser.add_argument('--legend-mode', '--legend', choices=['separate', 'inline'], default='separate', dest='legend_mode', help='Legend rendering mode: separate saves a *_legend.pdf; inline draws legend on the plot (default: separate)')
    parser.add_argument(
        '--y-axis',
        choices=['throughput', 'runtime', 'comp_time', 'makespan'],
        default='throughput',
        help='Y-axis variable (default: throughput). runtime/comp_time mean solver compute time per step (sec/step). makespan means makespan (RHCR: simulation_time).',
    )
    parser.add_argument('--annotate-rightmost-only', action='store_true', help='Only annotate the algorithm that appears furthest to the right')
    args = parser.parse_args(argv)

    results_root = args.results
    if not os.path.exists(results_root):
        print(f'[ERR] Results directory not found: {results_root}')
        return 2

    points = collect_results(results_root)
    if not points:
        print('[WARN] No points parsed. Check results directory or file formats.')
    
    # Group points by map. Prefer header `map_file` when it yields a
    # specific name; otherwise derive from the result filename pattern
    # `result_<algo>_<map>_<agents>_... .txt` so that RHCR also groups
    # correctly per map.
    def _sanitize(name: str) -> str:
        return ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in name)

    def _map_from_header(p: ResultPoint) -> Optional[str]:
        if not p.map_file:
            return None
        base = os.path.basename(p.map_file)
        name, _ext = os.path.splitext(base)
        # Ignore overly generic names like "map" used by some RHCR logs
        if not name or name.lower() in ('map', 'unknown'):
            return None
        return _sanitize(name)

    def _map_from_path(p: ResultPoint) -> Optional[str]:
        stem = os.path.splitext(os.path.basename(p.path))[0]
        toks = stem.split('_')
        # Expect: ["result", <algo>, <map>, <agents>, ...]
        if len(toks) >= 3 and toks[0].lower() == 'result':
            return _sanitize(toks[2])
        return None

    def _map_label(p: ResultPoint) -> str:
        return _map_from_header(p) or _map_from_path(p) or 'unknown'

    by_map: Dict[str, List[ResultPoint]] = {}
    for p in points:
        by_map.setdefault(_map_label(p), []).append(p)

    def _with_suffix(path: str, suffix: str) -> str:
        d = os.path.dirname(path)
        b = os.path.basename(path)
        stem, ext = os.path.splitext(b)
        if not stem:
            stem = 'scatter'
        if not ext:
            ext = '.pdf'
        return os.path.join(d, f"{stem}_{suffix}{ext}")

    out_dir = os.path.dirname(args.out)
    _stem, out_ext = os.path.splitext(os.path.basename(args.out))
    if not out_ext:
        out_ext = '.pdf'

    def _out_path(map_label: Optional[str], agents: Optional[int]) -> str:
        parts = [_sanitize(str(args.x_axis)), _sanitize(str(args.y_axis))]
        if map_label and map_label != 'unknown':
            parts.append(_sanitize(map_label))
        if agents is not None:
            parts.append(f"agents{int(agents)}")
        filename = "_".join(parts) + out_ext
        return os.path.join(out_dir, filename) if out_dir else filename

    if not args.split_by_agents:
        if len(by_map) <= 1:
            # Either all points share a map or map info is absent; single figure.
            only_lbl = next(iter(by_map.keys()), None)
            plot(points, out_path=_out_path(only_lbl, None), show=bool(args.show), hide_labels=bool(args.hide_labels), x_axis_mode=args.x_axis, y_axis_mode=args.y_axis, hide_upper_bound=bool(args.hide_upper_bound), annotate_rightmost_only=args.annotate_rightmost_only, legend_mode=args.legend_mode, marker_by=args.marker_by, connect_same_color=bool(args.connect_same_color))
        else:
            for lbl, plist in sorted(by_map.items()):
                plot(plist, out_path=_out_path(lbl, None), show=bool(args.show), hide_labels=bool(args.hide_labels), x_axis_mode=args.x_axis, y_axis_mode=args.y_axis, hide_upper_bound=bool(args.hide_upper_bound), annotate_rightmost_only=args.annotate_rightmost_only, legend_mode=args.legend_mode, marker_by=args.marker_by, connect_same_color=bool(args.connect_same_color))
        return 0

    skipped_missing_agents = 0
    for lbl, plist in sorted(by_map.items()):
        by_agents: Dict[int, List[ResultPoint]] = {}
        for p in plist:
            if p.agents is None:
                skipped_missing_agents += 1
                continue
            by_agents.setdefault(int(p.agents), []).append(p)
        for n, alist in sorted(by_agents.items(), key=lambda kv: kv[0]):
            plot(alist, out_path=_out_path(lbl, n), show=bool(args.show), hide_labels=bool(args.hide_labels), x_axis_mode=args.x_axis, y_axis_mode=args.y_axis, hide_upper_bound=bool(args.hide_upper_bound), annotate_rightmost_only=args.annotate_rightmost_only, legend_mode=args.legend_mode, marker_by=args.marker_by, connect_same_color=bool(args.connect_same_color))

    if skipped_missing_agents:
        print(f'[WARN] --split-by-agents: skipped {skipped_missing_agents} points without agents=')
    return 0

if __name__ == '__main__':
    sys.exit(main())
