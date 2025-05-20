# local guidance for MAPF

This repository is based on [lacam3](https://kei18.github.io/lacam3/).

## Building

All you need is [CMake](https://cmake.org/) (≥v3.16).
The code is written in C++(17).

First, clone this repo with submodules.

```sh
git clone --recursive {this repo}
```

Then, build the project.

```sh
cmake -B build && make -C build -j4
```

## Usage

```sh
build/main -i assets/random-32-32-10-random-1.scen -m assets/random-32-32-10.map -N 400 -v 3
```

The result will be saved in `build/result.txt`.

You can find details of all parameters with:

```sh
build/main --help
```

## Visualizer

This repository is compatible with [kei18@mapf-visualizer](https://github.com/kei18/mapf-visualizer).
For example,

```sh
mapf-visualizer assets/random-32-32-10.map build/result.txt
```

## Plotting Results

This repository includes Python scripts to parse and plot experiment results.

### Parsing Raw Results

The `parse_results.py` script processes raw output files (e.g., from `run_experiments.sh`) and compiles them into a summary CSV file (`experiment_summary.csv`).

```sh
python parse_results.py
```

### Plotting Summary Statistics

The `plot_results.py` script generates various summary plots (success rate, runtime, sum of costs) from `experiment_summary.csv`. These plots are saved in the `plots_summary/` or `plots_per_scenario_lines/` directory.

```sh
# For summary plots (boxplots, bar charts)
python plot_results.py --mode summary

# For plots with one line per scenario
python plot_results.py --mode per_scenario
```

### Plotting Agent-Specific Scatter Plots

The `plot_agent_results.py` script generates scatter plots of SOC (Sum of Costs) vs. Runtime for specific agent counts. It reads data from directories like `build/agent_100/`, `build/agent_200/`, etc. The plots are saved in the `plots_agent_specific/` directory.

To generate plots for agent counts 100, 200, 300, and 400:
```sh
python plot_agent_results.py build/agent_100 build/agent_200 build/agent_300 build/agent_400
```
You can specify one or more agent data directories. The output directory can be changed using the `--output_dir` argument.

## Notes

### install pre-commit for formatting

```sh
pre-commit install
```

### simple test

```sh
ctest --test-dir ./build
```
