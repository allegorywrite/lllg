# LG-LaCAM Benchmark Evaluation

This directory contains the benchmark evaluation pipeline for LG-LaCAM (Local Guidance enhanced LaCAM) research.

## Overview

The benchmark compares Multi-Agent Path Finding (MAPF) algorithms configured via YAML:
- **LG-LaCAM (Proposed)**: Enhanced LaCAM with local guidance (`--lg --lg_window 8`)
- **LaCAM (Baseline)**: Standard LaCAM algorithm
- **Other algorithms**: Configurable via config.yaml

Evaluation is performed across configurable maps with agent counts [200, 400, 600, 800, 1000], measuring runtime vs Flow Time/Lower Bound ratio.

## Configuration

Maps, algorithms, and settings are configured via `config.yaml`. Default maps include:

1. empty-48-48
2. random-64-64-20
3. room-64-64-8
4. Paris_1_256
5. ost003d
6. warehouse-20-40-10-2-2

Additional maps (like brc202d, den312d, Berlin_1_256) can be enabled by uncommenting them in config.yaml.

## Complete Execution Steps

Follow these steps to run the complete benchmark evaluation:

```bash
cd ~/aist_ws/lg_lacam/
cmake -B build && make -C build -j4
python -m venv venv
source venv/bin/activate
pip install pandas matplotlib numpy pyyaml
cd benchmark/
source ../venv/bin/activate
python run_benchmark.py
```

This will:
- Run experiments for all maps configured in config.yaml
- Test with agent counts specified in config.yaml [200, 400, 600, 800, 1000]
- Execute 1 random scenario per configuration (configurable)
- Save results to `benchmark_results/benchmark_results.csv`

### Generate Plots

Create publication-quality individual map plots:

```bash
python plot_individual_maps.py benchmark_results/benchmark_results.csv
```

Plots are saved to `individual_plots/` directory with all 6 maps.

## Output Files

### Data Files
- `benchmark_results/benchmark_results.csv`: Raw experimental data
- `benchmark_results/benchmark_results.json`: Structured JSON format
- `individual_plots/summary_by_map.csv`: Aggregated statistics by map

### Plots
- `individual_plots/runtime_vs_flow_ratio_[map_name].png`: Individual map comparisons
- Publication-ready plots with log-scale runtime axis
- LG-LaCAM: Blue lines with star markers
- LaCAM: Green lines with triangle markers

## Algorithm Configurations

### LG-LaCAM (Proposed)
```bash
build/main -i [scenario] -m [map] -N [agents] -v 3 --lg --lg_window 8
```

### LaCAM (Baseline)
```bash
build/main -i [scenario] -m [map] -N [agents] -v 3
```

## Directory Structure

```
benchmark/
├── README.md                    # This file
├── config.yaml                 # YAML configuration file
├── run_benchmark.py            # Main benchmark execution script
├── plot_individual_maps.py     # Individual map plotting script
├── run_full_benchmark.sh       # Complete automation pipeline
├── benchmark_results/          # Experimental data
│   ├── benchmark_results.csv
│   ├── benchmark_results.json
│   └── summary_by_map.csv
└── individual_plots/           # Publication-quality plots
    ├── runtime_vs_flow_ratio_empty_48_48.png
    ├── runtime_vs_flow_ratio_random_64_64_20.png
    ├── runtime_vs_flow_ratio_room_64_64_8.png
    ├── runtime_vs_flow_ratio_Paris_1_256.png
    ├── runtime_vs_flow_ratio_ost003d.png
    └── runtime_vs_flow_ratio_warehouse_20_40_10_2_2.png
```

## Customization

### Adding New Maps

Edit `config.yaml` to add new maps:

```yaml
maps:
  - name: "your_map"
    path: "path/to/your_map.map"
    scenarios: "path/to/scenarios/directory"
```

### Adding New Algorithms

Edit `config.yaml` and add to the algorithms section:

```yaml
algorithms:
  your_algorithm:
    name: "Your Algorithm Name"
    executable: "../build/main"
    args_template: "-i {scenario} -m {map_file} -N {agents} --your --flags -o {result_file}"
    result_file_template: "../build/result_{algorithm}_{map_base}_{agents}_{scenario_base}.txt"
    available: true
```

### Modifying Agent Counts

Change the `agent_counts` list in `config.yaml`:

```yaml
agent_counts: [200, 400, 600, 800, 1000]  # Modify as needed
```

### Plot Styling

Modify `ALGORITHM_STYLES` in `plot_individual_maps.py` for custom colors and markers.

## Notes

- Uses only random scenarios (not even scenarios) to support full agent count range
- Default: 1 scenario per configuration for quick testing
- Use `--num-scenarios N` for multiple scenarios and statistical significance
- Runtime measurements include algorithm execution time only
- Flow Time/Lower Bound ratio measures solution quality vs optimal lower bound

## Command Line Options

```bash
python run_benchmark.py --help
```

Key options:
- `--config CONFIG`: Specify YAML configuration file (default: config.yaml)
- `--asset-dir ASSET_DIR`: Override assets directory from config
- `--algorithms lg_lacam lacam`: Run specific algorithms only
- `--maps brc202d empty-48-48`: Run specific maps only
- `--agents 200 400`: Run specific agent counts only
- `--build`: Build lg_lacam before running experiments

### Example Usage

```bash
# Run with default configuration
python run_benchmark.py

# Run specific map with specific algorithm
python run_benchmark.py --maps brc202d --algorithms lg_lacam --agents 200

# Use custom configuration file
python run_benchmark.py --config my_config.yaml

# Override assets directory
python run_benchmark.py --asset-dir /path/to/assets
```