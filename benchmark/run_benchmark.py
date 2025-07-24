#!/usr/bin/env python3
"""
Multi-agent Path Finding (MAPF) Benchmark Evaluation Script

This script runs benchmarks for 4 different MAPF algorithms:
1. lg_lacam (proposed): Local guidance enhanced LaCAM
2. EECBS-f: Enhanced Explicit Coordination with Bounded Search 
3. LNS2: Large Neighborhood Search 2
4. LaCAM+: LaCAM with improvements

The script evaluates all algorithms on 6 different maps with agent counts N=[200, 400, 600, 800, 1000]
and generates runtime vs Flow time/Lower bound comparison plots.
"""

import os
import sys
import subprocess
import json
import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import yaml

def load_config(config_file: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    try:
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file {config_file} not found. Using default configuration.")
        return get_default_config()
    except yaml.YAMLError as e:
        print(f"Error parsing config file: {e}")
        return get_default_config()

def get_default_config() -> Dict:
    """Get default configuration if YAML file is not available."""
    return {
        "agent_counts": [200, 400, 600, 800, 1000],
        "maps": [
            {"name": "Paris_1_256", "path": "asset_for_benchmark/Paris_1_256.map", "scenarios": "asset_for_benchmark/scen-random"},
            {"name": "empty-48-48", "path": "asset_for_benchmark/empty-48-48.map", "scenarios": "asset_for_benchmark/scen-random"},
            {"name": "ost003d", "path": "asset_for_benchmark/ost003d.map", "scenarios": "asset_for_benchmark/scen-random"},
            {"name": "random-64-64-20", "path": "asset_for_benchmark/random-64-64-20.map", "scenarios": "asset_for_benchmark/scen-random"},
            {"name": "room-64-64-8", "path": "asset_for_benchmark/room-64-64-8.map", "scenarios": "asset_for_benchmark/scen-random"},
            {"name": "warehouse-20-40-10-2-2", "path": "asset_for_benchmark/warehouse-20-40-10-2-2.map", "scenarios": "asset_for_benchmark/scen-random"}
        ],
        "algorithms": {
            "lg_lacam": {
                "name": "LG-LaCAM (Proposed)",
                "executable": "../build/main",
                "args_template": "-i {scenario} -m {map_file} -N {agents} -v 3 -t 20 --lg --lg_collision_sort --lg_window 8 -o {result_file}",
                "result_file_template": "../build/result_{algorithm}_{map_base}_{agents}_{scenario_base}.txt",
                "available": True
            },
            "lacam": {
                "name": "LaCAM",
                "executable": "../build/main",
                "args_template": "-i {scenario} -m {map_file} -N {agents} -v 3 -t 20 -o {result_file}",
                "result_file_template": "../build/result_{algorithm}_{map_base}_{agents}_{scenario_base}.txt",
                "available": True
            }
        },
        "settings": {
            "timeout": 300,
            "num_scenarios": 1,
            "scenario_pattern": "{map_base}-random-*.scen",
            "assets_base_dir": "../assets"
        }
    }

class BenchmarkRunner:
    def __init__(self, config: Dict, asset_dir: str = None, output_dir: str = "benchmark_results"):
        self.config = config
        self.asset_dir = Path(asset_dir) if asset_dir else Path(config["settings"]["assets_base_dir"])
        self.output_dir = Path(output_dir)
        self.num_scenarios = config["settings"]["num_scenarios"]
        self.timeout = config["settings"]["timeout"]
        self.scenario_pattern = config["settings"]["scenario_pattern"]
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for each algorithm
        for alg_id in config["algorithms"].keys():
            (self.output_dir / alg_id).mkdir(exist_ok=True)
    
    def get_scenarios_for_map(self, map_config: Dict) -> List[str]:
        """Get all scenario files for a given map configuration."""
        map_name = map_config["name"]
        
        # Use map-specific scenarios dir if specified, otherwise use default
        if "scenarios" in map_config:
            scenarios_dir = map_config["scenarios"]
        else:
            scenarios_dir = self.config["settings"].get("default_scenarios_dir", "scen-random")
        
        scenarios = []
        
        # Build full path to scenarios directory
        scenario_dir_path = self.asset_dir / scenarios_dir
        if scenario_dir_path.exists():
            pattern = self.scenario_pattern.format(map_base=map_name)
            for scen_file in scenario_dir_path.glob(pattern):
                scenarios.append(str(scen_file))
        
        return sorted(scenarios)[:self.num_scenarios]  # Use specified number of scenarios per map
    
    def run_single_experiment(self, algorithm: str, map_config: Dict, scenario: str, 
                            agent_count: int, timespan: int = 20) -> Optional[Dict]:
        """Run a single experiment for given parameters."""
        alg_config = self.config["algorithms"][algorithm]
        
        if not alg_config["available"]:
            print(f"Skipping {algorithm} - not available yet")
            return None
        
        map_path = self.asset_dir / map_config["path"]
        map_name = map_config["name"]
        
        # Generate unique result file path
        map_base = map_name.replace('-', '_')
        scenario_base = Path(scenario).stem.replace('-', '_')
        
        if algorithm == "lns2":
            # LNS2 uses different output format
            result_file_base = alg_config["result_file_template"].format(
                algorithm=algorithm,
                map_base=map_base,
                agents=agent_count,
                scenario_base=scenario_base
            )
            result_file = result_file_base + ".txt"
        else:
            result_file = alg_config["result_file_template"].format(
                algorithm=algorithm,
                map_base=map_base,
                agents=agent_count,
                scenario_base=scenario_base
            )
        
        # Format command arguments
        if algorithm == "lns2":
            args = alg_config["args_template"].format(
                scenario=scenario,
                map_file=str(map_path),
                agents=agent_count,
                timespan=timespan,
                result_file_base=result_file_base
            )
        else:
            args = alg_config["args_template"].format(
                scenario=scenario,
                map_file=str(map_path),
                agents=agent_count,
                timespan=timespan,
                result_file=result_file
            )
        
        cmd = f"{alg_config['executable']} {args}"
        
        print(f"Running: {cmd}")
        
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd.split(),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(Path.cwd())
            )
            end_time = time.time()
            
            if result.returncode != 0:
                print(f"Error running {algorithm}: {result.stderr}")
                return None
            
            # Parse result file
            if algorithm == "lns2":
                # LNS2 creates CSV file with different naming
                lns2_csv_file = Path(result_file_base + "-LNS.csv")
                if lns2_csv_file.exists():
                    result_data = self.parse_result_file(lns2_csv_file, algorithm, map_name, 
                                                       scenario, agent_count, end_time - start_time, timespan)
                    # Clean up result file after parsing
                    try:
                        lns2_csv_file.unlink()
                    except:
                        pass
                    return result_data
                else:
                    print(f"LNS2 result file not found: {lns2_csv_file}")
                    return None
            elif algorithm == "guided_lacam2":
                # guided_lacam2 outputs to stdout, parse directly
                return self.parse_guided_lacam2_stdout(result.stdout, algorithm, map_name,
                                                     scenario, agent_count, end_time - start_time, timespan)
            else:
                result_file_path = Path(result_file)
                if result_file_path.exists():
                    result_data = self.parse_result_file(result_file_path, algorithm, map_name, 
                                                       scenario, agent_count, end_time - start_time, timespan)
                    # Clean up result file after parsing
                    try:
                        result_file_path.unlink()
                    except:
                        pass
                    return result_data
                else:
                    print(f"Result file not found: {result_file_path}")
                    return None
                
        except subprocess.TimeoutExpired:
            print(f"Timeout for {algorithm} on {map_name} with {agent_count} agents")
            return {
                "algorithm": algorithm,
                "map": map_name,
                "scenario": os.path.basename(scenario),
                "agents": agent_count,
                "timespan": timespan,
                "runtime": self.timeout,
                "success": False,
                "timeout": True,
                "flow_time": None,
                "lower_bound": None,
                "flow_time_ratio": None,
                "flow_time_init_ratio": None,
                "comp_time_init": None
            }
        except Exception as e:
            print(f"Exception running {algorithm}: {e}")
            return None
    
    def parse_result_file(self, result_file: Path, algorithm: str, map_name: str,
                         scenario: str, agent_count: int, runtime: float, timespan: int = 20) -> Dict:
        """Parse the result file to extract metrics."""
        
        try:
            with open(result_file, 'r') as f:
                content = f.read()
            
            result_data = {
                "algorithm": algorithm,
                "map": map_name,
                "scenario": os.path.basename(scenario),
                "agents": agent_count,
                "timespan": timespan,
                "runtime": runtime,
                "success": False,
                "timeout": False,
                "flow_time": None,
                "flow_time_init": None,
                "lower_bound": None,
                "flow_time_ratio": None,
                "flow_time_init_ratio": None,
                "makespan": None,
                "makespan_lb": None,
                "comp_time_init": None
            }
            
            # Parse lg_lacam output format
            if algorithm.startswith("lg_lacam"):
                return self.parse_lg_lacam_result(content, result_data)
            if algorithm == "gg_lacam":
                return self.parse_lg_lacam_result(content, result_data)
            if algorithm.startswith("lg&gg_lacam"):
                return self.parse_lg_lacam_result(content, result_data)
            if algorithm == "lns":
                return self.parse_lg_lacam_result(content, result_data)
            elif algorithm == "eecbs_f":
                return self.parse_eecbs_f_result(content, result_data)
            elif algorithm == "lns2":
                return self.parse_lns2_result(content, result_data)
            elif algorithm == "lacam":
                return self.parse_lacam_result(content, result_data)
            else:
                print(f"Unknown algorithm: {algorithm}")
                return result_data
            
        except Exception as e:
            print(f"Error parsing result file {result_file}: {e}")
            return None
    
    def parse_lg_lacam_result(self, content: str, result_data: Dict) -> Dict:
        """Parse lg_lacam specific output format."""
        lines = content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('solved='):
                result_data['success'] = line.split('=')[1] == '1'
            elif line.startswith('soc='):
                result_data['flow_time'] = int(line.split('=')[1])
            elif line.startswith('soc_lb='):
                result_data['lower_bound'] = int(line.split('=')[1])
            elif line.startswith('soc_init='):
                result_data['flow_time_init'] = int(line.split('=')[1])
            elif line.startswith('makespan='):
                result_data['makespan'] = int(line.split('=')[1])
            elif line.startswith('makespan_lb='):
                result_data['makespan_lb'] = int(line.split('=')[1])
            elif line.startswith('comp_time='):
                # Convert from milliseconds to seconds
                result_data['runtime'] = float(line.split('=')[1]) / 1000.0
            elif line.startswith('comp_time_init='):
                # Extract comp_time_init for LNS
                result_data['comp_time_init'] = float(line.split('=')[1])
        
        # Calculate flow time ratio
        if result_data['flow_time'] and result_data['lower_bound'] and result_data['lower_bound'] > 0:
            result_data['flow_time_ratio'] = result_data['flow_time'] / result_data['lower_bound']
        if result_data['flow_time_init'] and result_data['lower_bound'] and result_data['lower_bound'] > 0:
            result_data['flow_time_init_ratio'] = result_data['flow_time_init'] / result_data['lower_bound']
        
        return result_data
    
    def parse_eecbs_f_result(self, content: str, result_data: Dict) -> Dict:
        """Parse EECBS-f output format (placeholder)."""
        # TODO: Implement when EECBS-f is available
        # This is a placeholder for the actual implementation
        result_data['success'] = False
        return result_data
    
    def parse_lns2_result(self, content: str, result_data: Dict) -> Dict:
        """Parse LNS2 output format from CSV file."""
        try:
            lines = content.strip().split('\n')
            if len(lines) < 2:
                result_data['success'] = False
                return result_data
            
            # Parse CSV header and data
            header = lines[0].split(',')
            data = lines[1].split(',')
            
            if len(header) != len(data):
                result_data['success'] = False
                return result_data
            
            # Create mapping from header to data
            lns2_result = dict(zip(header, data))
            
            # Extract relevant metrics
            runtime = float(lns2_result.get('runtime', 0))
            solution_cost = int(lns2_result.get('solution cost', 0))
            lower_bound = int(lns2_result.get('lower bound', 0))
            
            result_data['runtime'] = runtime
            result_data['flow_time'] = solution_cost
            result_data['lower_bound'] = lower_bound
            result_data['success'] = solution_cost > 0
            
            # Calculate flow time ratio
            if solution_cost > 0 and lower_bound > 0:
                result_data['flow_time_ratio'] = solution_cost / lower_bound
            
            return result_data
            
        except Exception as e:
            print(f"Error parsing LNS2 result: {e}")
            result_data['success'] = False
            return result_data
    
    def parse_lacam_result(self, content: str, result_data: Dict) -> Dict:
        """Parse LaCAM (base) output format."""
        # LaCAM base uses the same output format as lg_lacam
        return self.parse_lg_lacam_result(content, result_data)
    
    def parse_guided_lacam2_stdout(self, stdout: str, algorithm: str, map_name: str,
                                 scenario: str, agent_count: int, runtime: float, timespan: int = 20) -> Dict:
        """Parse guided_lacam2 stdout output format."""
        result_data = {
            "algorithm": algorithm,
            "map": map_name,
            "scenario": os.path.basename(scenario),
            "agents": agent_count,
            "timespan": timespan,
            "runtime": runtime,
            "success": False,
            "timeout": False,
            "flow_time": None,
            "lower_bound": None,
            "flow_time_ratio": None,
            "comp_time_init": None
        }
        
        try:
            lines = stdout.strip().split('\n')
            last_line = lines[-1] if lines else ""
            
            # Look for the final result line like:
            # solved: 20259ms	makespan: 125 (lb=112, ub=1.12)	sum_of_costs: 11439 (lb=8371, ub=1.37)	sum_of_loss: 9786 (lb=8371, ub=1.17)
            if "solved:" in last_line and "sum_of_costs:" in last_line:
                result_data['success'] = True
                
                # Extract sum_of_costs (flow_time)
                import re
                soc_match = re.search(r'sum_of_costs:\s*(\d+)', last_line)
                if soc_match:
                    result_data['flow_time'] = int(soc_match.group(1))
                
                # Extract lower bound from sum_of_costs
                soc_lb_match = re.search(r'sum_of_costs:.*lb=(\d+)', last_line)
                if soc_lb_match:
                    result_data['lower_bound'] = int(soc_lb_match.group(1))
                
                # Calculate ratio
                if result_data['flow_time'] and result_data['lower_bound']:
                    result_data['flow_time_ratio'] = result_data['flow_time'] / result_data['lower_bound']
            
            return result_data
            
        except Exception as e:
            print(f"Error parsing guided_lacam2 stdout: {e}")
            result_data['success'] = False
            return result_data
    
    def run_all_experiments(self, algorithms: List[str] = None, 
                          maps: List[str] = None, 
                          agent_counts: List[int] = None):
        """Run all benchmark experiments."""
        
        if algorithms is None:
            algorithms = list(self.config["algorithms"].keys())
        if maps is None:
            maps = [m["name"] for m in self.config["maps"]]
        if agent_counts is None:
            agent_counts = self.config["agent_counts"]
        
        # Get timespan values from config
        timespans = self.config.get("timespan", [20])
        
        # Create map lookup
        map_configs = {m["name"]: m for m in self.config["maps"]}
        
        results = []
        total_experiments = len(algorithms) * len(maps) * len(agent_counts) * len(timespans) * self.num_scenarios
        print("algorithms:", len(algorithms), ", maps:", len(maps), ", agent_counts:", len(agent_counts), ", timespans:", len(timespans), ", scenarios:", self.num_scenarios)
        current_experiment = 0
        
        for algorithm in algorithms:
            if not self.config["algorithms"][algorithm]["available"]:
                print(f"Skipping {algorithm} - not available")
                continue
                
            for map_name in maps:
                if map_name not in map_configs:
                    print(f"Map configuration not found for {map_name}")
                    continue
                    
                map_config = map_configs[map_name]
                scenarios = self.get_scenarios_for_map(map_config)
                
                if not scenarios:
                    print(f"No scenarios found for map {map_name}")
                    continue
                
                for agent_count in agent_counts:
                    for timespan in timespans:
                        for scenario in scenarios:
                            current_experiment += 1
                            print(f"\nExperiment {current_experiment}/{total_experiments}")
                            print(f"Algorithm: {algorithm}, Map: {map_name}, "
                                  f"Agents: {agent_count}, Timespan: {timespan}, Scenario: {os.path.basename(scenario)}")
                            
                            result = self.run_single_experiment(
                                algorithm, map_config, scenario, agent_count, timespan
                            )
                            
                            if result:
                                results.append(result)
        
        # Save results
        self.save_results(results)
        return results
    
    def save_results(self, results: List[Dict]):
        """Save benchmark results to CSV file."""
        if not results:
            print("No results to save")
            return
        
        # Group results by timespan and save separate files for each
        timespan_results = {}
        for result in results:
            timespan = result.get("timespan", 20)
            if timespan not in timespan_results:
                timespan_results[timespan] = []
            timespan_results[timespan].append(result)
        
        # Save separate files for each timespan
        for timespan, timespan_data in timespan_results.items():
            output_file = self.output_dir / f"benchmark_results_timespan_{timespan}.csv"
            json_file = self.output_dir / f"benchmark_results_timespan_{timespan}.json"
            
            fieldnames = timespan_data[0].keys()
            
            with open(output_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(timespan_data)
            
            print(f"Results for timespan {timespan} saved to {output_file}")
            
            # Also save as JSON
            with open(json_file, 'w') as f:
                json.dump(timespan_data, f, indent=2)
            
            print(f"Results for timespan {timespan} also saved to {json_file}")
        
        # Also save combined results
        combined_output_file = self.output_dir / "benchmark_results_combined.csv"
        combined_json_file = self.output_dir / "benchmark_results_combined.json"
        
        fieldnames = results[0].keys()
        
        with open(combined_output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"Combined results saved to {combined_output_file}")
        
        # Also save as JSON
        with open(combined_json_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Combined results also saved to {combined_json_file}")

def main():
    parser = argparse.ArgumentParser(description="Run MAPF benchmark experiments")
    parser.add_argument("--config", default="config.yaml",
                       help="YAML configuration file (default: config.yaml)")
    parser.add_argument("--asset-dir", 
                       help="Directory containing maps and scenarios (overrides config)")
    parser.add_argument("--output-dir", default="benchmark_results",
                       help="Output directory for results")
    parser.add_argument("--algorithms", nargs="+", 
                       help="Algorithms to test (default: all available from config)")
    parser.add_argument("--maps", nargs="+",
                       help="Maps to test (default: all from config)")
    parser.add_argument("--agents", nargs="+", type=int,
                       help="Agent counts to test (default: all from config)")
    parser.add_argument("--build", action="store_true",
                       help="Build lg_lacam before running experiments")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Validate algorithm choices
    if args.algorithms:
        available_algorithms = list(config["algorithms"].keys())
        for alg in args.algorithms:
            if alg not in available_algorithms:
                print(f"Error: Algorithm '{alg}' not found in config. Available: {available_algorithms}")
                return 1
    
    # Validate map choices
    if args.maps:
        available_maps = [m["name"] for m in config["maps"]]
        for map_name in args.maps:
            if map_name not in available_maps:
                print(f"Error: Map '{map_name}' not found in config. Available: {available_maps}")
                return 1
    
    # Validate agent counts
    if args.agents:
        available_agents = config["agent_counts"]
        for agent_count in args.agents:
            if agent_count not in available_agents:
                print(f"Error: Agent count {agent_count} not in config. Available: {available_agents}")
                return 1
    
    # Build lg_lacam if requested
    if args.build:
        print("Building lg_lacam...")
        try:
            subprocess.run(["make", "-C", "../build", "-j4"], check=True)
            print("Build successful")
        except subprocess.CalledProcessError as e:
            print(f"Build failed: {e}")
            return 1
    
    # Check if lg_lacam executable exists
    if not os.path.exists("../build/main"):
        print("lg_lacam executable not found. Please build first:")
        print("  cmake -B build && make -C build -j4")
        print("Or use --build flag")
        return 1
    
    # Run benchmarks
    runner = BenchmarkRunner(config, args.asset_dir, args.output_dir)
    results = runner.run_all_experiments(
        algorithms=args.algorithms,
        maps=args.maps, 
        agent_counts=args.agents
    )
    
    print(f"\nCompleted {len(results)} successful experiments")
    return 0

if __name__ == "__main__":
    sys.exit(main())