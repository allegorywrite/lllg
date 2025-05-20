import os
import re
import csv

def parse_result_file(filepath):
    data = {}
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    data[key] = value
    except FileNotFoundError:
        print(f"Error: File not found {filepath}")
        return None
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None
    return data

def main():
    results_dir = "build"
    output_csv_file = "experiment_summary.csv"
    
    all_results = []
    
    # Regex to parse filename
    # result_even-1_N100_nogg.txt
    # result_random-1_N100_gg.txt
    filename_pattern = re.compile(r"result_(even|random)-(\d+)_N(\d+)_(gg|nogg)\.txt")

    for filename in os.listdir(results_dir):
        if filename.startswith("result_") and filename.endswith(".txt"):
            match = filename_pattern.match(filename)
            if not match:
                print(f"Skipping file with unexpected name format: {filename}")
                continue

            scenario_type, scenario_id_str, num_agents_str, gg_status_str = match.groups()
            scenario_id = int(scenario_id_str)
            num_agents = int(num_agents_str)
            gg_enabled = (gg_status_str == "gg")
            
            filepath = os.path.join(results_dir, filename)
            parsed_data = parse_result_file(filepath)
            
            if parsed_data:
                try:
                    solved = int(parsed_data.get("solved", -1))
                    soc = int(parsed_data.get("soc", -1))
                    makespan = int(parsed_data.get("makespan", -1))
                    comp_time = int(parsed_data.get("comp_time", -1)) # Assuming comp_time is integer
                    
                    all_results.append({
                        "scenario_type": scenario_type,
                        "scenario_id": scenario_id,
                        "num_agents": num_agents,
                        "gg_enabled": gg_enabled,
                        "solved": solved,
                        "soc": soc,
                        "makespan": makespan,
                        "comp_time": comp_time,
                        "filename": filename # For debugging if needed
                    })
                except ValueError as ve:
                    print(f"ValueError parsing data from {filename}: {ve}. Data: {parsed_data}")
                except Exception as e:
                    print(f"Unexpected error processing data from {filename}: {e}. Data: {parsed_data}")


    if not all_results:
        print("No result files found or parsed.")
        return

    # Sort results for consistent output
    all_results.sort(key=lambda x: (x["scenario_type"], x["scenario_id"], x["num_agents"], x["gg_enabled"]))
    
    header = ["scenario_type", "scenario_id", "num_agents", "gg_enabled", 
              "solved", "soc", "makespan", "comp_time", "filename"]
              
    with open(output_csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        writer.writerows(all_results)
        
    print(f"Successfully parsed {len(all_results)} result files.")
    print(f"Summary saved to {output_csv_file}")

if __name__ == "__main__":
    main()
