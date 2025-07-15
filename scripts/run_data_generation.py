#!/usr/bin/env python3
import yaml
import os
import subprocess
import datetime
import argparse
import itertools
import json

def load_yaml_template(yaml_file):
    """YAMLテンプレートファイルを読み込む"""
    with open(yaml_file, 'r') as f:
        return yaml.safe_load(f)

def create_command(properties, output_file=None):
    """プロパティを実行コマンドの引数に変換する"""
    cmd = ["build/main"]
    processed_keys = set()

    # 特別な処理が必要なキーを先に処理
    if 'i' in properties:
        cmd.extend(["-i", properties['i']])
        processed_keys.add('i')
    if 'm' in properties:
        cmd.extend(["-m", properties['m']])
        processed_keys.add('m')
    if 'N' in properties:
        cmd.extend(["-N", str(properties['N'])])
        processed_keys.add('N')
    if 'v' in properties:
        cmd.extend(["-v", str(properties['v'])])
        processed_keys.add('v')

    # その他のプロパティを汎用的に処理
    for key, value in properties.items():
        if key in processed_keys or key == 'o': # 'o' は後で特別処理
            continue
        
        option_name = f"--{key}" # build/main が --lg_window のようなアンダースコア形式を期待するため
        
        if isinstance(value, bool):
            if value:
                cmd.append(option_name)
        elif value is not None: # None や空文字の場合はオプションを追加しないことも考慮できる
            cmd.extend([option_name, str(value)])
        processed_keys.add(key)
            
    # 出力ファイルが指定されている場合は-oオプションを追加
    if output_file:
        cmd.extend(["-o", output_file])
    
    return cmd

def run_experiment(cmd, output_dir, properties, property_name):
    """実験を実行し、結果を保存する"""
    # 実行コマンドを表示
    cmd_str = ' '.join(cmd)
    print(f"実行コマンド: {cmd_str}")
    
    # 結果ファイル名を生成
    result_filename = f"result_{property_name}.txt"
    result_path = os.path.join(output_dir, result_filename)
    
    # ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)
    
    # コマンドを実行
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # 結果をファイルに保存
        with open(result_path, 'w') as f:
            f.write(f"# 実行コマンド: {cmd_str}\n")
            f.write(f"# 実行時間: {datetime.datetime.now()}\n")
            f.write("# プロパティ:\n")
            for k, v in properties.items():
                f.write(f"# {k}: {v}\n")
            f.write("\n")
            f.write(result.stdout)
        
        print(f"結果を保存しました: {result_path}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"エラー: {e}")
        print(f"標準出力: {e.stdout}")
        print(f"標準エラー: {e.stderr}")
        return None

def parse_result(result_text):
    """実行結果からデータを抽出する"""
    import re
    data = {}
    for line in result_text.splitlines():
        # 正規表現を使って "solved comp_time_ms: X makespan: Y (lb=Z, ub=W) sum_of_costs: V (lb=A, ub=B)" のような行を解析
        if "solved" in line and "comp_time_ms:" in line and "sum_of_costs:" in line:
            # 計算時間を抽出
            comp_time_match = re.search(r"comp_time_ms: (\d+)", line)
            if comp_time_match:
                data["comp_time_ms"] = comp_time_match.group(1)
                # plot_agent_results.pyに合わせてruntimeキーも追加
                data["runtime"] = comp_time_match.group(1)
            
            # 総コスト（SOC）を抽出
            sum_of_costs_match = re.search(r"sum_of_costs: (\d+)", line)
            if sum_of_costs_match:
                data["sum_of_costs"] = sum_of_costs_match.group(1)
                # plot_agent_results.pyに合わせてsocキーも追加
                data["soc"] = sum_of_costs_match.group(1)
            
            # SOCの下界（LB）を抽出
            soc_lb_match = re.search(r"sum_of_costs: \d+ \(lb=(\d+)", line)
            if soc_lb_match:
                data["soc_lb"] = soc_lb_match.group(1)
            
            # makespanを抽出
            makespan_match = re.search(r"makespan: (\d+)", line)
            if makespan_match:
                data["makespan"] = makespan_match.group(1)
        
        # 通常のkey=value形式も処理
        elif '=' in line and not line.startswith('#'):
            key, value = line.strip().split('=', 1)
            data[key.strip()] = value.strip()
    
    return data

def save_experiment_data(all_results, baseline_results, output_dir):
    """実験データをJSONファイルに保存する"""
    data = {
        'timestamp': datetime.datetime.now().isoformat(),
        'main_results': all_results,
        'baseline_results': baseline_results
    }
    
    json_path = os.path.join(output_dir, 'experiment_data.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"実験データを保存しました: {json_path}")
    return json_path

def main():
    parser = argparse.ArgumentParser(description="YAMLテンプレートから実験を実行してデータを生成")
    parser.add_argument('--template', type=str, default="assets/template.yaml",
                        help="YAMLテンプレートファイルのパス")
    parser.add_argument('--output_dir', type=str, default=None,
                        help="結果を保存するディレクトリ（未指定の場合は日時を含むディレクトリが自動生成されます）")
    args = parser.parse_args()
    
    # YAMLテンプレートの読み込み
    properties = load_yaml_template(args.template)
    
    # プロット設定を抽出（データ生成では使用しないが、後の可視化のために保存）
    plot_settings = properties.pop('plot_settings', None)
    
    # ベースライン設定を抽出（baseline_で始まるすべてのキー）
    baseline_configs = {}
    keys_to_remove = []
    for key, value in properties.items():
        if isinstance(key, str) and key.startswith('baseline_'):
            baseline_configs[key] = value
            keys_to_remove.append(key)
    
    # baseline_で始まるキーを削除
    for key in keys_to_remove:
        properties.pop(key)
    
    # 実験結果を保存するディレクトリを作成
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = args.output_dir if args.output_dir else f"results/{timestamp}"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # シナリオとマップについては全要素実行する
    scenario_files = properties.get('i', [])
    map_files = properties.get('m', [])
    
    # パラメータを分類
    single_varying_params = {}
    collab_definitions = {}
    fixed_params_from_yaml = {} # YAMLにリストで定義されていても、varyingやcollabで使われないものは固定値扱い

    for key, values in properties.items():
        if isinstance(key, str) and key.startswith('&'):
            collab_key = key[1:] # '&' を除去
            collab_definitions[collab_key] = values 
        # i, m は特別なのでここでは処理しない (fixed_params_from_yaml にも入れない)
        elif key in ['i', 'm']:
            continue
        elif isinstance(key, str) and isinstance(values, list) and len(values) > 1:
            # この時点では single_varying_params に一旦入れる (後でコラボで使われるものは除外)
            single_varying_params[key] = values
        else:
            if key is not None: 
                 fixed_params_from_yaml[key] = values[0] if isinstance(values, list) else values
            else:
                 print("警告: YAMLファイルに None 型のキーが見つかりました。スキップします。")

    # コラボ定義で使われるパラメータを single_varying_params から削除
    collab_props_to_remove_from_single = set()
    for collab_def in collab_definitions.values():
        for prop_name in collab_def.get('properties', []):
            collab_props_to_remove_from_single.add(prop_name)
    
    for prop_to_remove in collab_props_to_remove_from_single:
        if prop_to_remove in single_varying_params:
            del single_varying_params[prop_to_remove]

    # 結果を格納する辞書
    all_results_for_plotting = {} # キーは single_param_name または collab_key
    baseline_results = {} # ベースライン結果を格納（キー: baseline名, 値: 結果リスト）

    # 1. 単独で変化するパラメータの組み合わせを生成 (コラボで使われるものは除外済み)
    single_param_names = list(single_varying_params.keys())
    single_param_value_lists = [single_varying_params[k] for k in single_param_names]
    
    single_param_combinations = list(itertools.product(*single_param_value_lists))
    if not single_param_combinations: # 単独で変化するパラメータがない場合
        single_param_combinations = [()] # 空のタプルでループを1回実行させる

    # 各単独パラメータのプロット用に結果を初期化
    for p_name in single_param_names:
        all_results_for_plotting[p_name] = []
    # 各コラボレーションプロット用に結果を初期化
    for c_key in collab_definitions.keys():
        all_results_for_plotting[c_key] = []

    # 各シナリオとマップの組み合わせに対して実行
    for i, (scenario, map_file) in enumerate(zip(scenario_files, map_files)):
        print(f"\n実験 {i+1}/{len(scenario_files)}: シナリオ={scenario}, マップ={map_file}")
        
        # --- 0. ベースライン実験の実行 ---
        for baseline_name, baseline_config in baseline_configs.items():
            print(f"{baseline_name}実験を実行中...")
            
            # Nがパラメータに含まれている場合、すべてのN値についてベースライン実験を実行
            n_values = properties.get('N', [100])  # デフォルトは100のみ
            if not isinstance(n_values, list):
                n_values = [n_values]
            
            for n_value in n_values:
                baseline_props = fixed_params_from_yaml.copy()
                baseline_props['i'] = scenario
                baseline_props['m'] = map_file
                baseline_props['N'] = n_value  # N値を明示的に設定
                baseline_props.update(baseline_config)
                
                # 他のパラメータの初期値を設定（Nは既に設定済み）
                for sp_key, sp_values_list in single_varying_params.items():
                    if sp_key not in baseline_props and sp_key != 'N':  # Nは既に設定済み
                        baseline_props[sp_key] = sp_values_list[0]
                
                for collab_def_key, collab_def_details in collab_definitions.items():
                    for collab_prop_name_in_def in collab_def_details['properties']:
                        if collab_prop_name_in_def not in baseline_props and collab_prop_name_in_def in properties and isinstance(properties[collab_prop_name_in_def], list):
                            if collab_prop_name_in_def == 'N':
                                baseline_props[collab_prop_name_in_def] = n_value  # N値を明示的に設定
                            else:
                                baseline_props[collab_prop_name_in_def] = properties[collab_prop_name_in_def][0]
                
                num_agents = baseline_props.get('N', 0)
                scenario_id_for_file = os.path.basename(scenario).split('.')[0]
                baseline_output_dir = os.path.join(base_output_dir, baseline_name)
                os.makedirs(baseline_output_dir, exist_ok=True)
                
                result_filename = f"result_{scenario_id_for_file}_N{num_agents}_{baseline_name}.txt"
                result_path = os.path.join(baseline_output_dir, result_filename)
                
                cmd = create_command(baseline_props, result_path)
                print(f"実行コマンド ({baseline_name}, N={n_value}): {' '.join(cmd)}")
                result_text = run_experiment(cmd, baseline_output_dir, baseline_props, f"{scenario_id_for_file}_{baseline_name}")
                
                if result_text:
                    result_data = parse_result(result_text)
                    result_data.update(baseline_props)
                    result_data['experiment_type'] = baseline_name
                    if baseline_name not in baseline_results:
                        baseline_results[baseline_name] = []
                    baseline_results[baseline_name].append(result_data)

        # --- 1. 単独パラメータのプロットのための実験実行とデータ収集 ---
        for p_name_to_plot in single_param_names:
            # p_name_to_plot 以外の単独パラメータと、コラボ定義内のパラメータは最初の値に固定
            base_props_for_single_plot = fixed_params_from_yaml.copy()
            base_props_for_single_plot['i'] = scenario
            base_props_for_single_plot['m'] = map_file

            for sp_key, sp_values_list in single_varying_params.items():
                if sp_key != p_name_to_plot: # プロット対象でない単独パラメータは最初の値に固定
                    base_props_for_single_plot[sp_key] = sp_values_list[0]
            
            for collab_def_key, collab_def_details in collab_definitions.items():
                for collab_prop_name_in_def in collab_def_details['properties']:
                    # コラボ定義内のパラメータも最初の値に固定 (ルート定義から取得)
                    if collab_prop_name_in_def in properties and isinstance(properties[collab_prop_name_in_def], list):
                         base_props_for_single_plot[collab_prop_name_in_def] = properties[collab_prop_name_in_def][0]

            # プロット対象のパラメータ p_name_to_plot の各値でループ
            for p_value in single_varying_params[p_name_to_plot]:
                current_properties = base_props_for_single_plot.copy()
                current_properties[p_name_to_plot] = p_value
                
                num_agents = current_properties.get('N', 0)
                exp_type = "vanilla"
                if current_properties.get('lg', False):
                    exp_type = "lg_dynamic" if current_properties.get('lg_dynamic_window', False) else "lg"
                scenario_id_for_file = os.path.basename(scenario).split('.')[0]

                output_dir_val_specific = os.path.join(base_output_dir, p_name_to_plot, str(p_value))
                os.makedirs(output_dir_val_specific, exist_ok=True)
                
                result_filename = f"result_{scenario_id_for_file}_N{num_agents}_{exp_type}.txt"
                result_path = os.path.join(output_dir_val_specific, result_filename)

                cmd = create_command(current_properties, result_path)
                print(f"実行コマンド (単独: {p_name_to_plot}={p_value}): {' '.join(cmd)}")
                result_text = run_experiment(cmd, output_dir_val_specific, current_properties, f"{scenario_id_for_file}")

                if result_text:
                    result_data = parse_result(result_text)
                    result_data.update(current_properties)
                    all_results_for_plotting[p_name_to_plot].append(result_data)

        # --- 2. コラボレーションプロットのための実験実行とデータ収集 ---
        for collab_key_to_plot, definition in collab_definitions.items():
            base_props_for_collab_plot = fixed_params_from_yaml.copy()
            base_props_for_collab_plot['i'] = scenario
            base_props_for_collab_plot['m'] = map_file

            # コラボプロット対象外の単独パラメータは最初の値に固定
            for sp_key, sp_values_list in single_varying_params.items():
                # コラボ定義に *含まれていない* 単独パラメータを固定
                if sp_key not in definition['properties']:
                     base_props_for_collab_plot[sp_key] = sp_values_list[0]

            collab_prop_names = definition['properties']
            collab_value_lists_for_product = []
            valid_collab_def = True
            for prop_n in collab_prop_names:
                if 'values' in definition and prop_n in definition['values']:
                    collab_value_lists_for_product.append(definition['values'][prop_n])
                elif prop_n in properties and isinstance(properties[prop_n], list):
                    collab_value_lists_for_product.append(properties[prop_n])
                else:
                    print(f"警告: コラボレーション '{collab_key_to_plot}' のパラメータ '{prop_n}' の値リストが見つかりません。このコラボプロットをスキップします。")
                    valid_collab_def = False
                    break
            if not valid_collab_def:
                continue

            collab_param_combinations = list(itertools.product(*collab_value_lists_for_product))

            for collab_combo_values in collab_param_combinations:
                current_properties = base_props_for_collab_plot.copy()
                collab_config_parts = []
                for idx, prop_n in enumerate(collab_prop_names):
                    current_properties[prop_n] = collab_combo_values[idx]
                    collab_config_parts.append(f"{prop_n}={collab_combo_values[idx]}")
                collab_config_str = "_".join(collab_config_parts)

                num_agents = current_properties.get('N', 0)
                exp_type = "vanilla"
                if current_properties.get('lg', False):
                    exp_type = "lg_dynamic" if current_properties.get('lg_dynamic_window', False) else "lg"
                scenario_id_for_file = os.path.basename(scenario).split('.')[0]
                
                collab_output_dir_specific = os.path.join(base_output_dir, collab_key_to_plot, collab_config_str)
                os.makedirs(collab_output_dir_specific, exist_ok=True)
                
                result_filename = f"result_{scenario_id_for_file}_N{num_agents}_{exp_type}_{collab_config_str}.txt"
                result_path = os.path.join(collab_output_dir_specific, result_filename)

                cmd = create_command(current_properties, result_path)
                print(f"実行コマンド (コラボ: {collab_key_to_plot} - {collab_config_str}): {' '.join(cmd)}")
                result_text = run_experiment(cmd, collab_output_dir_specific, current_properties, f"{scenario_id_for_file}_{collab_config_str}")

                if result_text:
                    result_data = parse_result(result_text)
                    result_data.update(current_properties)
                    result_data['collab_config_str'] = collab_config_str 
                    result_data['collab_key'] = collab_key_to_plot
                    all_results_for_plotting[collab_key_to_plot].append(result_data)

    # 実験データをJSONに保存
    save_experiment_data(all_results_for_plotting, baseline_results, base_output_dir)
    
    # プロット設定も保存（後で可視化スクリプトが使用）
    if plot_settings:
        plot_settings_path = os.path.join(base_output_dir, 'plot_settings.json')
        with open(plot_settings_path, 'w', encoding='utf-8') as f:
            json.dump(plot_settings, f, ensure_ascii=False, indent=2)
        print(f"プロット設定を保存しました: {plot_settings_path}")

    print(f"\nデータ生成が完了しました。結果は {base_output_dir} に保存されています。")
    print(f"可視化を行う場合は以下のコマンドを実行してください:")
    print(f"python scripts/visualize_results.py --data_dir {base_output_dir}")

if __name__ == "__main__":
    main()