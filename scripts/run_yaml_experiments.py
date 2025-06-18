#!/usr/bin/env python3
import yaml
import os
import subprocess
import datetime
import argparse
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.patches import Ellipse
import re

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
        
        # option_name = f"--{key.replace('_', '-')}" # スネークケースをケバブケースに変換することが多い
                                                 # もしYAMLキーとオプション名が完全に一致するなら f"--{key}"
        option_name = f"--{key}" # build/main が --lg_window のようなアンダースコア形式を期待するため元に戻す
        
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

def plot_results(results, output_dir, vary_property, plot_settings=None):
    """結果をグラフにプロットする"""
    df = pd.DataFrame(results)
    
    # 数値型に変換
    for col in ['sum_of_costs', 'soc', 'comp_time_ms', 'runtime', 'soc_lb']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 正規化したSOCを計算（SOC/SOC_LB）
    if 'soc' in df.columns and 'soc_lb' in df.columns:
        df['soc_normalized'] = np.where(
            (df['soc_lb'] > 0),
            df['soc'] / df['soc_lb'],
            np.nan  # soc_lbが0以下の場合はNaNを割り当て
        )
        df.dropna(subset=['soc_normalized'], inplace=True)
    
    # プロットモードを決定
    plot_mode = "scatter"  # デフォルト
    if plot_settings and 'mode' in plot_settings:
        plot_mode = plot_settings['mode']
    
    if plot_mode == "violin":
        plot_violin(df, output_dir, vary_property, plot_settings)
    else:
        plot_scatter(df, output_dir, vary_property)

def plot_violin(df, output_dir, vary_property, plot_settings):
    """バイオリンプロットを作成する"""
    violin_params = plot_settings.get('violin_params', {})
    y_axis_param = violin_params.get('y_axis', 'soc')
    orient = violin_params.get('orient', 'vertical')
    
    # vary_property がコラボキーかどうかを判定
    is_collab_plot = 'collab_config_str' in df.columns and df['collab_key'].iloc[0] == vary_property if 'collab_key' in df.columns and not df.empty else False
    
    # グループ化に使用する列を決定
    if is_collab_plot:
        grouping_column = 'collab_config_str'
        x_axis_param = grouping_column
    else:
        grouping_column = vary_property
        x_axis_param = vary_property
    
    # データの準備
    if grouping_column not in df.columns:
        print(f"エラー: グループ化列 '{grouping_column}' がDataFrameに存在しません。")
        return
    
    if y_axis_param not in df.columns:
        print(f"エラー: y軸パラメータ '{y_axis_param}' がDataFrameに存在しません。")
        return
    
    # データのフィルタリング
    valid_data = df.dropna(subset=[grouping_column, y_axis_param])
    
    if valid_data.empty:
        print(f"警告: 有効なデータがありません ({grouping_column}, {y_axis_param})。")
        return
    
    # グループの一意値を取得
    unique_group_values = valid_data[grouping_column].unique()
    
    if len(unique_group_values) < 2:
        print(f"警告: バイオリンプロットには少なくとも2つのグループが必要です。現在のグループ数: {len(unique_group_values)}")
        return
    
    # カラーパレットを設定
    num_colors = len(unique_group_values)
    if num_colors <= 9:
        palette = sns.color_palette("Set1", n_colors=num_colors)
        box_palette = sns.color_palette("Set2", n_colors=num_colors)
    elif num_colors <= 10:
        palette = sns.color_palette("tab10", n_colors=num_colors)
        box_palette = sns.color_palette("Paired", n_colors=num_colors)
    elif num_colors <= 20:
        palette = sns.color_palette("tab20", n_colors=num_colors)
        box_palette = sns.color_palette("tab20b", n_colors=num_colors)
    else: 
        palette = sns.color_palette("husl", n_colors=num_colors)
        box_palette = sns.color_palette("pastel", n_colors=num_colors)
    
    plt.figure(figsize=(12, 8))
    
    # バイオリンプロットを描画
    if orient == 'horizontal':
        ax = sns.violinplot(data=valid_data, x=y_axis_param, y=grouping_column, 
                           inner=None, palette=palette)
        # 箱ひげ図を重ね描き（外れ値を非表示）
        sns.boxplot(data=valid_data, x=y_axis_param, y=grouping_column, 
                   width=0.3, color='white', ax=ax, showfliers=False,
                   boxprops=dict(edgecolor='black'), whiskerprops=dict(color='black'),
                   capprops=dict(color='black'), medianprops=dict(color='black'))
        plt.xlabel(get_axis_label(y_axis_param))
        
        # コラボプロットの場合は適切なy軸ラベルを設定
        if is_collab_plot:
            plt.ylabel(f"{vary_property} Configuration")
        else:
            plt.ylabel(get_axis_label(vary_property))
    else:  # vertical
        ax = sns.violinplot(data=valid_data, x=grouping_column, y=y_axis_param, 
                           inner=None, palette=palette)
        # 箱ひげ図を重ね描き（外れ値を非表示）
        sns.boxplot(data=valid_data, x=grouping_column, y=y_axis_param, 
                   width=0.3, color='white', ax=ax, showfliers=False,
                   boxprops=dict(edgecolor='black'), whiskerprops=dict(color='black'),
                   capprops=dict(color='black'), medianprops=dict(color='black'))
        
        # コラボプロットの場合は適切なx軸ラベルを設定
        if is_collab_plot:
            plt.xlabel(f"{vary_property} Configuration")
        else:
            plt.xlabel(get_axis_label(vary_property))
        
        plt.ylabel(get_axis_label(y_axis_param))
        
        # x軸のラベルを回転（読みやすくするため）
        plt.xticks(rotation=45, ha='right')
    
    # タイトルを設定
    title_x_label = f"{vary_property} Configuration" if is_collab_plot else get_axis_label(vary_property)
    plt.title(f"Distribution of {get_axis_label(y_axis_param)} by {title_x_label}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # グラフを保存
    safe_vary_property = vary_property.replace('/', '_').replace('\\', '_')  # ファイル名に安全な文字を使用
    plot_path = os.path.join(output_dir, f"violin_{safe_vary_property}_{y_axis_param}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"バイオリンプロットを保存しました: {plot_path}")

def get_axis_label(param_name):
    """パラメータ名に対応する軸ラベルを返す"""
    label_map = {
        'soc': 'Sum of Costs (SoC)',
        'soc_normalized': 'Normalized SoC (SoC/SoC_LB)',
        'runtime': 'Runtime (ms)',
        'comp_time_ms': 'Computation Time (ms)',
        'use_sipp': 'Use SIPP',
        'lg': 'Local Guidance',
        'gg': 'Global Guidance',
        'lg_window': 'LG Window Size',
        'lg_collision_cost': 'LG Collision Cost',
        'lg_collision_cost_order': 'LG Collision Cost Order',
        'N': 'Number of Agents'
    }
    return label_map.get(param_name, param_name)

def plot_scatter(df, output_dir, vary_property):
    """従来の散布図プロットを作成する"""
    # シナリオとマップごとの散布図
    if vary_property: # vary_property は単独パラメータ名、またはコラボキー名
        plt.figure(figsize=(12, 8))
        
        # vary_property がコラボキーかどうかを判定 (dfに 'collab_key' 列があり、その値が vary_property と一致するか)
        # または、プロット対象の列が 'collab_config_str' かどうかで判定する方が堅牢かもしれない
        is_collab_plot = 'collab_config_str' in df.columns and df['collab_key'].iloc[0] == vary_property if 'collab_key' in df.columns and not df.empty else False

        grouping_column = 'collab_config_str' if is_collab_plot else vary_property
        
        if grouping_column not in df.columns:
            print(f"エラー: プロット用のグループ化列 '{grouping_column}' がDataFrameに存在しません。")
            return

        unique_group_values = df[grouping_column].unique()
        num_colors = len(unique_group_values)

        if num_colors == 0:
            print(f"警告: プロット対象のデータグループがありません ({grouping_column})。")
            return

        # 各プロパティ値/組み合わせに異なる色を割り当て
        if num_colors <= 9:
            palette = sns.color_palette("Set1", n_colors=num_colors)
        elif num_colors <= 10:
            palette = sns.color_palette("tab10", n_colors=num_colors)
        elif num_colors <= 20:
            palette = sns.color_palette("tab20", n_colors=num_colors)
        else: 
            palette = sns.color_palette("husl", n_colors=num_colors)
        
        color_map = dict(zip(unique_group_values, palette))
        
        # グラフを描画
        ax = plt.subplot(111)
        
        # 使用するデータ列を決定
        x_col = 'runtime' if 'runtime' in df.columns else 'comp_time_ms'
        y_col = 'soc_normalized' if 'soc_normalized' in df.columns else 'soc'
        
        for group_val in unique_group_values:
            subset = df[df[grouping_column] == group_val]
            if x_col not in subset or y_col not in subset or subset.empty:
                print(f"警告: グループ '{group_val}' のデータがありません")
                continue
            
            x = subset[x_col].values
            y = subset[y_col].values
            
            # 無効なデータをフィルタリング
            valid_indices = ~np.isnan(x) & ~np.isnan(y)
            x = x[valid_indices]
            y = y[valid_indices]
            
            if len(x) == 0 or len(y) == 0:
                print(f"警告: グループ '{group_val}' の有効なデータがありません")
                continue
            
            # 散布図のラベル
            if is_collab_plot:
                label_str = str(group_val) # collab_config_str をそのままラベルに
            else: # 単独パラメータの場合
                value_str = str(group_val) if not isinstance(group_val, bool) else ('True' if group_val else 'False')
                label_str = f"{vary_property}={value_str}"

            plt.scatter(x, y, label=label_str, 
                       color=color_map[group_val], alpha=0.7, s=50)
            
            # 十分なデータポイントがある場合、楕円と重心をプロット
            if len(x) >= 2:
                # 重心
                centroid_x = np.mean(x)
                centroid_y = np.mean(y)
                plt.scatter(centroid_x, centroid_y, marker='X', s=100, 
                           color=color_map[group_val], edgecolor='black', zorder=5) # value を group_val に変更
                
                # 楕円（信頼区間）
                if len(x) >= 3:  # 共分散行列を計算するには最低3点必要
                    # x, yの標準偏差が0に近い場合はスキップ
                    if np.isclose(np.std(x), 0) or np.isclose(np.std(y), 0):
                        print(f"警告: グループ '{group_val}' の標準偏差が0に近いため、楕円をスキップします")
                        continue
                    
                    cov = np.cov(x, y)
                    
                    # 共分散行列が無効な場合はスキップ
                    if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
                        print(f"警告: グループ '{group_val}' の共分散行列が無効です: {cov}")
                        continue
                    
                    try:
                        eigenvalues, eigenvectors = np.linalg.eig(cov)
                    except np.linalg.LinAlgError:
                        print(f"警告: グループ '{group_val}' の固有値分解に失敗しました")
                        continue
                    
                    # 固有値が負にならないようにする
                    eigenvalues = np.maximum(eigenvalues, 0)
                    
                    # 固有値と固有ベクトルを並べ替え
                    order = eigenvalues.argsort()[::-1]
                    eigenvalues = eigenvalues[order]
                    eigenvectors = eigenvectors[:,order]
                    
                    # 楕円の角度
                    angle = np.degrees(np.arctan2(*eigenvectors[:,0][::-1]))
                    
                    # 楕円の幅と高さ（2標準偏差）
                    n_std = 2
                    width, height = 2 * n_std * np.sqrt(eigenvalues)
                    
                    # 楕円を追加
                    ellipse = Ellipse(xy=(centroid_x, centroid_y),
                                     width=width, height=height,
                                     angle=angle,
                                     facecolor=color_map[group_val], alpha=0.2,  # value を group_val に変更
                                     edgecolor=color_map[group_val], linestyle='--') # value を group_val に変更
                    ax.add_patch(ellipse)
        
        # y_colに基づいてタイトルと軸ラベルを設定
        if y_col == 'soc_normalized':
            plt.title(f"Normalized SOC vs Computation Time (by {vary_property})") # 日本語を英語に変更
            plt.ylabel("Normalized SOC (SOC/SOC_LB)")
        else:
            plt.title(f"Total Cost (SoC) vs Computation Time (by {vary_property})") # 日本語を英語に変更
            plt.ylabel("Total Cost (sum_of_costs)") # 日本語を英語に変更
        
        plt.xlabel("Computation Time (ms)") # 日本語を英語に変更
        plt.grid(True, which="both", ls="--", alpha=0.3)
        
        # 凡例の整理
        handles, labels = ax.get_legend_handles_labels()
        unique_labels_dict = {}
        for handle, label in zip(handles, labels):
            if label not in unique_labels_dict:
                unique_labels_dict[label] = handle
        
        if unique_labels_dict:
            ax.legend(unique_labels_dict.values(), unique_labels_dict.keys(), title=f"Value of {vary_property}") # 日本語を英語に変更
        
        plt.tight_layout()
        
        # グラフを保存
        plot_path = os.path.join(output_dir, f"plot_{vary_property}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"グラフを保存しました: {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="YAMLテンプレートから実験を実行")
    parser.add_argument('--template', type=str, default="assets/template.yaml",
                        help="YAMLテンプレートファイルのパス")
    parser.add_argument('--output_dir', type=str, default=None,
                        help="結果を保存するディレクトリ（未指定の場合は日時を含むディレクトリが自動生成されます）")
    args = parser.parse_args()
    
    # YAMLテンプレートの読み込み
    properties = load_yaml_template(args.template)
    
    # プロット設定を抽出
    plot_settings = properties.pop('plot_settings', None)
    
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
            # もし single_varying_params にあったら、それは固定値として扱うべきかもしれない
            # (コラボで変動するが、単独では変動しない、しかし値はリストで定義されている場合)
            # ここでは、コラボで使われるなら単独変動からは除外する
            del single_varying_params[prop_to_remove]
            # fixed_params_from_yaml にも入れない（コラボ側で値が決定されるため）


    # 結果を格納する辞書
    all_results_for_plotting = {} # キーは single_param_name または collab_key

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
                    # else: 警告やエラー処理 (必要であれば)

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
            
            # (他のコラボ定義のパラメータも最初の値に固定するロジックは、通常コラボ定義は一つと仮定し省略)

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

    # --- 3. グラフプロット処理 ---
    for plot_key, results_list in all_results_for_plotting.items():
        if results_list:
            plot_output_dir = os.path.join(base_output_dir, plot_key) 
            os.makedirs(plot_output_dir, exist_ok=True)
            
            # plot_results に渡す vary_property は、単独パラメータの場合はその名前、
            # コラボの場合は、df内で組み合わせを識別する列名 (例: 'collab_config_str')
            # plot_results関数側で is_collab_plot フラグを使って判定する
            plot_results(results_list, plot_output_dir, plot_key, plot_settings)

if __name__ == "__main__":
    main()
