#!/usr/bin/env python3
import json
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.patches import Ellipse
import re
from collections import defaultdict

# Set style for publication-quality plots
plt.rcParams.update({
    'font.size': 18,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'serif'],
    'axes.linewidth': 2.5,
    'axes.edgecolor': 'black',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.linewidth': 2.0,
    'grid.color': 'gray',
    'grid.alpha': 0.3,
    'figure.facecolor': 'white',
    'figure.edgecolor': 'black',
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 10,
    'ytick.major.size': 10,
    'xtick.minor.size': 6,
    'ytick.minor.size': 6,
    'xtick.major.pad': 10,
    'ytick.major.pad': 10,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.frameon': True,
    'legend.fancybox': False,
    'legend.edgecolor': 'black',
    'legend.facecolor': 'white',
    'legend.framealpha': 1.0,
    'legend.fontsize': 24,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

def load_experiment_data(data_dir):
    """実験データをJSONファイルから読み込む"""
    json_path = os.path.join(data_dir, 'experiment_data.json')
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"実験データファイルが見つかりません: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data['main_results'], data['baseline_results']

def load_plot_settings(data_dir):
    """プロット設定をJSONファイルから読み込む"""
    plot_settings_path = os.path.join(data_dir, 'plot_settings.json')
    if os.path.exists(plot_settings_path):
        with open(plot_settings_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def get_axis_label(param_name):
    """パラメータ名に対応する軸ラベルを返す"""
    label_map = {
        'soc': 'Sum of Costs (SoC)',
        'soc_normalized': 'Normalized SoC (SoC/SoC_LB)',
        'benchmark_normalized': 'Normalized SoC (SoC/Benchmark_SoC)',
        'runtime': 'Runtime (ms)',
        'comp_time_ms': 'Computation Time (ms)',
        'lg': 'Local Guidance',
        'gg': 'Global Guidance',
        'lg_window': 'LG Window Size',
        'lg_collision_cost': 'LG Collision Cost',
        'lg_collision_cost_order': 'LG Collision Cost Order',
        'lg_num_refine': 'LG Number of Refinements',
        'N': 'Number of Agents',
        'throughput_tasks': 'Throughput (Tasks/s)',
        'throughput_makespan': 'Throughput (Makespan/s)',
        'total_completed_tasks': 'Total Completed Tasks'
    }
    return label_map.get(param_name, param_name)

def add_mean_line_to_scatter(df, grouping_column, x_col, y_col, line_property, is_collab_plot, plot_settings=None, color_index=0, legend_order='asc'):
    """散布図に平均値を線で結ぶ機能を追加する"""
    import re
    from collections import defaultdict
    
    if is_collab_plot:
        # コラボレーションプロットの場合、line_propertyの値でグループ分けして線を引く
        line_groups = defaultdict(list)
        
        for group_val in df[grouping_column].unique():
            subset = df[df[grouping_column] == group_val]
            if subset.empty:
                continue
            
            # collab_config_strからline_propertyの値を抽出
            if isinstance(group_val, str) and f'{line_property}=' in group_val:
                try:
                    # 正規表現でline_propertyの値を抽出
                    # 正規表現でline_propertyの値を抽出
                    # 値にアンダースコアが含まれる場合（local_guideなど）や、後ろに別のパラメータが続く場合を考慮
                    match = re.search(f'{line_property}=(.+?)(?:_[a-zA-Z0-9_]+=|$)', group_val)
                    if match:
                        line_value = match.group(1)
                        try:
                            # 数値の場合
                            line_value_numeric = float(line_value)
                        except ValueError:
                            # ブール値の場合
                            if line_value.lower() in ['true', 'false']:
                                line_value_numeric = 1.0 if line_value.lower() == 'true' else 0.0
                            else:
                                # 文字列の場合 (例: local_guide)
                                line_value_numeric = line_value
                        
                        # x軸とy軸の平均値を計算
                        mean_x = subset[x_col].mean()
                        mean_y = subset[y_col].mean()
                        if not np.isnan(mean_x) and not np.isnan(mean_y):
                            # 他のプロパティの値を抽出（line_property以外の変動要素）
                            other_properties = {}
                            for param in ['lg_num_refine', 'lg_window', 'lg_collision_cost', 'lg_collision_sort', 'N', 'lg_pruning_rate']:
                                if param != line_property and f'{param}=' in group_val:
                                    param_match = re.search(f'{param}=([^_]+)', group_val)
                                    if param_match:
                                        try:
                                            other_properties[param] = float(param_match.group(1))
                                        except ValueError:
                                            # ブール値の場合
                                            param_value = param_match.group(1)
                                            if param_value.lower() in ['true', 'false']:
                                                other_properties[param] = param_value.lower() == 'true'
                                            else:
                                                other_properties[param] = param_value
                            
                            # line_propertyの値でグループ分け
                            line_groups[line_value_numeric].append((other_properties, mean_x, mean_y, group_val))
                except Exception as e:
                    print(f"Error parsing group_val: {group_val}, error: {e}")
                    continue
        
        print(f"Debug: line_property={line_property}")
        print(f"Debug: Found {len(line_groups)} line groups: {list(line_groups.keys())}")
        for l_val, items in line_groups.items():
            print(f"  Line val {l_val}: {len(items)} points")

        # 各line_propertyの値について、他のプロパティでソートして線を引く
        # color_indexに基づいてbase_colorsから色を選択
        base_colors = ['red', 'blue', 'green', 'orange', 'purple', 'hotpink', 'gray', 'olive', 'cyan']
        colors = []
        for j in range(len(base_colors)):
            colors.append(base_colors[(color_index + j) % len(base_colors)])
        
        color_idx = 0
        
        # 最後のトレンド線（最大のline_value）を特定
        max_line_value = max(line_groups.keys()) if line_groups else None
        
        # line_groups をlegend_orderに基づいてソート
        sorted_line_groups = sorted(line_groups.items(), key=lambda x: x[0], reverse=(legend_order == 'desc'))

        print("legend order:", legend_order)
        
        for line_value, group_data in sorted_line_groups:
            if len(group_data) < 2:
                continue  # 線を引くには最低2点必要
            
            # 他のプロパティの値でソート（最初に見つかったプロパティを使用）
            if group_data[0][0]:  # other_propertiesが空でない場合
                sort_key = list(group_data[0][0].keys())[0]
                group_data.sort(key=lambda x: x[0].get(sort_key, 0))
            
            # 平均値の点を線で結ぶ
            x_values = [item[1] for item in group_data]
            y_values = [item[2] for item in group_data]
            
            color = colors[color_idx % len(colors)]
            color_idx += 1
            
            # ラベル用にline_valueを適切にフォーマット（元データの型に基づく）
            # 元のデータでline_propertyがブール型かチェック
            original_values = df[df[grouping_column].notna()][grouping_column].iloc[0] if not df.empty else None
            is_bool_property = False
            
            if original_values and isinstance(original_values, str) and line_property in original_values:
                # collab_config_strの場合、元データからブール型かチェック
                sample_match = re.search(f'{line_property}=([^_]+)', str(original_values))
                if sample_match and sample_match.group(1).lower() in ['true', 'false']:
                    is_bool_property = True
            elif line_property in df.columns:
                # 単独パラメータの場合、DataFrameの列の型をチェック
                sample_values = df[line_property].dropna()
                if len(sample_values) > 0 and isinstance(sample_values.iloc[0], bool):
                    is_bool_property = True
            
            if is_bool_property:
                if line_value == 1.0:
                    label_value = 'true'
                elif line_value == 0.0:
                    label_value = 'false'
                else:
                    label_value = str(line_value)
            else:
                label_value = str(int(line_value)) if isinstance(line_value, float) and line_value.is_integer() else str(line_value)
            
            # カスタム凡例名を取得
            custom_legend = None
            if plot_settings and 'violin_params' in plot_settings:
                custom_legend = plot_settings['violin_params'].get('legend', None)

            # legendがnullの場合は数字のみ表示
            if custom_legend is None:
                label_text = str(label_value) + " inheritance"
            else:
                legend_property_name = custom_legend if custom_legend else line_property
                label_text = f'{legend_property_name}={label_value}'
            
            plt.plot(x_values, y_values, linestyle='-', color=color, alpha=0.8, linewidth=4, zorder=10, 
                    label=label_text)
            
            # plt.xticks(x_values)
            # plt.yticks(y_values)
            
            # 平均値の点をマークで表示
            plt.scatter(x_values, y_values, marker='o', s=200, color=color, edgecolor='black', zorder=11)
            
            # hide_scatterがtrueの場合、トレンドラインの上下限をfill_betweenで表示
            if plot_settings and plot_settings.get('hide_scatter', False):
                # 各データポイントでの上下限を計算
                y_min_values = []
                y_max_values = []
                
                for other_properties, mean_x, mean_y, group_val in group_data:
                    # 同じグループのデータを取得
                    subset = df[df[grouping_column] == group_val]
                    if not subset.empty and y_col in subset.columns:
                        y_data = subset[y_col].dropna()
                        if len(y_data) > 0:
                            y_min_values.append(y_data.min())
                            y_max_values.append(y_data.max())
                        else:
                            y_min_values.append(mean_y)
                            y_max_values.append(mean_y)
                    else:
                        y_min_values.append(mean_y)
                        y_max_values.append(mean_y)
                
                # fill_betweenで上下限を薄く表示
                if len(x_values) == len(y_min_values) == len(y_max_values):
                    plt.fill_between(x_values, y_min_values, y_max_values, 
                                   alpha=0.2, color=color, zorder=1)
            
            # 各データポイントにラベルを追加（trend以外のプロパティを表示）
            # hide_data_labelがtrueの場合はラベルを表示しない
            # hide_data_labelがfalseの場合は最後のトレンド線のみラベルを表示
            hide_data_label = plot_settings.get('hide_data_label', False) if plot_settings else False
            if not hide_data_label and line_value == max_line_value:
                for idx, item in enumerate(group_data):
                    other_properties, x_coord, y_coord, group_val = item
                    # line_property以外のプロパティを文字列として構築
                    label_parts = []
                    for prop, value in other_properties.items():
                        if prop != line_property:
                            if isinstance(value, float) and value.is_integer():
                                # 最後のデータ点のみプロパティ名を含める
                                # if idx == len(group_data) - 1:
                                if idx == 0:
                                    label_parts.append(f'Refinement={int(value)}')
                                else:
                                    label_parts.append(str(int(value)))
                            elif isinstance(value, bool) or str(value).lower() in ['true', 'false']:
                                # ブール値の場合
                                bool_str = 'T' if str(value).lower() == 'true' else 'F'
                                # if idx == len(group_data) - 1:
                                if idx == 0:
                                    label_parts.append(f'{prop}={bool_str}')
                                else:
                                    label_parts.append(bool_str)
                            else:
                                # 最後のデータ点のみプロパティ名を含める
                                # if idx == len(group_data) - 1:
                                if idx == 0:
                                    label_parts.append(f'{prop}={value}')
                                else:
                                    label_parts.append(str(value))
                    
                    if label_parts:
                        label_text = ', '.join(label_parts)
                        plt.annotate(label_text, (x_coord, y_coord), 
                                   xytext=(10, 10), textcoords='offset points',
                                   fontsize=28, ha='left', va='bottom', 
                                   zorder=15)
        
    else:
        # 単独パラメータの場合、line_propertyでグループ分けして線を引く
        line_groups = defaultdict(list)
        
        for group_val in df[grouping_column].unique():
            subset = df[df[grouping_column] == group_val]
            if subset.empty:
                continue
            
            # line_propertyが同じグループ内に存在するかチェック
            if line_property in subset.columns:
                line_values = subset[line_property].unique()
                if len(line_values) == 1:  # 同じline_property値を持つグループ
                    try:
                        # 数値の場合
                        line_value_numeric = float(line_values[0])
                    except ValueError:
                        # ブール値の場合
                        if isinstance(line_values[0], bool) or str(line_values[0]).lower() in ['true', 'false']:
                            line_value_numeric = 1.0 if str(line_values[0]).lower() == 'true' else 0.0
                        else:
                            continue
                    
                    mean_x = subset[x_col].mean()
                    mean_y = subset[y_col].mean()
                    if not np.isnan(mean_x) and not np.isnan(mean_y):
                        line_groups[line_value_numeric].append((group_val, mean_x, mean_y))
        
        # 各line_propertyの値について線を引く
        # color_indexに基づいてbase_colorsから色を選択
        base_colors = ['red', 'blue', 'green', 'orange', 'purple', 'pink', 'gray', 'olive', 'cyan']
        colors = []
        for j in range(len(base_colors)):
            colors.append(base_colors[(color_index + j) % len(base_colors)])
        
        color_idx = 0
        
        # 最後のトレンド線（最大のline_value）を特定
        max_line_value = max(line_groups.keys()) if line_groups else None
        
        # line_groups をlegend_orderに基づいてソート
        sorted_line_groups_2 = sorted(line_groups.items(), key=lambda x: x[0], reverse=(legend_order == 'desc'))
        
        for line_value, group_data in sorted_line_groups_2:
            if len(group_data) < 2:
                continue  # 線を引くには最低2点必要
            
            # group_valでソート
            group_data.sort(key=lambda x: x[0])
            
            # 平均値の点を線で結ぶ
            x_values = [item[1] for item in group_data]
            y_values = [item[2] for item in group_data]
            
            color = colors[color_idx % len(colors)]
            color_idx += 1
            
            # ラベル用にline_valueを適切にフォーマット（元データの型に基づく）
            # 元のデータでline_propertyがブール型かチェック
            original_values = df[df[grouping_column].notna()][grouping_column].iloc[0] if not df.empty else None
            is_bool_property = False
            
            if original_values and isinstance(original_values, str) and line_property in original_values:
                # collab_config_strの場合、元データからブール型かチェック
                sample_match = re.search(f'{line_property}=([^_]+)', str(original_values))
                if sample_match and sample_match.group(1).lower() in ['true', 'false']:
                    is_bool_property = True
            elif line_property in df.columns:
                # 単独パラメータの場合、DataFrameの列の型をチェック
                sample_values = df[line_property].dropna()
                if len(sample_values) > 0 and isinstance(sample_values.iloc[0], bool):
                    is_bool_property = True
            
            if is_bool_property:
                if line_value == 1.0:
                    label_value = 'true'
                elif line_value == 0.0:
                    label_value = 'false'
                else:
                    label_value = str(line_value)
            else:
                label_value = str(int(line_value)) if isinstance(line_value, float) and line_value.is_integer() else str(line_value)
            
            # カスタム凡例名を取得
            custom_legend = None
            if plot_settings and 'violin_params' in plot_settings:
                custom_legend = plot_settings['violin_params'].get('legend', None)
            
            # legendがnullの場合は数字のみ表示
            if custom_legend is None:
                label_text = str(label_value)
            else:
                legend_property_name = custom_legend if custom_legend else line_property
                label_text = f'{legend_property_name}={label_value}'
            
            plt.plot(x_values, y_values, linestyle='--', color=color, alpha=0.8, linewidth=2, zorder=10, 
                    label=label_text)
            
            # 平均値の点をマークで表示
            plt.scatter(x_values, y_values, marker='o', s=80, color=color, edgecolor='black', zorder=11)
            
            # hide_scatterがtrueの場合、トレンドラインの上下限をfill_betweenで表示
            if plot_settings and plot_settings.get('hide_scatter', False):
                # 各データポイントでの上下限を計算
                y_min_values = []
                y_max_values = []
                
                for group_val, mean_x, mean_y in group_data:
                    # 同じグループのデータを取得
                    subset = df[df[grouping_column] == group_val]
                    if not subset.empty and y_col in subset.columns:
                        y_data = subset[y_col].dropna()
                        if len(y_data) > 0:
                            y_min_values.append(y_data.min())
                            y_max_values.append(y_data.max())
                        else:
                            y_min_values.append(mean_y)
                            y_max_values.append(mean_y)
                    else:
                        y_min_values.append(mean_y)
                        y_max_values.append(mean_y)
                
                # fill_betweenで上下限を薄く表示
                if len(x_values) == len(y_min_values) == len(y_max_values):
                    plt.fill_between(x_values, y_min_values, y_max_values, 
                                   alpha=0.2, color=color, zorder=1)
            
            # 各データポイントにラベルを追加（trend以外のプロパティを表示）
            # hide_data_labelがtrueの場合はラベルを表示しない
            # hide_data_labelがfalseの場合は最後のトレンド線のみラベルを表示
            hide_data_label = plot_settings.get('hide_data_label', False) if plot_settings else False
            if not hide_data_label and line_value == max_line_value:
                for idx, item in enumerate(group_data):
                    group_val, x_coord, y_coord = item
                    # グループ値をラベルとして表示
                    label_text = str(group_val)
                    plt.annotate(label_text, (x_coord, y_coord), 
                               xytext=(10, 10), textcoords='offset points',
                               fontsize=16, ha='center', va='bottom', 
                               zorder=15)
    
    # 凡例を更新
    plt.legend(loc='best')

def plot_violin(df, output_dir, vary_property, plot_settings, baseline_data=None):
    """バイオリンプロットを作成する"""
    if plot_settings is None:
        plot_settings = {}
    violin_params = plot_settings.get('violin_params', {})
    y_axis_param = violin_params.get('y_axis', 'soc')
    orient = violin_params.get('orient', 'vertical')
    
    # y_colの決定ロジック
    y_col = 'sum_of_costs' # デフォルト
    if y_axis_param in df.columns:
        y_col = y_axis_param
    elif y_axis_param == 'benchmark_normalized':
        y_col = 'benchmark_normalized'
    elif y_axis_param == 'soc_normalized':
        y_col = 'soc_normalized'
    elif y_axis_param == 'runtime':
        y_col = 'runtime' if 'runtime' in df.columns else 'comp_time_ms'
    elif y_axis_param == 'comp_time_ms':
        y_col = 'comp_time_ms'
    elif y_axis_param == 'runtime_sec':
        y_col = 'runtime_sec'
    else:
        y_col = 'soc_normalized' if 'soc_normalized' in df.columns else 'soc'
    
    # vary_property がコラボキーかどうかを判定
    is_collab_plot = 'collab_config_str' in df.columns and df['collab_key'].iloc[0] == vary_property if 'collab_key' in df.columns and not df.empty else False
    
    # グループ化に使用する列を決定
    if is_collab_plot:
        grouping_column = 'collab_config_str'
    else:
        grouping_column = vary_property
    
    # ベースラインデータをDataFrameに統合
    if baseline_data and len(baseline_data) > 0:
        all_baseline_dfs = []
        
        # baseline_dataが辞書の場合（複数のベースライン）
        if isinstance(baseline_data, dict):
            for baseline_name, baseline_list in baseline_data.items():
                if baseline_list and len(baseline_list) > 0:
                    baseline_df = pd.DataFrame(baseline_list)
                    # 数値型に変換
                    for col in ['sum_of_costs', 'soc', 'comp_time_ms', 'runtime', 'soc_lb', 'throughput_tasks', 'throughput_makespan', 'total_completed_tasks']:
                        if col in baseline_df.columns:
                            baseline_df[col] = pd.to_numeric(baseline_df[col], errors='coerce')
                    
                    # 正規化したSOCを計算
                    if 'soc' in baseline_df.columns and 'soc_lb' in baseline_df.columns:
                        baseline_df['soc_normalized'] = np.where(
                            (baseline_df['soc_lb'] > 0),
                            baseline_df['soc'] / baseline_df['soc_lb'],
                            np.nan
                        )
                    
                    # ベンチマーク正規化計算用にベンチマークSOCを記録
                    if 'soc' in baseline_df.columns:
                        baseline_df['benchmark_soc'] = baseline_df['soc']
                        # ベンチマーク正規化: 自分自身で正規化なので常に1
                        baseline_df['benchmark_normalized'] = 1.0
                    
                    # ベースラインデータにグループ化列を追加
                    if is_collab_plot:
                        baseline_df['collab_config_str'] = baseline_name
                        baseline_df['collab_key'] = vary_property
                    else:
                        baseline_df[vary_property] = baseline_name
                    
                    all_baseline_dfs.append(baseline_df)
        else:
            # 旧形式（リスト）の場合
            baseline_df = pd.DataFrame(baseline_data)
            # 数値型に変換
            for col in ['sum_of_costs', 'soc', 'comp_time_ms', 'runtime', 'soc_lb', 'throughput_tasks', 'throughput_makespan', 'total_completed_tasks']:
                if col in baseline_df.columns:
                    baseline_df[col] = pd.to_numeric(baseline_df[col], errors='coerce')
            
            # 正規化したSOCを計算
            if 'soc' in baseline_df.columns and 'soc_lb' in baseline_df.columns:
                baseline_df['soc_normalized'] = np.where(
                    (baseline_df['soc_lb'] > 0),
                    baseline_df['soc'] / baseline_df['soc_lb'],
                    np.nan
                )
            
            # ベンチマーク正規化計算用にベンチマークSOCを記録
            if 'soc' in baseline_df.columns:
                baseline_df['benchmark_soc'] = baseline_df['soc']
                # ベンチマーク正規化: 自分自身で正規化なので常に1
                baseline_df['benchmark_normalized'] = 1.0
            
            # ベースラインデータにグループ化列を追加
            if is_collab_plot:
                baseline_df['collab_config_str'] = 'Baseline'
                baseline_df['collab_key'] = vary_property
            else:
                baseline_df[vary_property] = 'Baseline'
            
            all_baseline_dfs.append(baseline_df)
        
        # ベースラインデータをメインデータに統合
        if all_baseline_dfs:
            all_baseline_df = pd.concat(all_baseline_dfs, ignore_index=True)
            df = pd.concat([df, all_baseline_df], ignore_index=True)
    
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
    
    if len(unique_group_values) < 1:
        print(f"警告: 有効なグループがありません。")
        return
    
    # カラーパレットを設定（ベースラインがある場合は赤色を予約）
    num_colors = len(unique_group_values)
    baseline_color = None
    
    # ベースライン用の色を予約
    baseline_values = [val for val in unique_group_values if isinstance(val, str) and val.startswith('baseline_')]
    other_values = [val for val in unique_group_values if not (isinstance(val, str) and val.startswith('baseline_'))]
    
    # ベースラインがある場合の色設定
    if baseline_values:
        # ベースライン用の色（赤系統）
        baseline_colors = ['red', 'darkred', 'crimson', 'firebrick', 'indianred']
        
        # 他の値用の色
        num_other_colors = len(other_values)
        if num_other_colors <= 8:
            other_palette = sns.color_palette("Set1", n_colors=num_other_colors)
        elif num_other_colors <= 9:
            other_palette = sns.color_palette("tab10", n_colors=num_other_colors)
        elif num_other_colors <= 19:
            other_palette = sns.color_palette("tab20", n_colors=num_other_colors)
        else: 
            other_palette = sns.color_palette("husl", n_colors=num_other_colors)
        
        # パレットを作成
        palette = []
        baseline_idx = 0
        other_idx = 0
        for val in unique_group_values:
            if isinstance(val, str) and val.startswith('baseline_'):
                if baseline_idx < len(baseline_colors):
                    palette.append(baseline_colors[baseline_idx])
                else:
                    palette.append('red')  # デフォルトで赤
                baseline_idx += 1
            else:
                palette.append(other_palette[other_idx])
                other_idx += 1
    else:
        # ベースラインがない場合は通常の色設定
        if num_colors <= 9:
            palette = sns.color_palette("Set1", n_colors=num_colors)
        elif num_colors <= 10:
            palette = sns.color_palette("tab10", n_colors=num_colors)
        elif num_colors <= 20:
            palette = sns.color_palette("tab20", n_colors=num_colors)
        else: 
            palette = sns.color_palette("husl", n_colors=num_colors)
    
    plt.figure(figsize=(12, 8))
    
    # バイオリンプロットを描画
    if orient == 'horizontal':
        ax = sns.violinplot(data=valid_data, x=y_col, y=grouping_column, 
                           inner=None, palette=palette)
        # 箱ひげ図を重ね描き（外れ値を非表示）
        sns.boxplot(data=valid_data, x=y_col, y=grouping_column, 
                   width=0.3, color='white', ax=ax, showfliers=False,
                   boxprops=dict(edgecolor='black'), whiskerprops=dict(color='black'),
                   capprops=dict(color='black'), medianprops=dict(color='black'))
        
        plt.xlabel(get_axis_label(y_axis_param), fontweight='bold', fontsize=22)
        
        # コラボプロットの場合は適切なy軸ラベルを設定
        if is_collab_plot:
            plt.ylabel(f"{vary_property} Configuration", fontweight='bold', fontsize=22)
        else:
            plt.ylabel(get_axis_label(vary_property), fontweight='bold', fontsize=22)
    else:  # vertical
        ax = sns.violinplot(data=valid_data, x=grouping_column, y=y_col, 
                           inner=None, palette=palette)
        # 箱ひげ図を重ね描き（外れ値を非表示）
        sns.boxplot(data=valid_data, x=grouping_column, y=y_col, 
                   width=0.3, color='white', ax=ax, showfliers=False,
                   boxprops=dict(edgecolor='black'), whiskerprops=dict(color='black'),
                   capprops=dict(color='black'), medianprops=dict(color='black'))
        
        # コラボプロットの場合は適切なx軸ラベルを設定
        if is_collab_plot:
            plt.xlabel(f"{vary_property} Configuration", fontweight='bold', fontsize=22)
        else:
            plt.xlabel(get_axis_label(vary_property), fontweight='bold', fontsize=22)
        
        plt.ylabel(get_axis_label(y_axis_param), fontweight='bold', fontsize=22)
        
        # x軸のラベルを回転（読みやすくするため）
        plt.xticks(rotation=45, ha='right')
    
    
    # タイトルを設定
    title_x_label = f"{vary_property} Configuration" if is_collab_plot else get_axis_label(vary_property)
    plt.title(f"Distribution of {get_axis_label(y_axis_param)} by {title_x_label}",
             fontweight='bold', fontsize=24, y=1.05)
    
    # 軸の境界線を太くする
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(3.0)
        spine.set_edgecolor('black')
    plt.tight_layout(pad=3.0)
    
    # グラフを保存
    safe_vary_property = vary_property.replace('/', '_').replace('\\', '_')  # ファイル名に安全な文字を使用
    plot_path = os.path.join(output_dir, f"violin_{safe_vary_property}_{y_axis_param}.pdf")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', pad_inches=0.5, facecolor='white', edgecolor='black')
    plt.close()
    print(f"バイオリンプロットを保存しました: {plot_path}")

def _plot_single_scatter(df, output_dir, vary_property, baseline_data=None, plot_settings=None, lg_window_suffix=None, color_index=0):
    """単一の散布図を作成するヘルパー関数"""
    plt.figure(figsize=(8, 8))
    
    # vary_property がコラボキーかどうかを判定
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
    # lg_window_suffixが指定されている場合は、color_indexに基づいて色パレットを選択
    if lg_window_suffix is not None:
        # 明確に区別できる色パレットを定義
        base_colors = [
            'red', 'blue', 'green', 'orange', 'purple', 
            'pink', 'gray', 'olive', 'cyan'
        ]
        
        # color_indexに基づいてベース色を選択し、その色系統のパレットを作成
        base_color = base_colors[color_index % len(base_colors)]
        
        # ベース色から色のバリエーションを作成
        if base_color == 'red':
            palette = sns.color_palette("Reds_r", n_colors=max(3, num_colors))[:num_colors]
        elif base_color == 'blue':
            palette = sns.color_palette("Blues_r", n_colors=max(3, num_colors))[:num_colors]
        elif base_color == 'green':
            palette = sns.color_palette("Greens_r", n_colors=max(3, num_colors))[:num_colors]
        elif base_color == 'orange':
            palette = sns.color_palette("Oranges_r", n_colors=max(3, num_colors))[:num_colors]
        elif base_color == 'purple':
            palette = sns.color_palette("Purples_r", n_colors=max(3, num_colors))[:num_colors]
        else:
            # その他の色の場合はHSVで色相を調整
            hue = 0.1 * color_index  # 色相を段階的に変更
            palette = sns.color_palette("husl", n_colors=num_colors, h=hue)
        
    else:
        # 通常の色パレット
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
    # plot_settingsからx_axis、y_axis、ラベル、タイトル、x_typeを取得
    x_axis_param = None
    y_axis_param = None
    custom_x_label = None
    custom_y_label = None
    custom_title = None
    x_type = None
    if plot_settings and 'violin_params' in plot_settings:
        x_axis_param = plot_settings['violin_params'].get('x_axis', None)
        y_axis_param = plot_settings['violin_params'].get('y_axis', None)
        custom_x_label = plot_settings['violin_params'].get('x_label', None)
        custom_y_label = plot_settings['violin_params'].get('y_label', None)
        custom_title = plot_settings['violin_params'].get('title', None)
        x_type = plot_settings['violin_params'].get('x_type', None)
    
    # x軸の列を決定
    if x_axis_param and x_axis_param in df.columns:
        x_col = x_axis_param
    elif x_axis_param == 'runtime_sec':
        x_col = 'runtime_sec'
    else:
        x_col = 'runtime' if 'runtime' in df.columns else 'comp_time_ms'
    
    if y_axis_param and y_axis_param in df.columns:
        y_col = y_axis_param
    elif y_axis_param == 'benchmark_normalized':
        y_col = 'benchmark_normalized'
    elif y_axis_param == 'soc_normalized':
        y_col = 'soc_normalized'
    elif y_axis_param == 'runtime':
        y_col = 'runtime' if 'runtime' in df.columns else 'comp_time_ms'
    elif y_axis_param == 'comp_time_ms':
        y_col = 'comp_time_ms'
    elif y_axis_param == 'runtime_sec':
        y_col = 'runtime_sec'
    else:
        y_col = 'soc_normalized' if 'soc_normalized' in df.columns else 'soc'
    
    # ベースラインデータの処理
    baseline_data_processed = {}
    if baseline_data and len(baseline_data) > 0:
        # baseline_dataが辞書の場合（複数のベースライン）
        if isinstance(baseline_data, dict):
            for baseline_name, baseline_list in baseline_data.items():
                if baseline_list and len(baseline_list) > 0:
                    baseline_df = pd.DataFrame(baseline_list)
                    # 数値型に変換
                    for col in ['sum_of_costs', 'soc', 'comp_time_ms', 'runtime', 'soc_lb', 'throughput_tasks', 'throughput_makespan', 'total_completed_tasks']:
                        if col in baseline_df.columns:
                            baseline_df[col] = pd.to_numeric(baseline_df[col], errors='coerce')
                    
                    # 正規化したSOCを計算
                    if 'soc' in baseline_df.columns and 'soc_lb' in baseline_df.columns:
                        baseline_df['soc_normalized'] = np.where(
                            (baseline_df['soc_lb'] > 0),
                            baseline_df['soc'] / baseline_df['soc_lb'],
                            np.nan
                        )
                    
                    # ベンチマーク正規化計算用にベンチマークSOCを記録
                    if 'soc' in baseline_df.columns:
                        baseline_df['benchmark_soc'] = baseline_df['soc']
                        # ベンチマーク正規化: 自分自身で正規化なので常に1
                        baseline_df['benchmark_normalized'] = 1.0
                    
                    baseline_x_values = []
                    baseline_y_values = []
                    if x_col in baseline_df.columns:
                        baseline_x_values = baseline_df[x_col].dropna().tolist()
                    if y_col in baseline_df.columns:
                        baseline_y_values = baseline_df[y_col].dropna().tolist()
                    
                    baseline_data_processed[baseline_name] = {
                        'x_values': baseline_x_values,
                        'y_values': baseline_y_values
                    }
        else:
            # 旧形式（リスト）の場合
            baseline_df = pd.DataFrame(baseline_data)
            # 数値型に変換
            for col in ['sum_of_costs', 'soc', 'comp_time_ms', 'runtime', 'soc_lb', 'throughput_tasks', 'throughput_makespan', 'total_completed_tasks']:
                if col in baseline_df.columns:
                    baseline_df[col] = pd.to_numeric(baseline_df[col], errors='coerce')
            
            # 正規化したSOCを計算
            if 'soc' in baseline_df.columns and 'soc_lb' in baseline_df.columns:
                baseline_df['soc_normalized'] = np.where(
                    (baseline_df['soc_lb'] > 0),
                    baseline_df['soc'] / baseline_df['soc_lb'],
                    np.nan
                )
            
            # ベンチマーク正規化計算用にベンチマークSOCを記録
            if 'soc' in baseline_df.columns:
                baseline_df['benchmark_soc'] = baseline_df['soc']
                # ベンチマーク正規化: 自分自身で正規化なので常に1
                baseline_df['benchmark_normalized'] = 1.0
            
            baseline_x_values = []
            baseline_y_values = []
            if x_col in baseline_df.columns:
                baseline_x_values = baseline_df[x_col].dropna().tolist()
            if y_col in baseline_df.columns:
                baseline_y_values = baseline_df[y_col].dropna().tolist()
            
            baseline_data_processed['Baseline'] = {
                'x_values': baseline_x_values,
                'y_values': baseline_y_values
            }
    
    # データ収集フェーズ：重心座標を保存
    centroids = []  # [(group_val, centroid_x, centroid_y, color)]
    
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
        
        # hide_scatter設定をチェック
        hide_scatter = plot_settings.get('hide_scatter', False) if plot_settings else False
        
        # 散布図のラベル
        if is_collab_plot:
            label_str = str(group_val) # collab_config_str をそのままラベルに
        else: # 単独パラメータの場合
            value_str = str(group_val) if not isinstance(group_val, bool) else ('True' if group_val else 'False')
            # カスタム凡例名を取得
            custom_legend = None
            if plot_settings and 'violin_params' in plot_settings:
                custom_legend = plot_settings['violin_params'].get('legend', None)
            
            # legendがnullの場合は数字のみ表示
            if custom_legend is None:
                label_str = value_str
            else:
                legend_property_name = custom_legend if custom_legend else vary_property
                label_str = f"{legend_property_name}={value_str}"

        # hide_scatterがtrueでない場合のみ散布図を描画
        if not hide_scatter:
            plt.scatter(x, y, label=label_str, 
                       color=color_map[group_val], alpha=0.7, s=50)
        
        # 十分なデータポイントがある場合、楕円と重心をプロット
        if len(x) >= 2:
            # 重心
            centroid_x = np.mean(x)
            centroid_y = np.mean(y)
            if not hide_scatter:
                plt.scatter(centroid_x, centroid_y, marker='X', s=100, 
                           color=color_map[group_val], edgecolor='black', zorder=5)
            
            # 重心座標を保存（後で線で接続するため）
            centroids.append((group_val, centroid_x, centroid_y, color_map[group_val]))
            
            # 楕円（信頼区間） - hide_scatterがtrueの場合は描画しない
            if not hide_scatter and len(x) >= 3:  # 共分散行列を計算するには最低3点必要
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
                                 facecolor=color_map[group_val], alpha=0.2,
                                 edgecolor=color_map[group_val], linestyle='--')
                ax.add_patch(ellipse)
    
    # ベースラインデータの処理と描画
    all_group_values = list(unique_group_values) + list(baseline_data_processed.keys())
    total_colors = len(all_group_values)
    
    if total_colors <= 9:
        extended_palette = sns.color_palette("Set1", n_colors=total_colors)
    elif total_colors <= 10:
        extended_palette = sns.color_palette("tab10", n_colors=total_colors)
    elif total_colors <= 20:
        extended_palette = sns.color_palette("tab20", n_colors=total_colors)
    else: 
        extended_palette = sns.color_palette("husl", n_colors=total_colors)
    
    # ベースライン用のカラーマップを作成
    baseline_color_map = {}
    baseline_start_idx = len(unique_group_values)
    for i, baseline_name in enumerate(baseline_data_processed.keys()):
        baseline_color_map[baseline_name] = extended_palette[baseline_start_idx + i]
    
    for baseline_name, baseline_info in baseline_data_processed.items():
        baseline_x_values = baseline_info['x_values']
        baseline_y_values = baseline_info['y_values']
        
        if baseline_x_values and baseline_y_values:
            color = baseline_color_map[baseline_name]
            
            # hide_scatter設定をチェック
            hide_scatter = plot_settings.get('hide_scatter', False) if plot_settings else False
            
            # hide_scatterがtrueでない場合のみベースライン散布図を描画
            if not hide_scatter:
                plt.scatter(baseline_x_values, baseline_y_values, 
                          label=baseline_name, color=color, alpha=0.7, s=50)
            
            # 十分なデータポイントがある場合、楕円と重心をプロット
            if len(baseline_x_values) >= 2:
                # 重心
                centroid_x = np.mean(baseline_x_values)
                centroid_y = np.mean(baseline_y_values)
                if not hide_scatter:
                    plt.scatter(centroid_x, centroid_y, marker='X', s=100, 
                               color=color, edgecolor='black', zorder=5)
                
                # ベースラインも重心リストに追加
                centroids.append((baseline_name, centroid_x, centroid_y, color))
                
                # 楕円（信頼区間） - hide_scatterがtrueの場合は描画しない
                if not hide_scatter and len(baseline_x_values) >= 3:
                    x_vals = np.array(baseline_x_values)
                    y_vals = np.array(baseline_y_values)
                    
                    # x, yの標準偏差が0に近い場合はスキップ
                    if not (np.isclose(np.std(x_vals), 0) or np.isclose(np.std(y_vals), 0)):
                        cov = np.cov(x_vals, y_vals)
                        
                        # 共分散行列が有効な場合のみ楕円を描画
                        if not (np.any(np.isnan(cov)) or np.any(np.isinf(cov))):
                            try:
                                eigenvalues, eigenvectors = np.linalg.eig(cov)
                                eigenvalues = np.maximum(eigenvalues, 0)
                                order = eigenvalues.argsort()[::-1]
                                eigenvalues = eigenvalues[order]
                                eigenvectors = eigenvectors[:,order]
                                
                                angle = np.degrees(np.arctan2(*eigenvectors[:,0][::-1]))
                                n_std = 2
                                width, height = 2 * n_std * np.sqrt(eigenvalues)
                                
                                ellipse = Ellipse(xy=(centroid_x, centroid_y),
                                                 width=width, height=height,
                                                 angle=angle,
                                                 facecolor=color, alpha=0.2,
                                                 edgecolor=color, linestyle='--')
                                ax.add_patch(ellipse)
                            except np.linalg.LinAlgError:
                                pass  # 楕円描画をスキップ
    
    # x軸とy軸のラベルを決定（カスタムラベルがある場合はそれを使用）
    x_label = custom_x_label if custom_x_label else get_axis_label(x_col)
    y_label = custom_y_label if custom_y_label else get_axis_label(y_col)
    
    # y軸ラベルを設定（カスタムラベルまたはデフォルトラベル）
    if custom_y_label:
        plt.ylabel(y_label, fontweight='bold', fontsize=30)
    else:
        # デフォルトのy軸ラベル生成
        if y_col == 'soc_normalized':
            plt.ylabel("Normalized SOC (SOC/SOC_LB)", fontweight='bold', fontsize=30)
        elif y_col == 'benchmark_normalized':
            plt.ylabel("Normalized SOC (SOC/Benchmark_SOC)", fontweight='bold', fontsize=30)
        elif y_col == 'runtime':
            plt.ylabel("Runtime (ms)", fontweight='bold', fontsize=30)
        elif y_col == 'comp_time_ms':
            plt.ylabel("Computation Time (ms)", fontweight='bold', fontsize=30)
        elif y_col == 'runtime_sec':
            plt.ylabel("Runtime (sec)", fontweight='bold', fontsize=30)
        elif y_col == 'throughput_tasks':
            plt.ylabel("Throughput (Tasks/s)", fontweight='bold', fontsize=30)
        elif y_col == 'throughput_makespan':
            plt.ylabel("Throughput (Makespan/s)", fontweight='bold', fontsize=30)
        elif y_col == 'total_completed_tasks':
            plt.ylabel("Total Completed Tasks", fontweight='bold', fontsize=30)
        else:
            plt.ylabel("Total Cost (sum_of_costs)", fontweight='bold', fontsize=30)
    
            plt.ylabel("Total Cost (sum_of_costs)", fontweight='bold', fontsize=30)
    
    if x_col == 'runtime_sec':
         plt.xlabel("Runtime (sec)", fontweight='bold', fontsize=30)
    else:
         plt.xlabel(x_label, fontweight='bold', fontsize=30)
    
    # 軸の境界線を太くする
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(3.0)
        spine.set_edgecolor('black')
    
    # 実際のデータから軸の範囲を計算してマージンを追加
    if not df.empty:
        # hide_scatter=trueの場合は重心点の範囲を使用、それ以外は全データ点を使用
        hide_scatter = plot_settings.get('hide_scatter', False) if plot_settings else False
        
        if hide_scatter and centroids and False:
            # 重心点の座標を取得
            centroid_x_coords = [centroid[1] for centroid in centroids]
            centroid_y_coords = [centroid[2] for centroid in centroids]
            
            if len(centroid_x_coords) > 0 and len(centroid_y_coords) > 0:
                x_min, x_max = min(centroid_x_coords), max(centroid_x_coords)
                y_min, y_max = min(centroid_y_coords), max(centroid_y_coords)
            else:
                # 重心データがない場合は全データを使用
                x_data = df[x_col].dropna()
                y_data = df[y_col].dropna()
                x_min, x_max = x_data.min(), x_data.max()
                y_min, y_max = y_data.min(), y_data.max()
        else:
            # 全データ点を使用
            x_data = df[x_col].dropna()
            y_data = df[y_col].dropna()
            
            if len(x_data) > 0 and len(y_data) > 0:
                x_min, x_max = x_data.min(), x_data.max()
                y_min, y_max = y_data.min(), y_data.max()
            else:
                return  # データがない場合は処理を終了
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        # マージンを追加
        x_margin = x_range * 0.05 if x_range > 0 else 1
        y_margin = y_range * 0.05 if y_range > 0 else 0.1
        
        # ax.set_xlim(x_min - x_margin, x_max + x_margin)
        # ax.set_xlim(x_min - x_margin, x_max + x_margin)
        # ax.set_ylim(0.6,  1.5)
        # ax.set_ylim(0, y_max + y_margin)
        # ax.set_ylim(0, 2500)
        # ax.set_xlim(x_min, x_max)
        # ax.set_ylim(y_min, y_max)

    # x_type が int の場合、x軸の目盛りを整数に設定
    if x_type and x_type.lower() == 'int':
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 凡例の整理
    handles, labels = ax.get_legend_handles_labels()
    unique_labels_dict = {}
    for handle, label in zip(handles, labels):
        if label and label not in unique_labels_dict:  # 空のラベルを除外
            unique_labels_dict[label] = handle
    
    if unique_labels_dict:
        ax.legend(unique_labels_dict.values(), unique_labels_dict.keys(), 
                 title=f"Value of {vary_property}",
                 loc='best', frameon=True, edgecolor='black', fancybox=False)
    
    # line設定がある場合、指定されたプロパティで平均値を線で結ぶ
    if plot_settings:
        line_property = plot_settings.get('line', None)
        if line_property:
            # legend_order パラメータを plot_settings から取得
            legend_order = plot_settings.get('legend_order', 'asc') if plot_settings else 'asc'
            add_mean_line_to_scatter(df, grouping_column, x_col, y_col, line_property, is_collab_plot, plot_settings, color_index, legend_order)
    
    plt.tight_layout(pad=3.0)
    
    # グラフを保存
    if lg_window_suffix is not None:
        plot_path = os.path.join(output_dir, f"plot_{vary_property}_lg_window_{lg_window_suffix}.pdf")
    else:
        plot_path = os.path.join(output_dir, f"plot_{vary_property}.pdf")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', pad_inches=0.05, facecolor='white', edgecolor='black')
    # plt.savefig("img.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"グラフを保存しました: {plot_path}")

def plot_scatter(df, output_dir, vary_property, baseline_data=None, plot_settings=None):
    """従来の散布図プロットを作成する"""
    # シナリオとマップごとの散布図
    if vary_property: # vary_property は単独パラメータ名、またはコラボキー名
        # div_graph設定をチェック
        div_graph = plot_settings.get('div_graph', False) if plot_settings else False
        line_property = plot_settings.get('line', None) if plot_settings else None
        
        # div_graph=trueかつline_property="lg_window"の場合、lg_windowごとに別グラフを作成
        if div_graph and line_property == "lg_window":
            # vary_property がコラボキーかどうかを判定
            is_collab_plot = 'collab_config_str' in df.columns and df['collab_key'].iloc[0] == vary_property if 'collab_key' in df.columns and not df.empty else False
            grouping_column = 'collab_config_str' if is_collab_plot else vary_property
            
            # lg_windowの値ごとにデータを分離
            lg_window_groups = {}
            
            if is_collab_plot:
                # コラボプロットの場合、collab_config_strからlg_windowの値を抽出
                for _, row in df.iterrows():
                    config_str = row.get('collab_config_str', '')
                    if isinstance(config_str, str) and 'lg_window=' in config_str:
                        import re
                        match = re.search(r'lg_window=(\d+)', config_str)
                        if match:
                            lg_window_val = int(match.group(1))
                            if lg_window_val not in lg_window_groups:
                                lg_window_groups[lg_window_val] = []
                            lg_window_groups[lg_window_val].append(row)
            else:
                # 単独パラメータの場合
                if 'lg_window' in df.columns:
                    for lg_window_val in df['lg_window'].unique():
                        if not pd.isna(lg_window_val):
                            subset = df[df['lg_window'] == lg_window_val]
                            lg_window_groups[lg_window_val] = subset.to_dict('records')
            
            # 各lg_window値について別々のグラフを作成
            sorted_lg_window_vals = sorted(lg_window_groups.keys())
            for i, lg_window_val in enumerate(sorted_lg_window_vals):
                group_data = lg_window_groups[lg_window_val]
                if not group_data:
                    continue
                
                # グループデータをDataFrameに変換
                group_df = pd.DataFrame(group_data)
                
                # 単一のグラフを作成（インデックスiを渡す）
                _plot_single_scatter(group_df, output_dir, vary_property, baseline_data, plot_settings, lg_window_val, i)
            
            return  # div_graphの処理が完了したので関数を終了
        
        # 通常の処理（div_graph=falseまたはline_property!="lg_window"）
        _plot_single_scatter(df, output_dir, vary_property, baseline_data, plot_settings)

def plot_results(results, output_dir, vary_property, plot_settings=None, baseline_data=None):
    """結果をグラフにプロットする"""
    df = pd.DataFrame(results)
    
    # 数値型に変換
    for col in ['sum_of_costs', 'soc', 'comp_time_ms', 'runtime', 'soc_lb', 'throughput_tasks', 'throughput_makespan', 'total_completed_tasks']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # runtime_sec計算 (ms -> s)
    if 'runtime' in df.columns:
         df['runtime_sec'] = df['runtime'] / 1000.0
    elif 'comp_time_ms' in df.columns:
         df['runtime_sec'] = df['comp_time_ms'] / 1000.0
    
    # 正規化したSOCを計算（SOC/SOC_LB）
    if 'soc' in df.columns and 'soc_lb' in df.columns:
        df['soc_normalized'] = np.where(
            (df['soc_lb'] > 0),
            df['soc'] / df['soc_lb'],
            np.nan  # soc_lbが0以下の場合はNaNを割り当て
        )
        df.dropna(subset=['soc_normalized'], inplace=True)
    
    # ベンチマーク正規化したSOCを計算（SOC/Benchmark_SOC）
    # 各N値について別々のベンチマークSOCを使用
    if 'soc' in df.columns and 'N' in df.columns and baseline_data:
        # N値別のベンチマークSOCを計算
        benchmark_soc_by_n = {}
        
        if isinstance(baseline_data, dict):
            # 複数のベースラインがある場合、最初のものを使用
            for baseline_name, baseline_list in baseline_data.items():
                if baseline_list and len(baseline_list) > 0:
                    baseline_df = pd.DataFrame(baseline_list)
                    if 'soc' in baseline_df.columns and 'N' in baseline_df.columns:
                        # 数値変換を確実に行う
                        baseline_df['soc'] = pd.to_numeric(baseline_df['soc'], errors='coerce')
                        baseline_df['N'] = pd.to_numeric(baseline_df['N'], errors='coerce')
                        
                        # N値でグループ化してベンチマークSOCを計算
                        for n_value in baseline_df['N'].unique():
                            n_subset = baseline_df[baseline_df['N'] == n_value]
                            if not n_subset.empty:
                                benchmark_soc_by_n[n_value] = n_subset['soc'].mean()
                        break
        else:
            # 単一ベースラインの場合
            if baseline_data and len(baseline_data) > 0:
                baseline_df = pd.DataFrame(baseline_data)
                if 'soc' in baseline_df.columns and 'N' in baseline_df.columns:
                    # 数値変換を確実に行う
                    baseline_df['soc'] = pd.to_numeric(baseline_df['soc'], errors='coerce')
                    baseline_df['N'] = pd.to_numeric(baseline_df['N'], errors='coerce')
                    
                    # N値でグループ化してベンチマークSOCを計算
                    for n_value in baseline_df['N'].unique():
                        n_subset = baseline_df[baseline_df['N'] == n_value]
                        if not n_subset.empty:
                            benchmark_soc_by_n[n_value] = n_subset['soc'].mean()
        
        # 各データポイントを対応するN値のベンチマークSOCで正規化
        if benchmark_soc_by_n:
            df['benchmark_normalized'] = df.apply(
                lambda row: row['soc'] / benchmark_soc_by_n[row['N']] 
                if row['N'] in benchmark_soc_by_n and benchmark_soc_by_n[row['N']] > 0 
                else np.nan, 
                axis=1
            )
            df.dropna(subset=['benchmark_normalized'], inplace=True)
    
    # バイオリンプロットと散布図の両方を作成
    # plot_violin(df, output_dir, vary_property, plot_settings, baseline_data)
    plot_scatter(df, output_dir, vary_property, baseline_data, plot_settings)

def find_latest_experiment_dir():
    """最新の実験ディレクトリを見つける"""
    import glob
    import os
    
    results_dir = "results"
    if not os.path.exists(results_dir):
        return None
    
    # YYYYMMDD_HHMMSS形式のディレクトリを探す
    pattern = os.path.join(results_dir, "????????_??????")
    experiment_dirs = glob.glob(pattern)
    
    if not experiment_dirs:
        return None
    
    # 最新のディレクトリを返す（名前でソートして最後のもの）
    return sorted(experiment_dirs)[-1]

def main():
    parser = argparse.ArgumentParser(description="実験データを可視化")
    parser.add_argument('--data_dir', type=str, default=None,
                        help="実験データが保存されているディレクトリ（未指定の場合は最新の実験結果を使用）")
    parser.add_argument('--output_dir', type=str, default=None,
                        help="グラフを保存するディレクトリ（未指定の場合はdata_dirと同じ）")
    args = parser.parse_args()
    
    # data_dirが指定されていない場合は最新の実験ディレクトリを使用
    if args.data_dir is None:
        latest_dir = find_latest_experiment_dir()
        if latest_dir is None:
            print("エラー: results/ディレクトリに実験結果が見つかりません。")
            print("先にrun_data_generation.pyを実行するか、--data_dirを指定してください。")
            return
        args.data_dir = latest_dir
        print(f"最新の実験ディレクトリを使用: {args.data_dir}")
    
    # 出力ディレクトリの設定
    output_dir = args.output_dir if args.output_dir else args.data_dir
    
    # 実験データの読み込み
    try:
        all_results, baseline_results = load_experiment_data(args.data_dir)
        print(f"実験データを読み込みました: {args.data_dir}")
    except FileNotFoundError as e:
        print(f"エラー: {e}")
        return
    
    # プロット設定の読み込み
    plot_settings = load_plot_settings(args.data_dir)
    if plot_settings:
        print(f"プロット設定を読み込みました")
    
    # 各パラメータ/コラボレーションについてグラフを作成
    for plot_key, results_list in all_results.items():
        if results_list:
            # マップごとにデータを分割
            df_temp = pd.DataFrame(results_list)
            # マップ情報の列名を確認 (通常は 'm' だが念のため)
            map_col = 'm' if 'm' in df_temp.columns else None
            
            if map_col:
                # マップごとにループ
                unique_maps = df_temp[map_col].unique()
                for map_file in unique_maps:
                    # マップ名（ファイル名）を取得してディレクトリ名に使用
                    map_name = os.path.splitext(os.path.basename(str(map_file)))[0]
                    map_specific_results = [r for r in results_list if r.get(map_col) == map_file]
                    
                    if not map_specific_results:
                        continue
                        
                    plot_output_dir = os.path.join(output_dir, plot_key, map_name)
                    os.makedirs(plot_output_dir, exist_ok=True)
                    
                    print(f"\n{plot_key} のグラフを作成中 (Map: {map_name})...")
                    # ベースラインも同じマップのものだけフィルタリングすべきだが、
                    # 現状ベースラインデータの構造上、マップ情報との紐付けが厳密でない場合もあるため、
                    # 簡易的に全てのベースラインを渡すか、あるいはベースライン側も微修正が必要。
                    # ここでは一旦すべてのベースラインを渡す（比較対象として全て表示するため）
                    plot_results(map_specific_results, plot_output_dir, plot_key, plot_settings, baseline_results)
            else:
                # マップ情報がない場合は従来通りまとめてプロット
                plot_output_dir = os.path.join(output_dir, plot_key, "all_maps")
                os.makedirs(plot_output_dir, exist_ok=True)
                
                print(f"\n{plot_key} のグラフを作成中 (All Maps)...")
                plot_results(results_list, plot_output_dir, plot_key, plot_settings, baseline_results)
    
    print(f"\n可視化が完了しました。結果は {output_dir} に保存されています。")

if __name__ == "__main__":
    main()