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
from typing import Optional, Tuple, Any, Iterable

# Try to reuse the same palette/markers as plot_lifelong_scatter.py
try:
    from plot_lifelong_scatter import (  # type: ignore
        ALGORITHM_STYLES as _LIFELONG_ALGORITHM_STYLES,
        # MARKER_SIZES_PT as _LIFELONG_MARKER_SIZES_PT,
        # DEFAULT_MARKER_CYCLE as _LIFELONG_DEFAULT_MARKER_CYCLE,
    )
except Exception:
    _LIFELONG_ALGORITHM_STYLES = {}
    # _LIFELONG_MARKER_SIZES_PT = {
    #     'o': 14, 's': 16, 'D': 13, '^': 15, 'v': 16, 'P': 17, 'X': 17, '*': 21,
    #     'h': 16, 'H': 16, '<': 16, '>': 16, 'p': 18, '8': 18, 'd': 17,
    # }
_LIFELONG_MARKER_SIZES_PT = {
    'X': 17, 'o': 14, '^': 15, 's': 16, '*': 21, 'D': 13, '^': 15, 'v': 16, 'P': 17, 'X': 17, '*': 21,
    'h': 16, 'H': 16, '<': 16, '>': 16, 'p': 18, '8': 18, 'd': 17,
}
_LIFELONG_DEFAULT_MARKER_CYCLE = ['X', 'o', '^', 's', '*', 'D', '^', 'v', 'P', 'X', '*', 'h']

_PRIMARY_COLOR_CYCLE = [
    "#1f77b4",  # blue
    "#d62728",  # red
    "#2ca02c",  # green
    "#ff7f0e",  # orange
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
]


def _distinct_palette(n: int):
    if n <= 0:
        return []
    if n <= len(_PRIMARY_COLOR_CYCLE):
        return _PRIMARY_COLOR_CYCLE[:n]

    palette = list(_PRIMARY_COLOR_CYCLE)
    remaining = n - len(palette)

    if remaining <= 0:
        return palette

    if remaining <= 20:
        palette.extend(sns.color_palette("tab20", n_colors=remaining))
        return palette

    palette.extend(sns.color_palette("tab20", n_colors=20))
    remaining -= 20
    palette.extend(sns.color_palette("husl", n_colors=remaining))
    return palette


def _add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    # Runtime seconds (original runtime is typically milliseconds in this repo)
    if 'runtime_sec' not in df.columns:
        if 'runtime' in df.columns:
            df['runtime_sec'] = pd.to_numeric(df['runtime'], errors='coerce') / 1000.0
        elif 'comp_time_ms' in df.columns:
            df['runtime_sec'] = pd.to_numeric(df['comp_time_ms'], errors='coerce') / 1000.0

    # Makespan / simulation steps (preferred key order)
    if 'makespan' not in df.columns:
        makespan_series = None
        for key in ('simulation_time', 'S', 'simulation_steps', 'steps'):
            if key in df.columns:
                makespan_series = pd.to_numeric(df[key], errors='coerce')
                break
        if makespan_series is None and 'throughput_tasks' in df.columns and 'total_completed_tasks' in df.columns:
            th = pd.to_numeric(df['throughput_tasks'], errors='coerce')
            tct = pd.to_numeric(df['total_completed_tasks'], errors='coerce')
            makespan_series = np.where((th > 0) & np.isfinite(th), tct / th, np.nan)
        if makespan_series is not None:
            df['makespan'] = makespan_series

    # Runtime per step (sec/step), consistent with scripts/plot_lifelong_scatter.py
    if 'runtime_per_step' not in df.columns and 'runtime_sec' in df.columns and 'makespan' in df.columns:
        ms = pd.to_numeric(df['makespan'], errors='coerce')
        rt = pd.to_numeric(df['runtime_sec'], errors='coerce')
        df['runtime_per_step'] = np.where((ms > 0) & np.isfinite(ms), rt / ms, np.nan)

    # Throughput tasks per step, consistent with run_data_generation.py output used elsewhere in this repo:
    # throughput_tasks == total_completed_tasks / makespan(S)
    if 'total_completed_tasks' in df.columns and 'makespan' in df.columns:
        tct = pd.to_numeric(df['total_completed_tasks'], errors='coerce')
        ms = pd.to_numeric(df['makespan'], errors='coerce')
        derived_th = np.where((ms > 0) & np.isfinite(ms), tct / ms, np.nan)
        if 'throughput_tasks' not in df.columns:
            df['throughput_tasks'] = derived_th
        else:
            th = pd.to_numeric(df['throughput_tasks'], errors='coerce')
            df['throughput_tasks'] = np.where(np.isnan(th), derived_th, th)

    return df


def _marker_area(marker: str, scale: float = 1.0) -> float:
    pt = _LIFELONG_MARKER_SIZES_PT.get(marker, 14)
    return (pt * pt) * scale


def _try_int(value):
    try:
        f = float(value)
        if f.is_integer():
            return int(f)
        return None
    except Exception:
        return None


def _extract_config_value(config_str: str, key: str):
    if not isinstance(config_str, str) or not key:
        return None
    match = re.search(rf'{re.escape(key)}=([^_]+)', config_str)
    if not match:
        return None
    raw = match.group(1)
    iv = _try_int(raw)
    if iv is not None:
        return iv
    if raw.lower() in ['true', 'false']:
        return raw.lower() == 'true'
    return raw


def _config_keys(config_str: str) -> list[str]:
    if not isinstance(config_str, str):
        return []
    keys: list[str] = []
    # Keys in `collab_config_str` are expected to start with a letter/underscore.
    # This avoids mis-parsing patterns like `lg_window=5_lacam_horizon=20` as key `5_lacam_horizon`.
    # Also avoid capturing the underscore separator itself (e.g. `_lacam_horizon`).
    for k in re.findall(r'(?:^|_)([a-zA-Z][a-zA-Z0-9_]*)=', config_str):
        if k and k not in keys:
            keys.append(k)
    return keys


def _resolve_line_color(
    line_property: str,
    line_value,
    fallback_idx: int = 0,
    *,
    use_algorithm_style_colors: bool = False,
):
    if use_algorithm_style_colors and line_property == 'lacam_horizon':
        horizon = _try_int(line_value)
        if horizon is not None:
            key = f'lifelong_lg_lacam_horizon_{horizon}'
            style = _LIFELONG_ALGORITHM_STYLES.get(key)
            if style and 'color' in style:
                return style['color']
    return _PRIMARY_COLOR_CYCLE[fallback_idx % len(_PRIMARY_COLOR_CYCLE)]


def _collab_key_to_props(collab_key: str):
    # Legacy helper kept for backward compatibility; prefer parsing keys from `collab_config_str`.
    if not isinstance(collab_key, str):
        return []
    parts = [p for p in collab_key.split('_') if p]
    if parts and parts[-1].lower() == 'collab':
        parts = parts[:-1]
    return parts


def _short_annotate_key(key: str | None) -> str:
    if not key:
        return ''
    if key == 'lg_window':
        return 'window'
    if key == 'lacam_horizon':
        return 'horizon'
    return key


def _legend_sort_key(label: str):
    if label is None:
        return (3, "")
    s = str(label).strip()
    if not s:
        return (3, "")
    # Accept "20 window", "window=20", or plain "20".
    m = re.match(r'^\s*([-+]?\d+(?:\.\d+)?)\b', s)
    if not m:
        m = re.search(r'=\s*([-+]?\d+(?:\.\d+)?)\b', s)
    if not m:
        m = re.search(r'([-+]?\d+(?:\.\d+)?)\s*$', s)
    if m:
        try:
            return (0, float(m.group(1)))
        except Exception:
            pass
    return (2, s)

def _map_name_from_path(value: object) -> Optional[str]:
    if value is None:
        return None
    s = str(value)
    if not s:
        return None
    base = os.path.basename(s)
    stem, _ext = os.path.splitext(base)
    return stem or None


def _map_name_from_scenario_path(value: object) -> Optional[str]:
    if value is None:
        return None
    s = str(value)
    if not s:
        return None
    base = os.path.basename(s)
    stem, _ext = os.path.splitext(base)
    if not stem:
        return None
    # Typical: "<mapname>-random-<k>.scen"
    m = re.match(r'^(.*?)-random-\d+$', stem)
    if m:
        return m.group(1) or None
    return stem


def _scenario_id_from_value(value: object) -> Optional[str]:
    if value is None:
        return None
    s = str(value)
    if not s:
        return None
    base = os.path.basename(s)
    stem, _ext = os.path.splitext(base)
    return stem or None


def _map_name_from_scenario_id(scenario_id: object) -> Optional[str]:
    if scenario_id is None:
        return None
    s = str(scenario_id)
    if not s:
        return None
    m = re.match(r'^(.*?)-random-\d+$', s)
    if m:
        return m.group(1) or None
    return s


def _infer_map_groups(df: pd.DataFrame) -> Tuple[Optional[str], Optional[pd.Series]]:
    if df is None or df.empty:
        return None, None

    for col in ('map_name', 'm', 'map', 'map_file', 'map_path'):
        if col in df.columns and df[col].notna().any():
            return col, df[col]

    # Fallback: infer from scenario path.
    if 'i' in df.columns and df['i'].notna().any():
        return 'i', df['i']

    return None, None


def _read_head(path: str, max_bytes: int = 65536) -> str:
    try:
        with open(path, 'rb') as f:
            data = f.read(max_bytes)
        return data.decode('utf-8', errors='ignore')
    except Exception:
        return ""


def _read_tail(path: str, max_bytes: int = 131072) -> str:
    try:
        with open(path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            start = max(0, size - max_bytes)
            f.seek(start, os.SEEK_SET)
            data = f.read()
        return data.decode('utf-8', errors='ignore')
    except Exception:
        return ""


def _parse_result_text(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    if not text:
        return data
    for line in text.splitlines():
        if not line:
            continue
        if "Lifelong summary:" in line:
            m = re.search(r"total_completed_tasks=(\d+)", line)
            if m:
                data["total_completed_tasks"] = m.group(1)
            m = re.search(r"comp_time_ms=([\d\\.]+)", line)
            if m:
                data["comp_time_ms"] = m.group(1)
                data["runtime"] = m.group(1)
            m = re.search(r"throughput_tasks/s=([\d\\.]+)", line)
            if m:
                data["throughput_tasks"] = m.group(1)
            m = re.search(r"throughput_makespan/s=([\d\\.]+)", line)
            if m:
                data["throughput_makespan"] = m.group(1)

        if '=' in line and not line.lstrip().startswith('#'):
            try:
                key, value = line.strip().split('=', 1)
            except ValueError:
                continue
            k = key.strip()
            v = value.strip()
            if k:
                data[k] = v
    return data


def _parse_collab_config_str(collab_config_str: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not collab_config_str:
        return out
    for key in _config_keys(collab_config_str):
        out[key] = _extract_config_value(collab_config_str, key)
    return out


def _iter_result_files(data_dir: str) -> Iterable[tuple[str, str, str]]:
    """
    Yield (plot_key, collab_config_str, file_path) for result_*.txt under:
      <data_dir>/<plot_key>/<collab_config_str>/result_*.txt
    """
    try:
        for plot_key in os.listdir(data_dir):
            plot_path = os.path.join(data_dir, plot_key)
            if not os.path.isdir(plot_path):
                continue
            for collab_config_str in os.listdir(plot_path):
                cfg_path = os.path.join(plot_path, collab_config_str)
                if not os.path.isdir(cfg_path):
                    continue
                for name in os.listdir(cfg_path):
                    if not (name.startswith('result_') and name.endswith('.txt')):
                        continue
                    yield plot_key, collab_config_str, os.path.join(cfg_path, name)
    except FileNotFoundError:
        return


def _augment_from_result_files(data_dir: str, all_results: Any) -> Any:
    if not isinstance(all_results, dict):
        return all_results

    existing_ids: set[tuple] = set()
    for plot_key, recs in all_results.items():
        if not isinstance(recs, list):
            continue
        for r in recs:
            if not isinstance(r, dict):
                continue
            scenario_id = _scenario_id_from_value(r.get('i')) or str(r.get('scenario_id') or "")
            collab_config_str = r.get('collab_config_str')
            rid = (plot_key, collab_config_str, scenario_id, r.get('N'))
            existing_ids.add(rid)

    added = 0
    for plot_key, collab_config_str, path in _iter_result_files(data_dir):
        if plot_key not in all_results or not isinstance(all_results.get(plot_key), list):
            all_results[plot_key] = []

        filename = os.path.basename(path)
        m = re.match(r'^result_(.+?)_(N\d+)_([^_]+)_(.+)\.txt$', filename)
        scenario_id = None
        N = None
        if m:
            scenario_id = m.group(1)
            try:
                N = int(m.group(2)[1:])
            except Exception:
                N = None
        else:
            m2 = re.match(r'^result_(.+?)_(.+)\.txt$', filename)
            if m2:
                scenario_id = m2.group(1)

        record_id = (plot_key, collab_config_str, scenario_id or "", N)
        if record_id in existing_ids:
            continue

        parsed = {}
        parsed.update(_parse_result_text(_read_head(path)))
        parsed.update(_parse_result_text(_read_tail(path)))

        rec: dict[str, Any] = {}
        rec.update(parsed)
        rec.update(_parse_collab_config_str(collab_config_str))
        rec['collab_key'] = plot_key
        rec['collab_config_str'] = collab_config_str
        if scenario_id:
            rec['scenario_id'] = scenario_id
            rec['map_name'] = _map_name_from_scenario_id(scenario_id)
        if N is not None:
            rec['N'] = N

        all_results[plot_key].append(rec)
        existing_ids.add(record_id)
        added += 1

    if added:
        print(f"補足: experiment_data.json に無い result_*.txt から {added} 件を追加しました")
    return all_results


# Set style to match plot_lifelong_scatter.py as closely as possible
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
    'legend.frameon': False,
    'legend.fancybox': False,
    'legend.edgecolor': 'black',
    'legend.facecolor': 'white',
    'legend.framealpha': 1.0,
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
    
    all_results = data['main_results']
    baseline_results = data['baseline_results']
    all_results = _augment_from_result_files(data_dir, all_results)
    return all_results, baseline_results

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
        'runtime': 'Runtime per step (sec/step)',
        'runtime_per_step': 'Runtime per step (sec/step)',
        'runtime_sec': 'Runtime (sec)',
        'makespan': 'Makespan / Simulation steps',
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

def add_mean_line_to_scatter(
    df,
    grouping_column,
    x_col,
    y_col,
    line_property,
    is_collab_plot,
    plot_settings=None,
    color_index=0,
    legend_order='desc',
    annotate_rightmost_only: bool = False,
):
    """散布図に平均値を線で結ぶ機能を追加する"""
    import re
    from collections import defaultdict

    marker_cycle = list(_LIFELONG_DEFAULT_MARKER_CYCLE)
    debug_colors = bool(int(os.environ.get("LIFELONG_LACAM_DEBUG_COLORS", "0")))
    use_algorithm_style_colors = False
    if plot_settings and isinstance(plot_settings, dict):
        debug_colors = bool(plot_settings.get("debug_colors", debug_colors))
        use_algorithm_style_colors = bool(plot_settings.get("use_algorithm_style_colors", False))
    x_axis_setting = None
    if plot_settings and isinstance(plot_settings, dict):
        violin_params = plot_settings.get('violin_params', {})
        if isinstance(violin_params, dict):
            x_axis_setting = violin_params.get('x_axis')
    is_runtime_x_axis = (x_col in ('runtime', 'runtime_per_step')) or (x_axis_setting == 'runtime')

    debug_annotate = bool(int(os.environ.get("LIFELONG_LACAM_DEBUG_ANNOTATE", "0")))
    
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
                    match = re.search(f'{line_property}=(.+?)(?:_[a-zA-Z][a-zA-Z0-9_]*=|$)', group_val)
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

                            keys = [k for k in _config_keys(str(group_val)) if k != line_property]

                            for key in keys:
                                other_properties[key] = _extract_config_value(str(group_val), key)
                            
                            # line_propertyの値でグループ分け
                            line_groups[line_value_numeric].append((other_properties, mean_x, mean_y, group_val))
                except Exception as e:
                    print(f"Error parsing group_val: {group_val}, error: {e}")
                    continue
        
        print(f"Debug: line_property={line_property}")
        print(f"Debug: Found {len(line_groups)} line groups: {list(line_groups.keys())}")
        for l_val, items in line_groups.items():
            print(f"  Line val {l_val}: {len(items)} points")

        distinct_values_by_key = defaultdict(set)
        for items in line_groups.values():
            for other_properties, _, _, _ in items:
                for k, v in other_properties.items():
                    if v is not None:
                        distinct_values_by_key[k].add(v)

        marker_key = None
        marker_map = {}
        if is_runtime_x_axis and distinct_values_by_key:
            marker_key = max(distinct_values_by_key.items(), key=lambda kv: len(kv[1]))[0]

            def _sort_marker_value(v):
                if isinstance(v, bool):
                    return (0, int(v))
                if isinstance(v, (int, float)):
                    return (1, float(v))
                return (2, str(v))

            distinct_values = sorted(list(distinct_values_by_key.get(marker_key, [])), key=_sort_marker_value)
            marker_map = {v: marker_cycle[i % len(marker_cycle)] for i, v in enumerate(distinct_values)}

        # Fallback: if `line_property` is one key, prefer annotating the other key.
        if marker_key is None and line_groups:
            try:
                any_group_val = next(iter(next(iter(line_groups.values()))))[3]
            except Exception:
                any_group_val = None
            other_keys = [k for k in _config_keys(str(any_group_val)) if k != line_property]
            if len(other_keys) == 1:
                marker_key = other_keys[0]
        if debug_annotate:
            try:
                distinct_for_marker = sorted(list(distinct_values_by_key.get(marker_key, []))) if marker_key else None
            except Exception:
                distinct_for_marker = None
            print(f"Debug(annotate): is_runtime_x_axis={is_runtime_x_axis} marker_key={marker_key} distinct={distinct_for_marker}")

        rightmost_line_value = None
        if annotate_rightmost_only and line_groups:
            max_x_global = -np.inf
            for lv, items in line_groups.items():
                if not items:
                    continue
                max_x_local = max(float(item[1]) for item in items)
                if max_x_local > max_x_global:
                    max_x_global = max_x_local
                    rightmost_line_value = lv
        if debug_annotate:
            print(f"Debug(annotate): annotate_rightmost_only={annotate_rightmost_only} rightmost_line_value={rightmost_line_value}")

        label_key_preference = 'lacam_horizon' if line_property != 'lacam_horizon' else None
        
        # 最後のトレンド線（最大のline_value）を特定
        max_line_value = max(line_groups.keys()) if line_groups else None
        
        # line_groups をlegend_orderに基づいてソート
        sorted_line_groups = sorted(line_groups.items(), key=lambda x: x[0], reverse=(legend_order == 'desc'))

        print("legend order:", legend_order)
        
        for line_idx, (line_value, group_data) in enumerate(sorted_line_groups):
            if len(group_data) < 2:
                continue  # 線を引くには最低2点必要
            
            # 他のプロパティの値でソート（最初に見つかったプロパティを使用）
            if group_data[0][0]:  # other_propertiesが空でない場合
                sort_key = None
                if label_key_preference and (label_key_preference in group_data[0][0]):
                    sort_key = label_key_preference
                elif marker_key and (marker_key in group_data[0][0]):
                    sort_key = marker_key
                else:
                    sort_key = list(group_data[0][0].keys())[0]

                def _sort_val(v):
                    if isinstance(v, bool):
                        return (0, int(v))
                    if isinstance(v, (int, float)):
                        return (1, float(v))
                    if v is None:
                        return (3, "")
                    return (2, str(v))

                group_data.sort(key=lambda x: _sort_val(x[0].get(sort_key)))
            
            # 平均値の点を線で結ぶ
            x_values = [item[1] for item in group_data]
            y_values = [item[2] for item in group_data]
            
            color = _resolve_line_color(
                line_property,
                line_value,
                fallback_idx=line_idx,
                use_algorithm_style_colors=use_algorithm_style_colors,
            )
            if debug_colors:
                print(f"Debug(colors): mean-line {line_property}={line_value} -> {color}")
            
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
                label_text = str(label_value) + " window"
            else:
                legend_property_name = custom_legend if custom_legend else line_property
                label_text = f'{legend_property_name}={label_value}'
            
            ax = plt.gca()
            if not is_runtime_x_axis:
                legend_marker = marker_cycle[line_idx % len(marker_cycle)]
                ax.plot(
                    [],
                    [],
                    linestyle='-',
                    color=color,
                    linewidth=2,
                    marker=legend_marker,
                    markersize=float(_LIFELONG_MARKER_SIZES_PT.get(legend_marker, 12)),
                    markerfacecolor='white',
                    markeredgecolor=color,
                    markeredgewidth=2.0,
                    label=label_text,
                )
                plt.plot(x_values, y_values, linestyle='-', color=color, alpha=0.8, linewidth=2, zorder=10, label=None)
            else:
                plt.plot(x_values, y_values, linestyle='-', color=color, alpha=0.8, linewidth=2, zorder=10, label=label_text)
            
            # plt.xticks(x_values)
            # plt.yticks(y_values)
            
            # 平均値の点をマークで表示
            if is_runtime_x_axis:
                for point_idx, item in enumerate(group_data):
                    other_properties, xv, yv, _ = item
                    marker_val = other_properties.get(marker_key) if marker_key else None
                    marker = marker_map.get(marker_val, marker_cycle[point_idx % len(marker_cycle)])
                    plt.scatter([xv], [yv], marker=marker, s=_marker_area(marker, scale=1.0),
                                facecolors='white', edgecolors=color, linewidths=2.0, zorder=11)
            else:
                marker = marker_cycle[line_idx % len(marker_cycle)]
                plt.scatter(x_values, y_values, marker=marker, s=_marker_area(marker, scale=1.0),
                            facecolors='white', edgecolors=color, linewidths=2.0, zorder=11)
            
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
            should_annotate = (not hide_data_label) or bool(annotate_rightmost_only)
            if should_annotate:
                if annotate_rightmost_only:
                    if (rightmost_line_value is None) or (line_value != rightmost_line_value):
                        should_annotate = False
                else:
                    if line_value != max_line_value:
                        should_annotate = False

            if should_annotate:
                label_key = label_key_preference or marker_key
                if not label_key and group_data and group_data[0][0]:
                    label_key = next(iter(group_data[0][0].keys()))
                if debug_annotate:
                    print(f"Debug(annotate): line={line_property}={line_value} label_key={label_key}")

                def _sort_annotate_value(v):
                    if isinstance(v, bool):
                        return (0, int(v))
                    if isinstance(v, (int, float)):
                        return (1, float(v))
                    if v is None:
                        return (3, "")
                    return (2, str(v))

                annotate_vals = []
                for other_properties, _x_coord, _y_coord, _group_val in group_data:
                    val = other_properties.get(label_key) if label_key else None
                    if val is not None:
                        annotate_vals.append(val)
                min_annotate_val = min(annotate_vals, key=_sort_annotate_value) if annotate_vals else None
                min_labeled = False
                if debug_annotate:
                    print(f"Debug(annotate): min_annotate_val={min_annotate_val}")

                for item in group_data:
                    other_properties, x_coord, y_coord, _group_val = item
                    val = other_properties.get(label_key) if label_key else None
                    if val is None:
                        continue
                    if debug_annotate:
                        print(f"Debug(annotate): point val={val} x={x_coord} y={y_coord} other={other_properties}")
                    if isinstance(val, float) and val.is_integer():
                        text = str(int(val))
                    elif isinstance(val, bool):
                        text = 'T' if val else 'F'
                    else:
                        text = str(val)
                    if (not min_labeled) and (min_annotate_val is not None) and (val == min_annotate_val):
                        key_prefix = _short_annotate_key(label_key)
                        if key_prefix:
                            text = f"{key_prefix}={text}"
                            min_labeled = True
                    plt.annotate(
                        text,
                        (x_coord, y_coord),
                        xytext=(-5, -30),
                        textcoords='offset points',
                        fontsize=22,
                        ha='left',
                    )
        
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
        # 最後のトレンド線（最大のline_value）を特定
        max_line_value = max(line_groups.keys()) if line_groups else None

        rightmost_line_value = None
        if annotate_rightmost_only and line_groups:
            max_x_global = -np.inf
            for lv, items in line_groups.items():
                if not items:
                    continue
                max_x_local = max(float(item[1]) for item in items)
                if max_x_local > max_x_global:
                    max_x_global = max_x_local
                    rightmost_line_value = lv
        
        # line_groups をlegend_orderに基づいてソート
        sorted_line_groups_2 = sorted(line_groups.items(), key=lambda x: x[0], reverse=(legend_order == 'desc'))
        
        for line_idx, (line_value, group_data) in enumerate(sorted_line_groups_2):
            if len(group_data) < 2:
                continue  # 線を引くには最低2点必要
            
            # group_valでソート
            group_data.sort(key=lambda x: x[0])
            
            # 平均値の点を線で結ぶ
            x_values = [item[1] for item in group_data]
            y_values = [item[2] for item in group_data]
            
            color = _resolve_line_color(
                line_property,
                line_value,
                fallback_idx=line_idx,
                use_algorithm_style_colors=use_algorithm_style_colors,
            )
            if debug_colors:
                print(f"Debug(colors): mean-line {line_property}={line_value} -> {color}")
            
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
            
            ax = plt.gca()
            if not is_runtime_x_axis:
                legend_marker = marker_cycle[line_idx % len(marker_cycle)]
                ax.plot(
                    [],
                    [],
                    linestyle='--',
                    color=color,
                    linewidth=2,
                    marker=legend_marker,
                    markersize=float(_LIFELONG_MARKER_SIZES_PT.get(legend_marker, 12)),
                    markerfacecolor='white',
                    markeredgecolor=color,
                    markeredgewidth=2.0,
                    label=label_text,
                )
                plt.plot(x_values, y_values, linestyle='--', color=color, alpha=0.8, linewidth=2, zorder=10, label=None)
            else:
                plt.plot(x_values, y_values, linestyle='--', color=color, alpha=0.8, linewidth=2, zorder=10, label=label_text)
            
            # 平均値の点をマークで表示
            if is_runtime_x_axis:
                for point_idx, (xv, yv) in enumerate(zip(x_values, y_values)):
                    marker = marker_cycle[point_idx % len(marker_cycle)]
                    plt.scatter([xv], [yv], marker=marker, s=_marker_area(marker, scale=0.4),
                                facecolors='white', edgecolors=color, linewidths=2.0, zorder=11)
            else:
                marker = marker_cycle[line_idx % len(marker_cycle)]
                plt.scatter(x_values, y_values, marker=marker, s=_marker_area(marker, scale=0.4),
                            facecolors='white', edgecolors=color, linewidths=2.0, zorder=11)
            
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
            should_annotate = (not hide_data_label) or bool(annotate_rightmost_only)
            if should_annotate:
                if annotate_rightmost_only:
                    if (rightmost_line_value is None) or (line_value != rightmost_line_value):
                        should_annotate = False
                else:
                    if line_value != max_line_value:
                        should_annotate = False

            if should_annotate:
                def _sort_group_val(v):
                    if isinstance(v, bool):
                        return (0, int(v))
                    if isinstance(v, (int, float)):
                        return (1, float(v))
                    if v is None:
                        return (3, "")
                    return (2, str(v))

                group_vals = [item[0] for item in group_data if item and item[0] is not None]
                min_group_val = min(group_vals, key=_sort_group_val) if group_vals else None
                min_labeled = False

                for _idx, item in enumerate(group_data):
                    group_val, x_coord, y_coord = item
                    text = str(group_val)
                    if (not min_labeled) and (min_group_val is not None) and (group_val == min_group_val):
                        key_prefix = _short_annotate_key(grouping_column)
                        if key_prefix:
                            text = f"{key_prefix}={text}"
                            min_labeled = True
                    plt.annotate(
                        text,
                        (x_coord, y_coord),
                        xytext=(-5, -30),
                        textcoords='offset points',
                        fontsize=22,
                        ha='left',
                    )
    
    # 凡例は呼び出し元で整理して作成する

def plot_violin(df, output_dir, vary_property, plot_settings, baseline_data=None):
    """バイオリンプロットを作成する"""
    if plot_settings is None:
        plot_settings = {}
    violin_params = plot_settings.get('violin_params', {})
    y_axis_param = violin_params.get('y_axis', 'soc')
    orient = violin_params.get('orient', 'vertical')
    
    # y_colの決定ロジック（派生列を含む）
    y_col = 'sum_of_costs'  # デフォルト
    if y_axis_param == 'runtime':
        df = _add_derived_metrics(df)
        y_col = 'runtime_per_step' if 'runtime_per_step' in df.columns else ('runtime_sec' if 'runtime_sec' in df.columns else ('runtime' if 'runtime' in df.columns else 'comp_time_ms'))
    elif y_axis_param == 'runtime_per_step':
        df = _add_derived_metrics(df)
        y_col = 'runtime_per_step'
    elif y_axis_param == 'runtime_sec':
        df = _add_derived_metrics(df)
        y_col = 'runtime_sec'
    elif y_axis_param in df.columns:
        y_col = y_axis_param
    elif y_axis_param == 'benchmark_normalized':
        y_col = 'benchmark_normalized'
    elif y_axis_param == 'soc_normalized':
        y_col = 'soc_normalized'
    elif y_axis_param == 'comp_time_ms':
        y_col = 'comp_time_ms'
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

                    baseline_df = _add_derived_metrics(baseline_df)
                    
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

    df = _add_derived_metrics(df)
    
    # データの準備
    if grouping_column not in df.columns:
        print(f"エラー: グループ化列 '{grouping_column}' がDataFrameに存在しません。")
        return
    
    if y_axis_param not in df.columns:
        print(f"エラー: y軸パラメータ '{y_axis_param}' がDataFrameに存在しません。")
        return
    
    # データのフィルタリング
    valid_data = df.dropna(subset=[grouping_column, y_col])
    
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
        other_palette = _distinct_palette(num_other_colors)
        
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
        palette = _distinct_palette(num_colors)
    
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
        # if is_collab_plot:
        #     plt.xlabel(f"{vary_property} Configuration", fontweight='bold', fontsize=22)
        # else:
        #     plt.xlabel(get_axis_label(vary_property), fontweight='bold', fontsize=22)
        
        # plt.ylabel(get_axis_label(y_axis_param), fontweight='bold', fontsize=22)
        
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

def _plot_single_scatter(
    df,
    output_dir,
    vary_property,
    baseline_data=None,
    plot_settings=None,
    lg_window_suffix=None,
    color_index=0,
    annotate_rightmost_only: bool = False,
):
    """単一の散布図を作成するヘルパー関数"""
    plt.figure(figsize=(8, 8))
    
    # vary_property がコラボキーかどうかを判定
    is_collab_plot = 'collab_config_str' in df.columns and df['collab_key'].iloc[0] == vary_property if 'collab_key' in df.columns and not df.empty else False

    grouping_column = 'collab_config_str' if is_collab_plot else vary_property
    
    if grouping_column not in df.columns:
        print(f"エラー: プロット用のグループ化列 '{grouping_column}' がDataFrameに存在しません。")
        return

    df = _add_derived_metrics(df)

    unique_group_values = df[grouping_column].unique()
    num_colors = len(unique_group_values)

    if num_colors == 0:
        print(f"警告: プロット対象のデータグループがありません ({grouping_column})。")
        return

    line_property = plot_settings.get('line', None) if plot_settings else None
    marker_cycle = list(_LIFELONG_DEFAULT_MARKER_CYCLE)
    debug_colors = bool(int(os.environ.get("LIFELONG_LACAM_DEBUG_COLORS", "0")))
    use_algorithm_style_colors = False
    if plot_settings and isinstance(plot_settings, dict):
        debug_colors = bool(plot_settings.get("debug_colors", debug_colors))
        use_algorithm_style_colors = bool(plot_settings.get("use_algorithm_style_colors", False))

    marker_property = None
    marker_map = {}

    if is_collab_plot and line_property:
        keys_in_configs: list[str] = []
        for gv in unique_group_values:
            for k in _config_keys(str(gv)):
                if k and k not in keys_in_configs:
                    keys_in_configs.append(k)

        other_keys = [k for k in keys_in_configs if k != line_property]
        if len(other_keys) == 1:
            marker_property = other_keys[0]
        elif other_keys:
            distinct_by_key = {}
            for k in other_keys:
                vals = set()
                for gv in unique_group_values:
                    v = _extract_config_value(str(gv), k)
                    if v is not None:
                        vals.add(v)
                if vals:
                    distinct_by_key[k] = vals
            if distinct_by_key:
                marker_property = max(distinct_by_key.items(), key=lambda kv: len(kv[1]))[0]

        line_value_by_group = {}
        for group_val in unique_group_values:
            line_value_by_group[group_val] = _extract_config_value(str(group_val), line_property)

        unique_line_values = []
        for lv in line_value_by_group.values():
            if lv is not None and lv not in unique_line_values:
                unique_line_values.append(lv)

        def _sort_line_value(v):
            if isinstance(v, bool):
                return (0, int(v))
            if isinstance(v, (int, float)):
                return (1, float(v))
            return (2, str(v))

        unique_line_values.sort(key=_sort_line_value)
        line_color_map = {
            lv: _resolve_line_color(
                line_property,
                lv,
                fallback_idx=i,
                use_algorithm_style_colors=use_algorithm_style_colors,
            )
            for i, lv in enumerate(unique_line_values)
        }
        if debug_colors:
            print(f"Debug(colors): use_algorithm_style_colors={use_algorithm_style_colors}")
            for i, lv in enumerate(unique_line_values):
                print(f"Debug(colors): line_color_map[{i}] {line_property}={lv} -> {line_color_map.get(lv)}")
        color_map = {
            group_val: line_color_map.get(
                line_value_by_group.get(group_val),
                _resolve_line_color(
                    line_property,
                    None,
                    fallback_idx=0,
                    use_algorithm_style_colors=use_algorithm_style_colors,
                ),
            )
            for group_val in unique_group_values
        }

        if marker_property:
            marker_value_by_group = {
                group_val: _extract_config_value(str(group_val), marker_property)
                for group_val in unique_group_values
            }
            unique_marker_values = []
            for mv in marker_value_by_group.values():
                if mv is not None and mv not in unique_marker_values:
                    unique_marker_values.append(mv)
            unique_marker_values.sort(key=_sort_line_value)
            marker_map = {
                mv: marker_cycle[i % len(marker_cycle)]
                for i, mv in enumerate(unique_marker_values)
            }
    else:
        # Default: a distinct color per unique group value
        palette = _distinct_palette(num_colors)
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
    if x_axis_param == 'runtime':
        x_col = 'runtime_per_step' if 'runtime_per_step' in df.columns else ('runtime_sec' if 'runtime_sec' in df.columns else ('runtime' if 'runtime' in df.columns else 'comp_time_ms'))
    elif x_axis_param == 'runtime_per_step':
        x_col = 'runtime_per_step'
    elif x_axis_param == 'runtime_sec':
        x_col = 'runtime_sec'
    elif x_axis_param and x_axis_param in df.columns:
        x_col = x_axis_param
    else:
        x_col = 'runtime' if 'runtime' in df.columns else 'comp_time_ms'
    
    if y_axis_param and y_axis_param in df.columns:
        y_col = y_axis_param
    elif y_axis_param == 'benchmark_normalized':
        y_col = 'benchmark_normalized'
    elif y_axis_param == 'soc_normalized':
        y_col = 'soc_normalized'
    elif y_axis_param == 'runtime':
        y_col = 'runtime_per_step' if 'runtime_per_step' in df.columns else ('runtime_sec' if 'runtime_sec' in df.columns else ('runtime' if 'runtime' in df.columns else 'comp_time_ms'))
    elif y_axis_param == 'runtime_per_step':
        y_col = 'runtime_per_step'
    elif y_axis_param == 'runtime_sec':
        y_col = 'runtime_sec'
    elif y_axis_param == 'comp_time_ms':
        y_col = 'comp_time_ms'
    elif y_axis_param == 'runtime_sec':
        y_col = 'runtime_sec'
    else:
        y_col = 'soc_normalized' if 'soc_normalized' in df.columns else 'soc'

    # Make sure x/y are numeric so ticks can align with data points.
    if x_col in df.columns:
        df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
    if y_col in df.columns:
        df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
    
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

                    baseline_df = _add_derived_metrics(baseline_df)
                    
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
            marker = 'o'
            if is_collab_plot and marker_map and marker_property:
                mv = _extract_config_value(str(group_val), marker_property)
                marker = marker_map.get(mv, marker)
            plt.scatter(x, y, label=label_str, 
                       alpha=0.9, marker=marker, s=_marker_area(marker, scale=0.25),
                       facecolors='white', edgecolors=color_map[group_val],
                       linewidths=2.0)
        
        # 十分なデータポイントがある場合、楕円と重心をプロット
        if len(x) >= 2:
            # 重心
            centroid_x = np.mean(x)
            centroid_y = np.mean(y)
            if not hide_scatter:
                plt.scatter(centroid_x, centroid_y, marker='X', s=100, 
                           facecolors='white', edgecolors=color_map[group_val],
                           linewidths=2.0, zorder=5)
            
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

    extended_palette = _distinct_palette(total_colors)
    
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
                          label=baseline_name, alpha=0.9, s=_marker_area('o', scale=0.25),
                          facecolors='white', edgecolors=color, linewidths=2.0)
            
            # 十分なデータポイントがある場合、楕円と重心をプロット
            if len(baseline_x_values) >= 2:
                # 重心
                centroid_x = np.mean(baseline_x_values)
                centroid_y = np.mean(baseline_y_values)
                if not hide_scatter:
                    plt.scatter(centroid_x, centroid_y, marker='X', s=100, 
                               facecolors='white', edgecolors=color,
                               linewidths=2.0, zorder=5)
                
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
    
    # # y軸ラベルを設定（カスタムラベルまたはデフォルトラベル）
    # if custom_y_label:
    #     plt.ylabel(y_label, fontweight='bold', fontsize=30)
    # else:
    #     # デフォルトのy軸ラベル生成
    #     if y_col == 'soc_normalized':
    #         plt.ylabel("Normalized SOC (SOC/SOC_LB)", fontweight='bold', fontsize=30)
    #     elif y_col == 'benchmark_normalized':
    #         plt.ylabel("Normalized SOC (SOC/Benchmark_SOC)", fontweight='bold', fontsize=30)
    #     elif y_col == 'runtime':
    #         plt.ylabel("Runtime (ms)", fontweight='bold', fontsize=30)
    #     elif y_col == 'runtime_per_step':
    #         plt.ylabel("Runtime per step (sec/step)", fontweight='bold', fontsize=30)
    #     elif y_col == 'comp_time_ms':
    #         plt.ylabel("Computation Time (ms)", fontweight='bold', fontsize=30)
    #     elif y_col == 'runtime_sec':
    #         plt.ylabel("Runtime (sec)", fontweight='bold', fontsize=30)
    #     elif y_col == 'throughput_tasks':
    #         plt.ylabel("Throughput (Tasks/s)", fontweight='bold', fontsize=30)
    #     elif y_col == 'throughput_makespan':
    #         plt.ylabel("Throughput (Makespan/s)", fontweight='bold', fontsize=30)
    #     elif y_col == 'total_completed_tasks':
    #         plt.ylabel("Total Completed Tasks", fontweight='bold', fontsize=30)
    #     else:
    #         plt.ylabel("Total Cost (sum_of_costs)", fontweight='bold', fontsize=30)
    
    #         plt.ylabel("Total Cost (sum_of_costs)", fontweight='bold', fontsize=30)
    
    # if x_col == 'runtime_sec':
    #      plt.xlabel("Runtime (sec)", fontweight='bold', fontsize=30)
    # else:
    #      plt.xlabel(x_label, fontweight='bold', fontsize=30)
    
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
    # ax.set_xscale('log')

    # line設定がある場合、指定されたプロパティで平均値を線で結ぶ
    if plot_settings:
        line_property = plot_settings.get('line', None)
        if line_property:
            # legend_order パラメータを plot_settings から取得
            legend_order = plot_settings.get('legend_order', 'desc') if plot_settings else 'desc'
            add_mean_line_to_scatter(
                df,
                grouping_column,
                x_col,
                y_col,
                line_property,
                is_collab_plot,
                plot_settings,
                color_index,
                legend_order,
                annotate_rightmost_only=annotate_rightmost_only,
            )

    # x軸tick設定（runtime以外はデータ点に合わせる）
    from matplotlib.ticker import MaxNLocator
    x_tick_nbins = None
    if plot_settings:
        x_tick_nbins = plot_settings.get('x_tick_nbins', None)
        if 'violin_params' in plot_settings:
            x_tick_nbins = plot_settings['violin_params'].get('x_tick_nbins', x_tick_nbins)
    if x_tick_nbins is None:
        x_tick_nbins = 6
    try:
        is_numeric_x = x_col in df.columns and pd.api.types.is_numeric_dtype(df[x_col])
    except Exception:
        is_numeric_x = False

    if is_numeric_x:
        want_integer = False
        if x_type and isinstance(x_type, str) and x_type.lower() == 'int':
            try:
                vals = pd.to_numeric(df[x_col], errors='coerce').dropna()
                want_integer = (len(vals) > 0) and bool(np.all(np.isclose(vals, np.round(vals))))
            except Exception:
                want_integer = False

        # If x-axis is not runtime-like, align ticks to actual data points.
        runtime_like = x_col in ('runtime', 'runtime_sec', 'runtime_per_step')
        if not runtime_like:
            try:
                vals = pd.to_numeric(df[x_col], errors='coerce').dropna()
                uniq = sorted(set(float(v) for v in vals if np.isfinite(v)))
            except Exception:
                uniq = []
            if 0 < len(uniq) <= 30:
                if want_integer:
                    uniq = sorted(set(int(round(v)) for v in uniq))
                ax.set_xticks(uniq)
                ax.set_xticklabels([str(v) for v in uniq])
            else:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=x_tick_nbins, integer=want_integer))
        else:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=x_tick_nbins, integer=want_integer))

        ax.tick_params(axis='x', labelsize=26)

    # 凡例の整理（平均線なども含めて最後にまとめる）
    handles, labels = ax.get_legend_handles_labels()
    unique_labels_dict = {}
    for handle, label in zip(handles, labels):
        if not label or str(label).startswith('_'):
            continue
        if label not in unique_labels_dict:
            unique_labels_dict[label] = handle

    if unique_labels_dict:
        legend_order = 'desc'
        if plot_settings and isinstance(plot_settings, dict):
            legend_order = plot_settings.get('legend_order', legend_order)
        items = list(unique_labels_dict.items())
        items.sort(key=lambda kv: _legend_sort_key(kv[0]), reverse=(legend_order == 'desc'))
        sorted_labels = [lab for lab, _h in items]
        sorted_handles = [_h for _lab, _h in items]
        # ax.legend(
        #     sorted_handles,
        #     sorted_labels,
        #     loc='best',
        #     frameon=False,
        #     fancybox=False,
        #     fontsize=23
        # )
    
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

def plot_scatter(df, output_dir, vary_property, baseline_data=None, plot_settings=None, annotate_rightmost_only: bool = False):
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
                
                # 単一のグラフを作成（各グラフで色は青から開始）
                _plot_single_scatter(
                    group_df,
                    output_dir,
                    vary_property,
                    baseline_data,
                    plot_settings,
                    lg_window_val,
                    0,
                    annotate_rightmost_only=annotate_rightmost_only,
                )
            
            return  # div_graphの処理が完了したので関数を終了
        
        # 通常の処理（div_graph=falseまたはline_property!="lg_window"）
        _plot_single_scatter(
            df,
            output_dir,
            vary_property,
            baseline_data,
            plot_settings,
            annotate_rightmost_only=annotate_rightmost_only,
        )

def plot_results(results, output_dir, vary_property, plot_settings=None, baseline_data=None, annotate_rightmost_only: bool = False):
    """結果をグラフにプロットする"""
    df = pd.DataFrame(results)
    
    # 数値型に変換
    for col in ['sum_of_costs', 'soc', 'comp_time_ms', 'runtime', 'soc_lb', 'throughput_tasks', 'throughput_makespan', 'total_completed_tasks']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # runtime_sec計算 (ms -> s)
    df = _add_derived_metrics(df)
    
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
    plot_scatter(df, output_dir, vary_property, baseline_data, plot_settings, annotate_rightmost_only=annotate_rightmost_only)

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
    parser.add_argument('--annotate-rightmost-only', action='store_true',
                        help="平均線のラベル（例: horizon）を一番右のlineのみに付与する")
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
            map_col, map_series = _infer_map_groups(df_temp)

            if map_col and map_series is not None:
                map_names = []
                for v in map_series.dropna().unique():
                    if map_col == 'i':
                        mn = _map_name_from_scenario_path(v)
                    else:
                        mn = _map_name_from_path(v)
                    if mn and mn not in map_names:
                        map_names.append(mn)

                if map_names:
                    for map_name in map_names:
                        if map_col == 'i':
                            map_specific_results = [
                                r for r in results_list
                                if _map_name_from_scenario_path(r.get(map_col)) == map_name
                            ]
                        else:
                            map_specific_results = [
                                r for r in results_list
                                if _map_name_from_path(r.get(map_col)) == map_name
                            ]
                        if not map_specific_results:
                            continue

                        plot_output_dir = os.path.join(output_dir, plot_key, map_name)
                        os.makedirs(plot_output_dir, exist_ok=True)
                        print(f"\n{plot_key} のグラフを作成中 (Map: {map_name})...")
                        plot_results(
                            map_specific_results,
                            plot_output_dir,
                            plot_key,
                            plot_settings,
                            baseline_results,
                            annotate_rightmost_only=bool(args.annotate_rightmost_only),
                        )
                else:
                    map_col = None  # fall back to all_maps

            if not map_col:
                plot_output_dir = os.path.join(output_dir, plot_key, "all_maps")
                os.makedirs(plot_output_dir, exist_ok=True)
                print(f"\n{plot_key} のグラフを作成中 (All Maps)...")
                plot_results(
                    results_list,
                    plot_output_dir,
                    plot_key,
                    plot_settings,
                    baseline_results,
                    annotate_rightmost_only=bool(args.annotate_rightmost_only),
                )
    
    print(f"\n可視化が完了しました。結果は {output_dir} に保存されています。")

if __name__ == "__main__":
    main()
