/*
 * windowed soft PP
 */

#pragma once
#include "collision_table.hpp"
#include "dist_table.hpp"
#include "global_guide.hpp"
#include "graph.hpp"
#include "instance.hpp"
#include "sipp.hpp"  // Add SIPP header
#include "utils.hpp"

// windowed soft prioritized planning
struct WSPPNode {
  int when;
  Vertex* where;
  float g;
  float h;
  float f;
  WSPPNode* parent;

  WSPPNode(int _when = 0, Vertex* _where = nullptr, int _g = 0, int _h = 0,
           float _f = 0, WSPPNode* _parent = nullptr)
      : when(_when), where(_where), g(_g), h(_h), f(g + h), parent(_parent)
  {
  }
  ~WSPPNode(){};
};
using WSPPNodes = std::vector<WSPPNode*>;

using LocalHeuristic = int;

// ウィンドウサイズの更新判定タイプを定義
enum class WindowUpdateType {
  ACCESS_COUNT,  // アクセス回数ベース
  OCCUPANCY,     // 占有率ベース
  COLLISION      // 衝突量ベース
};

struct LocalGuide {
  const Instance* ins;
  std::mt19937 MT;

  // solver utils
  const int N;  // number of agents
  const int V_size;
  DistTable* D;

  // specific to solver
  CollisionTable CT;
  std::vector<Path> guide_paths;
  
  // 並列計算用のヘルパー関数
  Path computeGuidePath(int agent_id, const Config& Q_from);
  Path computeGuidePathCorrect(int agent_id, const Config& Q_from);
  Path computeGuidePathWithCT(int agent_id, const Config& Q_from, CollisionTable& ct);

  // 参照軌道の履歴を保存
  std::vector<std::vector<Path>> guide_paths_history;  // 各ステップでの参照軌道の履歴
  int current_step;  // 現在のステップ数

  WSPPNodes wspp_nodes;
  std::vector<WSPPNodes> CLOSED;
  Config Q_to;

  // hyper parameters
  static bool ON;
  static std::vector<int> WINDOWS;  // 各エージェントのウィンドウサイズ
  static int NUM_REFINE;
  static bool DYNAMIC_WINDOW;  // 動的ウィンドウサイズの有効/無効
  static WindowUpdateType WINDOW_UPDATE_TYPE;  // ウィンドウサイズ更新の判定タイプ
  static int MIN_WINDOW;       // 最小ウィンドウサイズ
  static int MAX_WINDOW;       // 最大ウィンドウサイズ
  static float OCCUPANCY_THRESHOLD;  // 占有率の閾値
  static float COLLISION_THRESHOLD;  // 衝突量の閾値
  static float ACCESS_COUNT_THRESHOLD;  // アクセス回数の閾値
  static float COLLISION_COST;  // 衝突コスト
  static float COLLISION_COST_ORDER;  // 衝突コストの係数
  static float GLOBAL_GUIDE_FIRST_ORDER;  // グローバルガイダンスの係数
  static float GLOBAL_GUIDE_SECOND_ORDER;  // グローバルガイダンスの係数
  static bool ENABLE_IMPROVED_HEURISTIC;  // 改善されたヒューリスティック関数の有効/無効
  static bool ENABLE_COLLISION_SORT;      // 衝突コストソートの有効/無効
  static bool ENABLE_OPTIMIZED_GUIDANCE;   // 最適化されたガイダンス計算の有効/無効
  static bool ENABLE_EARLY_TERMINATION;    // 早期終了の有効/無効
  static bool ENABLE_READONLY_PARALLEL_UPDATE; // 読み取り専用並列update_guide_pathの有効/無効
  static bool USE_SOFT_SIPP;               // ソフト制約SIPP (SIPPS) の使用フラグ
  static int GRID_PARTITION_SIZE;          // NxN grid partitioning size
  static bool ENABLE_K_STEP_UPDATE;       // k-step local guidance update の有効/無効
  static int K_STEP_INTERVAL;             // k-step update の間隔

  // guidance
  GlobalGuide* global_guide;
  bool use_sipp_; // Flag to use SIPP

  std::vector<int> node_access_counts;  // エージェントごとのノードアクセス回数の累積
  std::vector<float> cached_collision_costs;  // A*探索時の衝突コストをキャッシュ
  std::vector<int> step_counters;          // k-step update用の各エージェントのステップカウンタ

  LocalGuide(const Instance* _ins, DistTable* _D, int seed = 0,
             GlobalGuide* _global_guide = nullptr, bool _use_sipp = false);
  ~LocalGuide();

  void construct(const Config& Q_from, const std::vector<int>& order);
  LocalHeuristic get(const int i, Vertex* v);

  // 履歴関連のメソッド
  void clear_history();  // 履歴をクリア
  void save_current_paths();  // 現在の参照軌道を履歴に保存
  const std::vector<Path>& get_paths_at_step(int step) const;  // 特定のステップの参照軌道を取得
  int get_history_size() const;  // 履歴のサイズを取得

  // HNodeとの連携メソッド
  void set_guide_paths(const std::vector<Path>& paths);  // HNodeから参照軌道を設定
  std::vector<Path> get_current_guide_paths() const;     // 現在の参照軌道を取得
  void reconstruct_solution_paths(const std::vector<std::vector<Path>>& solution_paths);  // ソリューションに対応する参照軌道を再構成

  // 占有率計算と動的ウィンドウサイズ調整のメソッド
  float calculate_occupancy(const int i, const Config& Q_from);  // エージェントiの周りの占有率を計算
  float calculate_collision_rate(const int i, const Path& path);  // エージェントiの参照軌道の衝突率を計算
  void update_window_size(const int i, const Config& Q_from);    // 占有率または衝突量に基づいてウィンドウサイズを更新

  // 設定メソッド
  void set_window_update_type(WindowUpdateType type) { WINDOW_UPDATE_TYPE = type; }
  void set_occupancy_threshold(float threshold) { OCCUPANCY_THRESHOLD = threshold; }
  void set_collision_threshold(float threshold) { COLLISION_THRESHOLD = threshold; }
  void set_access_count_threshold(float threshold) { ACCESS_COUNT_THRESHOLD = threshold; }

  // ウィンドウサイズ更新用の補助関数
  void update_window_by_access_count(const int i);
  void update_window_by_occupancy(const int i, const Config& Q_from);
  void update_window_by_collision(const int i);
  
  // k-step update 判定用のヘルパー関数
  bool should_update_guide_path(int agent_id);
};
