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
  
  // // 並列計算用のヘルパー関数
  // Path computeGuidePath(int agent_id, const Config& Q_from);
  // Path computeGuidePathCorrect(int agent_id, const Config& Q_from);
  // Path computeGuidePathWithCT(int agent_id, const Config& Q_from, CollisionTable& ct);

  // 参照軌道の履歴を保存
  std::vector<std::vector<Path>> guide_paths_history;  // 各ステップでの参照軌道の履歴
  int current_step;  // 現在のステップ数

  WSPPNodes wspp_nodes;
  std::vector<WSPPNodes> CLOSED;
  Config Q_to;

  // hyper parameters
  static bool ON;
  static int WINDOW;
  static int NUM_REFINE;
  static float COLLISION_COST;  // 衝突コスト
  static float COLLISION_COST_ORDER;  // 衝突コストの係数
  static float GLOBAL_GUIDE_FIRST_ORDER;  // グローバルガイダンスの係数
  static float GLOBAL_GUIDE_SECOND_ORDER;  // グローバルガイダンスの係数
  static bool ENABLE_COLLISION_SORT;      // 衝突コストソートの有効/無効
  static bool ENABLE_OPTIMIZED_GUIDANCE;   // 最適化されたガイダンス計算の有効/無効
  static bool ENABLE_K_STEP_UPDATE;       // k-step local guidance update の有効/無効
  static int K_STEP_INTERVAL;             // k-step update の間隔
  static bool ENABLE_PRUNING;

  // guidance
  GlobalGuide* global_guide;
  bool use_sipp_; // Flag to use SIPP

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

  // k-step update 判定用のヘルパー関数
  bool should_update_guide_path(int agent_id);
};
