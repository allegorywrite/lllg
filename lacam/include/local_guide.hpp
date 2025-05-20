/*
 * windowed soft PP
 */

#pragma once
#include "collision_table.hpp"
#include "dist_table.hpp"
#include "global_guide.hpp"
#include "graph.hpp"
#include "instance.hpp"
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
  static bool USE_COLLISION_BASED_WINDOW;  // 衝突量ベースのウィンドウサイズ調整を有効にするかどうか
  static int MIN_WINDOW;       // 最小ウィンドウサイズ
  static int MAX_WINDOW;       // 最大ウィンドウサイズ
  static float OCCUPANCY_THRESHOLD;  // 占有率の閾値
  static float COLLISION_THRESHOLD;  // 衝突量の閾値

  // guidance
  GlobalGuide* global_guide;

  LocalGuide(const Instance* _ins, DistTable* _D, int seed = 0,
             GlobalGuide* _global_guide = nullptr);
  ~LocalGuide();

  void construct(const Config& Q_from, const std::vector<int>& order);
  LocalHeuristic get(const int i, Vertex* v);

  // 履歴関連のメソッド
  void clear_history();  // 履歴をクリア
  void save_current_paths();  // 現在の参照軌道を履歴に保存
  const std::vector<Path>& get_paths_at_step(int step) const;  // 特定のステップの参照軌道を取得
  int get_history_size() const;  // 履歴のサイズを取得

  // 占有率計算と動的ウィンドウサイズ調整のメソッド
  float calculate_occupancy(const int i, const Config& Q_from);  // エージェントiの周りの占有率を計算
  float calculate_collision_rate(const int i, const Path& path);  // エージェントiの参照軌道の衝突率を計算
  void update_window_size(const int i, const Config& Q_from);    // 占有率または衝突量に基づいてウィンドウサイズを更新
};
