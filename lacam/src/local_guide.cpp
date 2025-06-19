#include "../include/local_guide.hpp"
#include "../include/graph.hpp"  // get_x, get_y関数を使用するために追加
#include <iostream>  // デバッグログ用
#include <iomanip>   // デバッグログのフォーマット用
#include <thread>    // マルチスレッド用
#include <mutex>     // マルチスレッド用
#include <chrono>    // 時間計測用

// 静的メンバー変数の定義
bool LocalGuide::ON = true;
std::vector<int> LocalGuide::WINDOWS;  // 各エージェントのウィンドウサイズ
int LocalGuide::NUM_REFINE = 1;
bool LocalGuide::DYNAMIC_WINDOW = false;
WindowUpdateType LocalGuide::WINDOW_UPDATE_TYPE = WindowUpdateType::OCCUPANCY;  // デフォルトは占有率ベース
int LocalGuide::MIN_WINDOW = 5;
int LocalGuide::MAX_WINDOW = 20;
float LocalGuide::OCCUPANCY_THRESHOLD = 0.3f;
float LocalGuide::COLLISION_THRESHOLD = 0.5f;
float LocalGuide::ACCESS_COUNT_THRESHOLD = 100.0f;
float LocalGuide::COLLISION_COST = 1.0f;
float LocalGuide::COLLISION_COST_ORDER = 1e-7;
float LocalGuide::GLOBAL_GUIDE_FIRST_ORDER = 1e-2;
float LocalGuide::GLOBAL_GUIDE_SECOND_ORDER = 1e-4;
bool LocalGuide::ENABLE_IMPROVED_HEURISTIC = false;
bool LocalGuide::ENABLE_COLLISION_SORT = false;
bool LocalGuide::ENABLE_OPTIMIZED_GUIDANCE = false;
bool LocalGuide::ENABLE_EARLY_TERMINATION = false;
bool LocalGuide::ENABLE_READONLY_PARALLEL_UPDATE = true;

// 座標変換用の関数
inline int get_x(int k, const Graph* G) { return k % G->width; }
inline int get_y(int k, const Graph* G) { return k / G->width; }

LocalGuide::LocalGuide(const Instance* _ins, DistTable* _D, int seed,
                       GlobalGuide* gg, bool _use_sipp)
    : ins(_ins),
      MT(std::mt19937(seed)),
      N(ins->N),
      V_size(ins->G->size()),
      D(_D),                    // Dを先に初期化
      use_sipp_(_use_sipp),    // use_sipp_を後で初期化
      CT(ins, true),
      guide_paths(N),          // サイズは後で設定
      guide_paths_history(),
      current_step(0),
      CLOSED(),                // サイズは後で設定
      Q_to(N, nullptr),
      global_guide(gg),
      node_access_counts(N, 0), // アクセス回数を0で初期化
      cached_collision_costs(N, 0.0f) // 衝突コストキャッシュを0で初期化
{
  // 各エージェントのウィンドウサイズを初期化
  WINDOWS.resize(N, 10);  // デフォルト値は10
  
  // guide_pathsのサイズを各エージェントのウィンドウサイズに合わせて設定
  for (int i = 0; i < N; ++i) {
    guide_paths[i] = Path(WINDOWS[i], nullptr);
  }
  
  // CLOSEDのサイズを最大ウィンドウサイズに合わせて設定
  CLOSED.resize(MAX_WINDOW, WSPPNodes(V_size, nullptr));
  
  for (auto k = 0; k < 10000; ++k) wspp_nodes.push_back(new WSPPNode());
  clear_history();
}

LocalGuide::~LocalGuide()
{
  for (auto&& n : wspp_nodes)
    if (n != nullptr) delete n;
}

void LocalGuide::clear_history()
{
  guide_paths_history.clear();
  current_step = 0;
}

void LocalGuide::save_current_paths()
{
  // 現在の参照軌道のコピーを作成
  std::vector<Path> current_paths(N);
  for (int i = 0; i < N; ++i) {
    current_paths[i] = guide_paths[i];  // Pathのコピー
  }
  guide_paths_history.push_back(current_paths);
  ++current_step;
}

const std::vector<Path>& LocalGuide::get_paths_at_step(int step) const
{
  if (step < 0 || static_cast<size_t>(step) >= guide_paths_history.size()) {
    throw std::out_of_range("Invalid step number");
  }
  return guide_paths_history[step];
}

int LocalGuide::get_history_size() const
{
  return guide_paths_history.size();
}

float LocalGuide::calculate_occupancy(const int i, const Config& Q_from) {
  if (!DYNAMIC_WINDOW) return 0.0f;

  const auto v = Q_from[i];
  if (v == nullptr) {
    std::cout << "ERROR: agent " << i << " position is null" << std::endl;
    return 0.0f;
  }

  const int radius = 5;  // 周辺のグリッドを考慮する半径
  int occupied = 0;
  int total = 0;

  // エージェントの周辺グリッドをチェック
  for (int dy = -radius; dy <= radius; ++dy) {
    for (int dx = -radius; dx <= radius; ++dx) {
      const int x = v->x + dx;
      const int y = v->y + dy;
      
      // マップの範囲内かチェック
      if (x < 0 || x >= ins->G->width || y < 0 || y >= ins->G->height) {
        continue;
      }

      const int idx = y * ins->G->width + x;
      if (idx < 0 || static_cast<size_t>(idx) >= ins->G->U.size()) {
        std::cout << "ERROR: invalid index " << idx << " for position (" << x << "," << y << ")" << std::endl;
        continue;
      }

      const auto neighbor = ins->G->U[idx];
      
      // グリッドが存在しない（障害物）場合は占有率にカウント
      // if (neighbor == nullptr) {
      //   occupied++;
      //   total++;
      //   continue;
      // }

      // 他のエージェントがいる場合は占有率にカウント
      bool is_occupied = false;
      for (int j = 0; j < N; ++j) {
        if (j != i && Q_from[j] != nullptr && Q_from[j] == neighbor) {
          is_occupied = true;
          break;
        }
      }
      if (is_occupied) {
        occupied++;
      }
      total++;
    }
  }

  return total > 0 ? static_cast<float>(occupied) / total : 0.0f;
}

float LocalGuide::calculate_collision_rate(const int i, const Path& path) {
  if (WINDOW_UPDATE_TYPE != WindowUpdateType::COLLISION) return 0.0f;

  int total_collisions = 0;
  int total_steps = 0;

  // パスの各ステップで衝突をチェック
  for (size_t t = 0; t < path.size(); ++t) {
    if (path[t] == nullptr) continue;
    
    // 他のエージェントとの衝突をチェック
    for (int j = 0; j < N; ++j) {
      if (j == i) continue;
      if (t < guide_paths[j].size() && guide_paths[j][t] != nullptr) {
        if (path[t] == guide_paths[j][t]) {
          total_collisions++;
        }
      }
    }
    total_steps++;
  }

  return total_steps > 0 ? static_cast<float>(total_collisions) / total_steps : 0.0f;
}

// 各判定タイプに応じたウィンドウサイズの更新処理
void LocalGuide::update_window_by_access_count(const int i) {
  // const float avg_access_count = static_cast<float>(node_access_counts[i]) / WINDOWS[i]
  // debug log
  // std::cout << "update_window_by_access_count: agent " << i << " access count is " << node_access_counts[i] << std::endl;
  const float avg_access_count = static_cast<float>(node_access_counts[i]) / WINDOWS[i] / WINDOWS[i];
  if (avg_access_count > ACCESS_COUNT_THRESHOLD) {
    WINDOWS[i] = std::min(MAX_WINDOW, WINDOWS[i] + 1);
    // std::cout << "DEBUG: access count is high (" << avg_access_count << "), increasing window to " << WINDOWS[i] << std::endl;
  } else {
    WINDOWS[i] = std::max(MIN_WINDOW, WINDOWS[i] - 1);
    // std::cout << "DEBUG: access count is low (" << avg_access_count << "), decreasing window to " << WINDOWS[i] << std::endl;
  }
  node_access_counts[i] = 0;  // アクセス回数をリセット
}

void LocalGuide::update_window_by_occupancy(const int i, const Config& Q_from) {
  const float occupancy = calculate_occupancy(i, Q_from);
  if (occupancy < OCCUPANCY_THRESHOLD) {
    WINDOWS[i] = std::max(MIN_WINDOW, WINDOWS[i] - 1);
    // std::cout << "DEBUG: occupancy is low (" << occupancy << "), decreasing window to " << WINDOWS[i] << std::endl;
  } else {
    WINDOWS[i] = std::min(MAX_WINDOW, WINDOWS[i] + 1);
    // std::cout << "DEBUG: occupancy is high (" << occupancy << "), increasing window to " << WINDOWS[i] << std::endl;
  }
}

void LocalGuide::update_window_by_collision(const int i) {
  const float collision_rate = calculate_collision_rate(i, guide_paths[i]);
  if (collision_rate < COLLISION_THRESHOLD) {
    WINDOWS[i] = std::max(MIN_WINDOW, WINDOWS[i] - 1);
    // std::cout << "DEBUG: collision rate is low (" << collision_rate << "), decreasing window to " << WINDOWS[i] << std::endl;
  } else {
    WINDOWS[i] = std::min(MAX_WINDOW, WINDOWS[i] + 1);
    // std::cout << "DEBUG: collision rate is high (" << collision_rate << "), increasing window to " << WINDOWS[i] << std::endl;
  }
}

// ウィンドウサイズの更新処理を統合
// 並列計算用のヘルパー関数：CTを読み取り専用でパス計算
Path LocalGuide::computeGuidePath(int agent_id, const Config& Q_from) {
  if (Q_from[agent_id] == nullptr) return Path();
  
  // 特別なケース：既にゴールにいる場合
  if (Q_from[agent_id] == ins->goals[agent_id]) {
    Path path(WINDOWS[agent_id], Q_from[agent_id]);
    return path;
  }
  
  // WSPPアルゴリズム用の変数（スレッドローカル、適切なサイズで初期化）
  thread_local std::vector<std::vector<WSPPNode*>> thread_CLOSED;
  thread_local WSPPNodes thread_wspp_nodes;
  thread_local int thread_wspp_node_idx = 0;
  thread_local std::vector<std::pair<int, int>> thread_CLOSED_idx;
  
  const int window_size = WINDOWS[agent_id];
  
  // 初期化（初回のみ）
  if (thread_CLOSED.empty()) {
    thread_CLOSED.resize(window_size);
    for (int t = 0; t < window_size; ++t) {
      thread_CLOSED[t].resize(V_size, nullptr);
    }
    thread_wspp_nodes.resize(V_size * window_size * 2);
    for (int i = 0; i < thread_wspp_nodes.size(); ++i) {
      thread_wspp_nodes[i] = new WSPPNode;
    }
  }
  
  auto cmp = [&](WSPPNode* a, WSPPNode* b) {
    if (a->f != b->f) return a->f > b->f;
    if (a->g != b->g) return a->g < b->g;
    return false;
  };
  
  auto get_node = [&](Vertex* where, WSPPNode* parent) {
    auto n = thread_wspp_nodes[thread_wspp_node_idx];
    thread_wspp_node_idx = (thread_wspp_node_idx + 1) % thread_wspp_nodes.size();
    
    n->when = (parent == nullptr) ? 0 : parent->when + 1;
    n->where = where;
    n->parent = parent;
    
    // g-value with collision cost (read-only CT access with self-exclusion)
    auto collision = 0;
    if (parent != nullptr) {
      collision = CT.getCollisionCost(parent->where, where, parent->when, agent_id);
    }
    n->g = (parent == nullptr) ? 0 : parent->g + 1;
    if (collision >= 1) n->g += COLLISION_COST + collision * COLLISION_COST_ORDER;
    
    // h-value  
    n->h = D->get(agent_id, where);
    n->f = n->g + n->h;
    
    return n;
  };
  
  // WSPPアルゴリズムの実行
  std::priority_queue<WSPPNode*, std::vector<WSPPNode*>, decltype(cmp)> OPEN(cmp);
  thread_CLOSED_idx.clear();
  thread_wspp_node_idx = 0;
  
  // 初期ノード
  auto n_init = get_node(Q_from[agent_id], nullptr);
  OPEN.push(n_init);
  
  Path path(window_size, nullptr);
  
  // 探索
  while (!OPEN.empty()) {
    auto n = OPEN.top();
    OPEN.pop();
    
    // check closed
    if (thread_CLOSED[n->when][n->where->id] != nullptr) continue;
    thread_CLOSED[n->when][n->where->id] = n;
    thread_CLOSED_idx.emplace_back(n->when, n->where->id);
    
    // check goal condition
    if (n->when == window_size - 1) {
      // reconstruct path
      auto temp_n = n;
      while (temp_n != nullptr) {
        path[temp_n->when] = temp_n->where;
        temp_n = temp_n->parent;
      }
      break;
    }
    
    // 早期終了: ゴールに到達した場合
    if (n->where == ins->goals[agent_id]) {
      // reconstruct path
      auto temp_n = n;
      while (temp_n != nullptr) {
        path[temp_n->when] = temp_n->where;
        temp_n = temp_n->parent;
      }
      // 残りの時間ステップをゴールで埋める
      for (int t = n->when + 1; t < window_size; ++t) {
        path[t] = ins->goals[agent_id];
      }
      break;
    }
    
    // expand neighbors
    for (auto next_v : n->where->actions) {
      auto n_next = get_node(next_v, n);
      OPEN.push(n_next);
    }
  }
  
  // cleanup
  for (auto [t, v] : thread_CLOSED_idx) {
    thread_CLOSED[t][v] = nullptr;
  }
  
  return path;
}

void LocalGuide::update_window_size(const int i, const Config& Q_from) {
  if (!DYNAMIC_WINDOW) return;

  bool window_changed = false;

  // 判定タイプに応じた更新処理を実行
  switch (WINDOW_UPDATE_TYPE) {
    case WindowUpdateType::ACCESS_COUNT:
      // std::cout << "DEBUG: update_window_by_access_count" << std::endl;
      update_window_by_access_count(i);
      // std::cout << "DEBUG: WINDOWS[i] is " << WINDOWS[i] << std::endl;
      window_changed = true;
      break;
    case WindowUpdateType::OCCUPANCY:
      update_window_by_occupancy(i, Q_from);
      window_changed = true;
      break;
    case WindowUpdateType::COLLISION:
      update_window_by_collision(i);
      window_changed = true;
      break;
  }

  // ウィンドウサイズが変更された場合、guide_pathsのサイズも更新
  if (window_changed) {
    // 現在のパスを保存
    Path current_path = guide_paths[i];
    // 新しいサイズでパスを再初期化
    guide_paths[i] = Path(WINDOWS[i], nullptr);
    
    // 可能な限り既存のパスをコピー
    for (size_t t = 0; t < std::min(current_path.size(), guide_paths[i].size()); ++t) {
      guide_paths[i][t] = current_path[t];
    }
    
    // 残りの部分は最後の有効なノードで埋める
    Vertex* last_valid_node = nullptr;
    for (auto it = current_path.rbegin(); it != current_path.rend(); ++it) {
      if (*it != nullptr) {
        last_valid_node = *it;
        break;
      }
    }
    if (last_valid_node != nullptr) {
      for (size_t t = current_path.size(); t < guide_paths[i].size(); ++t) {
        guide_paths[i][t] = last_valid_node;
      }
    }
  }
}

void LocalGuide::construct(const Config& Q_from, const std::vector<int>& order)
{
  if (!ON || NUM_REFINE <= 0) return;

  // 動的ウィンドウサイズの更新
  if (DYNAMIC_WINDOW) {
    for (int i = 0; i < N; ++i) {
      if (Q_from[i] == nullptr) {
        std::cout << "ERROR: agent " << i << " position is null in Q_from" << std::endl;
        continue;
      }
      update_window_size(i, Q_from);
    }
  }

  auto cmp = [&](WSPPNode* a, WSPPNode* b) {
    if (a->f != b->f) return a->f > b->f;
    if (a->g != b->g) return a->g < b->g;
    return false;
  };

  thread_local int wspp_node_idx = 0;  // スレッドローカル変数に変更
  auto get_node = [&](const int who, Vertex* where, WSPPNode* parent) {
    // debug log
    // std::cout << "get_node: agent " << who << " access count is " << where->access_count << std::endl;
    auto n = wspp_nodes[wspp_node_idx];
    n->when = (parent == nullptr) ? 0 : parent->when + 1;
    n->where = where;
    n->parent = parent;

    // debug log
    // std::cout << "where: accessed_by_agents[who] is " << where->accessed_by_agents[who] << std::endl;
    if (!where->accessed_by_agents[who]) {
      where->accessed_by_agents[who] = true;
      where->access_count++;
    }

    // g-value
    auto collision = 0;  // 並列処理時は衝突コスト計算をスキップ
    if (parent != nullptr) {
      collision = CT.getCollisionCost(parent->where, where, parent->when);
    }
    n->g = (parent == nullptr) ? 0 : parent->g + 1;
    if (collision >= 1) n->g += COLLISION_COST + collision * COLLISION_COST_ORDER;

    // h-value
    if (ENABLE_IMPROVED_HEURISTIC) {
      // 改善されたヒューリスティック関数: より軽量な実装
      n->h = D->get(who, where);
      
      // 現在位置の衝突状況に基づいてヒューリスティックを調整
      float collision_penalty = 0;
      int collision_count = 0;
      
      // 隣接ノードの衝突をチェック（計算効率化のため制限）
      {
        for (auto& neighbor : where->actions) {
          auto collision = CT.getCollisionCost(where, neighbor, n->when);
          if (collision >= 1) {
            collision_penalty += collision * COLLISION_COST_ORDER;
            collision_count++;
          }
        }
      }
      
      // 衝突が多い場所では、より保守的なヒューリスティックを使用
      if (collision_count > 0) {
        n->h += collision_penalty * collision_count;
      }
    } else {
      n->h = D->get(who, where);
    }
    
    // グローバルガイダンスの最適化された適用
    if (ENABLE_OPTIMIZED_GUIDANCE) {
      auto&& gg_h = global_guide->get(who, where);
      // gに適用することでヒューリスティックの精度を向上
      n->g += gg_h.first * GLOBAL_GUIDE_FIRST_ORDER + gg_h.second * GLOBAL_GUIDE_SECOND_ORDER;
    } else {
      auto&& gg_h = global_guide->get(who, where);
      // 従来通りhに適用
      n->h += gg_h.first * GLOBAL_GUIDE_FIRST_ORDER + gg_h.second * GLOBAL_GUIDE_SECOND_ORDER;
    }

    n->f = n->g + n->h;
    wspp_node_idx += 1;
    return n;
  };
  thread_local std::vector<std::pair<int, int> > CLOSED_idx;  // スレッドローカル変数に変更

  auto update_guide_path = [&](const int i) {
    if (use_sipp_) {
      // SIPPを使用してパスを計算（ウィンドウサイズ制限付き）
      guide_paths[i] = sipp_window(i, Q_from[i], ins->goals[i], D, &CT, WINDOWS[i], nullptr);

      // パスの処理
      if (guide_paths[i].empty()) {
        if (Q_from[i] == ins->goals[i]) {
          guide_paths[i] = Path(WINDOWS[i], ins->goals[i]);
          cached_collision_costs[i] = 0.0f; // ゴールにいる場合は衝突コスト0
        } else {
          guide_paths[i] = Path(WINDOWS[i], Q_from[i]);
          cached_collision_costs[i] = 0.0f; // パスが生成できない場合は衝突コスト0
        }
      } else {
        // SIPPで生成されたパスの衝突コストを計算（並列処理では簡略化）
        {
          float total_collision_cost = 0;
          for (int t = 0; t < WINDOWS[i] - 1; ++t) {
            if (guide_paths[i][t] != nullptr && guide_paths[i][t+1] != nullptr) {
              auto collision = CT.getCollisionCost(guide_paths[i][t], guide_paths[i][t+1], t);
              if (collision >= 1) {
                total_collision_cost += COLLISION_COST + collision * COLLISION_COST_ORDER;
              }
            }
          }
          cached_collision_costs[i] = total_collision_cost;
        }
      }
      // sipp_windowは既にWINDOWS[i]サイズのパスを返すので、サイズ調整は不要
    } else {
      // Use space-time A* (original implementation)
      // special case
      if (Q_from[i] == ins->goals[i]) {
        for (auto t = 0; t < WINDOWS[i]; ++t) guide_paths[i][t] = Q_from[i];
        cached_collision_costs[i] = 0.0f; // ゴールにいる場合は衝突コスト0
        return;
      }

      // initialize search utils
      std::priority_queue<WSPPNode*, WSPPNodes, decltype(cmp)> OPEN(cmp);
      CLOSED_idx.clear();
      wspp_node_idx = 0;
      // initial node
      auto n_init = get_node(i, Q_from[i], nullptr);
      OPEN.push(n_init);

      // serach
      while (!OPEN.empty()) {
        // minimum node
        auto n = OPEN.top();
        OPEN.pop();

        // check closed
        if (CLOSED[n->when][n->where->id] != nullptr) continue;
        CLOSED[n->when][n->where->id] = n;
        CLOSED_idx.emplace_back(n->when, n->where->id);

        // check goal condition
        if (n->when == WINDOWS[i] - 1) {
          // register to CT and calculate collision cost
          float total_collision_cost = 0;
          auto temp_n = n;
          while (temp_n != nullptr) {
            guide_paths[i][temp_n->when] = temp_n->where;
            if (temp_n->parent != nullptr) {
              auto collision = CT.getCollisionCost(temp_n->parent->where, temp_n->where, temp_n->parent->when);
              if (collision >= 1) {
                total_collision_cost += COLLISION_COST + collision * COLLISION_COST_ORDER;
              }
            }
            temp_n = temp_n->parent;
          }
          cached_collision_costs[i] = total_collision_cost;
          break;
        }
        
        // 早期終了条件: 目標に到達した場合
        if (ENABLE_EARLY_TERMINATION && n->where == ins->goals[i]) {
          // パスを構築して衝突コストを計算
          float total_collision_cost = 0;
          auto temp_n = n;
          while (temp_n != nullptr) {
            guide_paths[i][temp_n->when] = temp_n->where;
            if (temp_n->parent != nullptr) {
              auto collision = CT.getCollisionCost(temp_n->parent->where, temp_n->where, temp_n->parent->when);
              if (collision >= 1) {
                total_collision_cost += COLLISION_COST + collision * COLLISION_COST_ORDER;
              }
            }
            temp_n = temp_n->parent;
          }
          // 残りの時間ステップを目標で埋める
          for (int t = n->when + 1; t < WINDOWS[i]; ++t) {
            guide_paths[i][t] = ins->goals[i];
          }
          cached_collision_costs[i] = total_collision_cost;
          break;
        }

        // expand
        auto&& C = n->where->actions;
        std::shuffle(C.begin(), C.end(), MT);
        for (auto&& v : C) {
          const auto t = n->when + 1;
          if (CLOSED[t][v->id] != nullptr) continue;
          auto n_new = get_node(i, v, n);
          OPEN.push(n_new);
          // アクセス回数を累積
          if (WINDOW_UPDATE_TYPE == WindowUpdateType::ACCESS_COUNT) {
            node_access_counts[i] += v->access_count-1;
          }
        }
      }
      // clear CLOSED
      for (auto&& st : CLOSED_idx) CLOSED[st.first][st.second] = nullptr;
    }
  };

  // create initial candidate
  for (auto i = 0; i < N; ++i) {
    if (Q_from[i] != guide_paths[i][1]) continue;
    for (auto t = 0; t < WINDOWS[i] - 1; ++t) {
      guide_paths[i][t] = guide_paths[i][t + 1];
    }
    CT.enrollPath(i, guide_paths[i]);
  }

  // 参照軌道の改善
  for (auto k = 0; k < NUM_REFINE; ++k) {
    // 全頂点のアクセスカウントとアクセス状態を初期化
    for (auto& v : ins->G->V) {
      v->access_count = 0;
      std::fill(v->accessed_by_agents.begin(), v->accessed_by_agents.end(), false);
    }
    
    std::cout << "DEBUG: Refine iteration " << k << ", NUM_REFINE=" << NUM_REFINE 
              << ", ENABLE_READONLY_PARALLEL_UPDATE=" << ENABLE_READONLY_PARALLEL_UPDATE << std::endl;
    
    if (ENABLE_READONLY_PARALLEL_UPDATE) {
      // CLAUDE.mdの指示に従った実装:
      // 1. CTに対し読み取り専用でupdate_guide_pathを並列実行
      // 2. updateする前の各エージェントにおけるguide_pathを使ってclearPath(直列実行)
      // 3. updateした各エージェントにおけるguide_pathを使ってenrollPath(直列実装)
      
      // 時間計測開始
      auto total_start = std::chrono::high_resolution_clock::now();
      
      // 前のステップのguide_pathを保存
      auto setup_start = std::chrono::high_resolution_clock::now();
      std::vector<Path> old_guide_paths(N);
      for (auto _i = 0; _i < N; ++_i) {
        const auto i = order[_i];
        old_guide_paths[i] = guide_paths[i];
      }
      auto setup_end = std::chrono::high_resolution_clock::now();
      
      // Phase 1: CTに対し読み取り専用でupdate_guide_pathを並列実行
      auto thread_creation_start = std::chrono::high_resolution_clock::now();
      const int num_threads = std::min(static_cast<int>(std::thread::hardware_concurrency()), N);
      const int agents_per_thread = (N + num_threads - 1) / num_threads;
      
      std::vector<std::vector<Path>> thread_results(num_threads);
      std::vector<std::vector<int>> thread_agent_ids(num_threads);
      std::vector<std::thread> threads;
      
      for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
        threads.emplace_back([&, thread_id]() {
          const int start_idx = thread_id * agents_per_thread;
          const int end_idx = std::min(start_idx + agents_per_thread, N);
          
          thread_results[thread_id].resize(end_idx - start_idx);
          thread_agent_ids[thread_id].resize(end_idx - start_idx);
          
          for (int idx = start_idx; idx < end_idx; ++idx) {
            const int i = order[idx];
            const int local_idx = idx - start_idx;
            
            thread_agent_ids[thread_id][local_idx] = i;
            // CTを読み取り専用でアクセスしてパス計算（自分のpathは除外される）
            thread_results[thread_id][local_idx] = computeGuidePath(i, Q_from);
          }
        });
      }
      auto thread_creation_end = std::chrono::high_resolution_clock::now();
      
      // すべてのスレッドの完了を待機
      auto thread_join_start = std::chrono::high_resolution_clock::now();
      for (auto& thread : threads) {
        thread.join();
      }
      auto thread_join_end = std::chrono::high_resolution_clock::now();
      
      // Phase 2: updateする前の各エージェントにおけるguide_pathを使ってclearPath(直列実行)
      auto clear_start = std::chrono::high_resolution_clock::now();
      for (auto _i = 0; _i < N; ++_i) {
        const auto i = order[_i];
        Q_to[i] = nullptr;
        CT.clearPath(i, old_guide_paths[i]);  // 前のステップのパスを使用
      }
      auto clear_end = std::chrono::high_resolution_clock::now();
      
      // スレッド結果をguide_pathsに統合
      auto merge_start = std::chrono::high_resolution_clock::now();
      for (int thread_id = 0; thread_id < thread_results.size(); ++thread_id) {
        for (int local_idx = 0; local_idx < thread_agent_ids[thread_id].size(); ++local_idx) {
          const int i = thread_agent_ids[thread_id][local_idx];
          guide_paths[i] = thread_results[thread_id][local_idx];
        }
      }
      auto merge_end = std::chrono::high_resolution_clock::now();
      
      // Phase 3: updateした各エージェントにおけるguide_pathを使ってenrollPath(直列実装)
      auto enroll_start = std::chrono::high_resolution_clock::now();
      for (auto _i = 0; _i < N; ++_i) {
        const auto i = order[_i];
        CT.enrollPath(i, guide_paths[i]);  // 新しく計算されたパスを使用
      }
      auto enroll_end = std::chrono::high_resolution_clock::now();
      
      auto total_end = std::chrono::high_resolution_clock::now();
      
      // 時間計測結果をログ出力
      auto setup_time = std::chrono::duration_cast<std::chrono::microseconds>(setup_end - setup_start).count();
      auto thread_creation_time = std::chrono::duration_cast<std::chrono::microseconds>(thread_creation_end - thread_creation_start).count();
      auto thread_join_time = std::chrono::duration_cast<std::chrono::microseconds>(thread_join_end - thread_join_start).count();
      auto clear_time = std::chrono::duration_cast<std::chrono::microseconds>(clear_end - clear_start).count();
      auto merge_time = std::chrono::duration_cast<std::chrono::microseconds>(merge_end - merge_start).count();
      auto enroll_time = std::chrono::duration_cast<std::chrono::microseconds>(enroll_end - enroll_start).count();
      auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start).count();
      
      static int timing_log_count = 0;
      if (timing_log_count < 5) {  // 最初の5回だけログ出力
        std::cout << "=== Parallel Update Timing (iteration " << timing_log_count << ") ===" << std::endl;
        std::cout << "Setup time: " << setup_time << " μs" << std::endl;
        std::cout << "Thread creation time: " << thread_creation_time << " μs" << std::endl;
        std::cout << "Thread join time: " << thread_join_time << " μs" << std::endl;
        std::cout << "Clear paths time: " << clear_time << " μs" << std::endl;
        std::cout << "Merge results time: " << merge_time << " μs" << std::endl;
        std::cout << "Enroll paths time: " << enroll_time << " μs" << std::endl;
        std::cout << "Total parallel time: " << total_time << " μs" << std::endl;
        std::cout << "Thread overhead: " << (thread_creation_time + thread_join_time) << " μs (" 
                  << (100.0 * (thread_creation_time + thread_join_time) / total_time) << "%)" << std::endl;
        std::cout << "Num threads: " << num_threads << ", Agents per thread: " << agents_per_thread << std::endl;
        std::cout << "=========================================" << std::endl;
        timing_log_count++;
      }
    } else if (ENABLE_COLLISION_SORT) {
      // Serial collision cost sorting (original implementation)
      std::vector<std::pair<float, int>> collision_costs;
      for (auto _i = 0; _i < N; ++_i) {
        const auto i = order[_i];
        collision_costs.push_back({cached_collision_costs[i], i});
      }
      
      // Sort by collision cost (high to low)
      std::sort(collision_costs.begin(), collision_costs.end(), 
                [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                  return a.first > b.first;
                });
      
      // Process agents serially in collision cost order
      for (const auto& cost_agent : collision_costs) {
        const auto i = cost_agent.second;
        Q_to[i] = nullptr;
        CT.clearPath(i, guide_paths[i]);
        update_guide_path(i);
        CT.enrollPath(i, guide_paths[i]);
      }
    } else {
      // 元の実装：エージェントオーダーに従って処理
      for (auto _i = 0; _i < N; ++_i) {
        const auto i = order[_i];
        Q_to[i] = nullptr;
        CT.clearPath(i, guide_paths[i]);
        update_guide_path(i);
        CT.enrollPath(i, guide_paths[i]);
      }
    }
  }
  
  // post processing
  for (int i = 0; i < N; ++i) {
    Q_to[i] = guide_paths[i][1];
    CT.clearPath(i, guide_paths[i]);
  }
}

LocalHeuristic LocalGuide::get(const int i, Vertex* v)
{
  if (!ON || NUM_REFINE <= 0 || Q_to[i] == nullptr)
    return D->get(i, v);
  if (v == Q_to[i]) return 0;
  return 1;
}

// HNodeとの連携メソッドの実装
void LocalGuide::set_guide_paths(const std::vector<Path>& paths) {
  if (paths.size() != static_cast<size_t>(N)) {
    std::cerr << "ERROR: paths size mismatch in set_guide_paths" << std::endl;
    return;
  }
  guide_paths = paths;
}

std::vector<Path> LocalGuide::get_current_guide_paths() const {
  return guide_paths;
}

void LocalGuide::reconstruct_solution_paths(const std::vector<std::vector<Path>>& solution_paths) {
  guide_paths_history.clear();
  guide_paths_history = solution_paths;
  current_step = solution_paths.size();
  
  // std::cout << "Reconstructed LocalGuide solution paths with " << current_step << " steps" << std::endl;
}
