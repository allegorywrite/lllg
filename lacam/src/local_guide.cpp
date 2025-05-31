#include "../include/local_guide.hpp"
#include "../include/graph.hpp"  // get_x, get_y関数を使用するために追加
#include <iostream>  // デバッグログ用
#include <iomanip>   // デバッグログのフォーマット用

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
      node_access_counts(N, 0) // アクセス回数を0で初期化
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

  int wspp_node_idx = 0;
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
    auto collision =
        (parent == nullptr)
            ? 0
            : CT.getCollisionCost(parent->where, where, parent->when);
    n->g = (parent == nullptr) ? 0 : parent->g + 1;
    if (collision >= 1) n->g += COLLISION_COST + collision * COLLISION_COST_ORDER;

    // h-value
    n->h = D->get(who, where);
    auto&& gg_h = global_guide->get(who, where);
    // n->h += gg_h.first * GLOBAL_GUIDE_FIRST_ORDER + gg_h.second * GLOBAL_GUIDE_SECOND_ORDER;
    n->g += gg_h.first * GLOBAL_GUIDE_FIRST_ORDER + gg_h.second * GLOBAL_GUIDE_SECOND_ORDER;

    n->f = n->g + n->h;
    wspp_node_idx += 1;
    return n;
  };
  std::vector<std::pair<int, int> > CLOSED_idx;

  auto update_guide_path = [&](const int i) {
    if (use_sipp_) {
      // SIPPを使用してパスを計算（ウィンドウサイズ制限付き）
      guide_paths[i] = sipp_window(i, Q_from[i], ins->goals[i], D, &CT, WINDOWS[i], nullptr);

      // パスの処理
      if (guide_paths[i].empty()) {
        if (Q_from[i] == ins->goals[i]) {
          guide_paths[i] = Path(WINDOWS[i], ins->goals[i]);
        } else {
          guide_paths[i] = Path(WINDOWS[i], Q_from[i]);
        }
      }
      // sipp_windowは既にWINDOWS[i]サイズのパスを返すので、サイズ調整は不要
    } else {
      // Use space-time A* (original implementation)
      // special case
      if (Q_from[i] == ins->goals[i]) {
        for (auto t = 0; t < WINDOWS[i]; ++t) guide_paths[i][t] = Q_from[i];
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
          // register to CT
          while (n != nullptr) {
            guide_paths[i][n->when] = n->where;
            n = n->parent;
          }
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
    
    for (auto _i = 0; _i < N; ++_i) {
      const auto i = order[_i];
      Q_to[i] = nullptr;
      CT.clearPath(i, guide_paths[i]);
      update_guide_path(i);
      CT.enrollPath(i, guide_paths[i]);
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
