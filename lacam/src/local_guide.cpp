#include "../include/local_guide.hpp"
#include "../include/graph.hpp"  // get_x, get_y関数を使用するために追加
#include <iostream>  // デバッグログ用
#include <iomanip>   // デバッグログのフォーマット用
#include <thread>    // マルチスレッド用
#include <mutex>     // マルチスレッド用
#include <atomic>    // アトミック操作用
#include <chrono>    // 時間計測用

// 静的メンバー変数の定義
bool LocalGuide::ON = true;
int LocalGuide::WINDOW = 10;
int LocalGuide::NUM_REFINE = 1;
float LocalGuide::COLLISION_COST = 1.0f;
float LocalGuide::COLLISION_COST_ORDER = 1e-7;
float LocalGuide::GLOBAL_GUIDE_FIRST_ORDER = 1e-2;
float LocalGuide::GLOBAL_GUIDE_SECOND_ORDER = 1e-4;
bool LocalGuide::ENABLE_COLLISION_SORT = false;
bool LocalGuide::ENABLE_OPTIMIZED_GUIDANCE = false;
bool LocalGuide::ENABLE_K_STEP_UPDATE = false;  // k-step local guidance update (disabled by default)
int LocalGuide::K_STEP_INTERVAL = 3;  // k-step update interval (default 3)

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
      guide_paths(N, Path(WINDOW, nullptr)),
      guide_paths_history(),
      current_step(0),
      CLOSED(WINDOW, WSPPNodes(V_size, nullptr)),
      Q_to(N, nullptr),
      global_guide(gg),
      cached_collision_costs(N, 0.0f), // 衝突コストキャッシュを0で初期化
      step_counters(N, 0) // k-step用ステップカウンタを0で初期化
{
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

// k-step update 判定用のヘルパー関数
bool LocalGuide::should_update_guide_path(int agent_id) {
  if (!ENABLE_K_STEP_UPDATE) {
    return true;  // 元の実装では毎回更新
  }
  
  // 範囲チェック
  if (agent_id < 0 || agent_id >= static_cast<int>(step_counters.size())) {
    return true;  // 安全のため常に更新
  }
  
  // 初回は必ず更新（guide_pathsが未初期化の場合）
  if (step_counters[agent_id] == 0) {
    // guide_pathsが全てnullptrの場合は初回とみなして必ず更新
    bool all_null = true;
    for (size_t t = 0; t < guide_paths[agent_id].size(); ++t) {
      if (guide_paths[agent_id][t] != nullptr) {
        all_null = false;
        break;
      }
    }
    if (all_null) {
      step_counters[agent_id] = 1;  // 初回更新としてカウント
      return true;
    }
  }
  
  // k-step毎に更新する
  step_counters[agent_id]++;
  if (step_counters[agent_id] >= K_STEP_INTERVAL) {
    step_counters[agent_id] = 0;  // カウンタをリセット
    return true;
  }
  return false;
}

void LocalGuide::construct(const Config& Q_from, const std::vector<int>& order)
{
  if (!ON || NUM_REFINE <= 0) return;

  auto cmp = [&](WSPPNode* a, WSPPNode* b) {
    if (a->f != b->f) return a->f > b->f;
    if (a->g != b->g) return a->g < b->g;
    return false;
  };

  thread_local int wspp_node_idx = 0;  // スレッドローカル変数に変更
  auto get_node = [&](const int who, Vertex* where, WSPPNode* parent) {

    auto n = wspp_nodes[wspp_node_idx];
    n->when = (parent == nullptr) ? 0 : parent->when + 1;
    n->where = where;
    n->parent = parent;

    // g-value
    auto collision = 0;  // 並列処理時は衝突コスト計算をスキップ
    if (parent != nullptr) {
      collision = CT.getCollisionCost(parent->where, where, parent->when);
    }
    n->g = (parent == nullptr) ? 0 : parent->g + 1;
    if (collision >= 1) n->g += COLLISION_COST + collision * COLLISION_COST_ORDER;

    n->h = D->get(who, where);
    
    // グローバルガイダンスの最適化された適用
    if (ENABLE_OPTIMIZED_GUIDANCE) {
      auto&& gg_h = global_guide->get(who, where);
      n->g += gg_h.first * GLOBAL_GUIDE_FIRST_ORDER + gg_h.second * GLOBAL_GUIDE_SECOND_ORDER;
    } else {
      auto&& gg_h = global_guide->get(who, where);
      n->h += gg_h.first * GLOBAL_GUIDE_FIRST_ORDER + gg_h.second * GLOBAL_GUIDE_SECOND_ORDER;
    }

    n->f = n->g + n->h;
    wspp_node_idx += 1;
    return n;
  };
  thread_local std::vector<std::pair<int, int> > CLOSED_idx;  // スレッドローカル変数に変更

  auto update_guide_path = [&](const int i) {
    // if (use_sipp_) {
    //   // SIPPを使用してパスを計算（ウィンドウサイズ制限付き）
    //   if (USE_SOFT_SIPP) {
    //     guide_paths[i] = sipps_window(i, Q_from[i], ins->goals[i], D, &CT, WINDOWS[i], nullptr);
    //   } else {
    //     guide_paths[i] = sipp_window(i, Q_from[i], ins->goals[i], D, &CT, WINDOWS[i], nullptr);
    //   }

    //   // パスの処理
    //   if (guide_paths[i].empty()) {
    //     if (Q_from[i] == ins->goals[i]) {
    //       guide_paths[i] = Path(WINDOWS[i], ins->goals[i]);
    //       cached_collision_costs[i] = 0.0f; // ゴールにいる場合は衝突コスト0
    //     } else {
    //       guide_paths[i] = Path(WINDOWS[i], Q_from[i]);
    //       cached_collision_costs[i] = 0.0f; // パスが生成できない場合は衝突コスト0
    //     }
    //   } else {
    //     // SIPPで生成されたパスの衝突コストを計算（並列処理では簡略化）
    //     {
    //       float total_collision_cost = 0;
    //       for (int t = 0; t < WINDOWS[i] - 1; ++t) {
    //         if (guide_paths[i][t] != nullptr && guide_paths[i][t+1] != nullptr) {
    //           auto collision = CT.getCollisionCost(guide_paths[i][t], guide_paths[i][t+1], t);
    //           if (collision >= 1) {
    //             total_collision_cost += COLLISION_COST + collision * COLLISION_COST_ORDER;
    //           }
    //         }
    //       }
    //       cached_collision_costs[i] = total_collision_cost;
    //     }
    //   }
    // } else {

    // Use space-time A* (original implementation)
    // special case
    if (Q_from[i] == ins->goals[i]) {
      for (auto t = 0; t < WINDOW; ++t) guide_paths[i][t] = Q_from[i];
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
      if (n->when == WINDOW - 1) {
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
      // expand - DETERMINISTIC (same as parallel version)
      auto&& C = n->where->actions;
      // Remove shuffle for consistency with parallel version
      std::shuffle(C.begin(), C.end(), MT);
      for (auto&& v : C) {
        const auto t = n->when + 1;
        if (CLOSED[t][v->id] != nullptr) continue;
        auto n_new = get_node(i, v, n);
        OPEN.push(n_new);
      }
    }
    // clear CLOSED
    for (auto&& st : CLOSED_idx) CLOSED[st.first][st.second] = nullptr;
  };

  // create initial candidate
  for (auto i = 0; i < N; ++i) {
    if (guide_paths[i].size() <= 1) continue;
    if (Q_from[i] != guide_paths[i][1]) continue;
    for (auto t = 0; t < WINDOW - 1; ++t) {
      guide_paths[i][t] = guide_paths[i][t + 1];
    }
    CT.enrollPath(i, guide_paths[i]);
  }

  // 参照軌道の改善
  for (auto k = 0; k < NUM_REFINE; ++k) {
    
    if (ENABLE_COLLISION_SORT) {
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
        
        // k-step optimization: only update guide path if needed
        if (should_update_guide_path(i)) {
          update_guide_path(i);
        } 
        
        CT.enrollPath(i, guide_paths[i]);
      }
    } else {
      // 元の実装：エージェントオーダーに従って処理
      for (auto _i = 0; _i < N; ++_i) {
        const auto i = order[_i];
        Q_to[i] = nullptr;
        CT.clearPath(i, guide_paths[i]);
        
        // k-step optimization: only update guide path if needed
        if (should_update_guide_path(i)) {
          update_guide_path(i);
        } 
        
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
}
