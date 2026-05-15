#include "../include/local_guide.hpp"

#include <atomic>    // For atomic operations
#include <chrono>    // For time measurement
#include <iomanip>   // For debug log formatting
#include <iostream>  // For debug logging
#include <mutex>     // For multithreading
#include <thread>    // For multithreading

#include "../include/graph.hpp"

// Definition of static member variables
bool LocalGuide::ON = true;
bool LocalGuide::DETERMINISTIC = false;
bool LocalGuide::CLEAR_GOAL_FIRST = false;
int LocalGuide::WINDOW = 10;
int LocalGuide::NUM_REFINE = 1;
float LocalGuide::COLLISION_COST = 1.0f;
float LocalGuide::COLLISION_COST_ORDER = 1e-7;

// Functions for coordinate transformation
inline int get_x(int k, const Graph* G) { return k % G->width; }
inline int get_y(int k, const Graph* G) { return k / G->width; }

LocalGuide::LocalGuide(const Instance* _ins, DistTable* _D, int seed,
                       GlobalGuide* gg)
    : ins(_ins),
      MT(std::mt19937(seed)),
      N(ins->N),
      V_size(ins->G->size()),
      D(_D),
      CT(ins, true),
      guide_paths(N, Path(WINDOW, nullptr)),
      guide_paths_history(),
      current_step(0),
      CLOSED(WINDOW, WSPPNodes(V_size, nullptr)),
      Q_to(N, nullptr),
      global_guide(gg),
      cached_collision_costs(N, 0.0f),  // Initialize collision cost cache to 0
      step_counters(N, 0)  // Initialize step counters for k-step to 0
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
  // Create a copy of the current reference trajectory
  std::vector<Path> current_paths(N);
  for (int i = 0; i < N; ++i) {
    current_paths[i] = guide_paths[i];  // Copy Path
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

int LocalGuide::get_history_size() const { return guide_paths_history.size(); }

void LocalGuide::construct(const Config& Q_from, const std::vector<int>& order)
{
  if (!ON || NUM_REFINE < 0) return;

  std::vector<uint8_t> is_at_goal;
  if (CLEAR_GOAL_FIRST) {
    is_at_goal.assign(N, 0);
    for (int i = 0; i < N; ++i) {
      if (Q_from[i] != nullptr && ins->goals[i] != nullptr &&
          Q_from[i]->id == ins->goals[i]->id) {
        is_at_goal[i] = 1;
      }
    }
  }

  // Minimal debug to catch early crashes on first call
  try {
    if (N > 0 && Q_from.size() == static_cast<size_t>(N)) {
      // Validate a couple of pointers
      (void)Q_from[0];
      (void)ins->goals[0];
    }
  } catch (...) {
    std::cerr << "LocalGuide construct: invalid Q_from or goals pointers"
              << std::endl;
  }

  auto cmp = [&](WSPPNode* a, WSPPNode* b) {
    if (a->f != b->f) return a->f > b->f;
    if (a->g != b->g) return a->g < b->g;
    return false;
  };

  thread_local int wspp_node_idx = 0;  // Changed to thread-local variable
  auto get_node = [&](const int who, Vertex* where, WSPPNode* parent) {
    // Safeguard: dynamically expand node pool if needed to avoid OOB
    if (wspp_node_idx >= static_cast<int>(wspp_nodes.size())) {
      wspp_nodes.push_back(new WSPPNode());
      // Minimal debug hint; low frequency
      if (wspp_node_idx % 10000 == 0) {
        std::cerr << "LocalGuide DEBUG: expand wspp_nodes to "
                  << wspp_nodes.size() << std::endl;
      }
    }
    auto n = wspp_nodes[wspp_node_idx];
    n->when = (parent == nullptr) ? 0 : parent->when + 1;
    n->where = where;
    n->parent = parent;

    // g-value
    auto collision = 0;
    if (parent != nullptr) {
      collision = CT.getCollisionCost(parent->where, where, parent->when);
    }
    n->g = (parent == nullptr) ? 0 : parent->g + 1;
    if (collision >= 1)
      n->g += COLLISION_COST + collision * COLLISION_COST_ORDER;

    n->h = D->get(who, where);

    // auto&& gg_h = global_guide->get(who, where);
    // n->h += gg_h.first * GLOBAL_GUIDE_FIRST_ORDER + gg_h.second *
    // GLOBAL_GUIDE_SECOND_ORDER;

    n->f = n->g + n->h;
    wspp_node_idx += 1;
    return n;
  };
  thread_local std::vector<std::pair<int, int>>
      CLOSED_idx;  // Changed to thread-local variable

  auto update_guide_path = [&](const int i) {
    // Use space-time A*
    // special case
    if (Q_from[i] == ins->goals[i]) {
      for (auto t = 0; t < WINDOW; ++t) guide_paths[i][t] = Q_from[i];
      cached_collision_costs[i] = 0.0f;  // Collision cost is 0 when at goal
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

      // check goal condition or time limit
      if (n->where == ins->goals[i] || n->when == WINDOW - 1) {
        // register to CT and calculate collision cost
        float total_collision_cost = 0;
        auto temp_n = n;
        while (temp_n != nullptr) {
          guide_paths[i][temp_n->when] = temp_n->where;
          if (temp_n->parent != nullptr) {
            auto collision = CT.getCollisionCost(
                temp_n->parent->where, temp_n->where, temp_n->parent->when);
            if (collision >= 1) {
              total_collision_cost +=
                  COLLISION_COST + collision * COLLISION_COST_ORDER;
            }
          }
          temp_n = temp_n->parent;
        }
        // Fill remaining time steps with goal (when goal is reached)
        if (n->where == ins->goals[i]) {
          for (int t = n->when + 1; t < WINDOW; ++t) {
            guide_paths[i][t] = ins->goals[i];
          }
        }
        cached_collision_costs[i] = total_collision_cost;
        break;
      }
      auto&& C = n->where->actions;
      if (!DETERMINISTIC) std::shuffle(C.begin(), C.end(), MT);
      for (auto&& v : C) {
        const auto t = n->when + 1;
        if (CLOSED[t][v->id] != nullptr) continue;

        auto n_new = get_node(i, v, n);
        OPEN.push(n_new);
      }
    }

    if (guide_paths[i][0] == nullptr) {
      std::cout << "Not Supposed Error" << std::endl;
      for (auto t = 0; t < WINDOW; ++t) {
        guide_paths[i][t] = Q_from[i];
      }
      cached_collision_costs[i] = 0.0f;
    }

    // clear CLOSED
    for (auto&& st : CLOSED_idx) CLOSED[st.first][st.second] = nullptr;
  };

  // create initial candidate (skip when NUM_REFINE=0)
  // if (NUM_REFINE != 0) {
  //   for (auto i = 0; i < N; ++i) {
  //     if (guide_paths[i].size() <= 1) continue;
  //     if (Q_from[i] != guide_paths[i][1]) continue;
  //     for (auto t = 0; t < WINDOW - 1; ++t) {
  //       guide_paths[i][t] = guide_paths[i][t + 1];
  //     }
  //     CT.enrollPath(i, guide_paths[i]);
  //   }
  // }
  for (auto i = 0; i < N; ++i) {
    if (guide_paths[i].size() <= 1) continue;
    if (Q_from[i] != guide_paths[i][1]) continue;
    for (auto t = 0; t < WINDOW - 1; ++t) {
      guide_paths[i][t] = guide_paths[i][t + 1];
    }
    CT.enrollPath(i, guide_paths[i]);
  }

  // Reference trajectory improvement
  int refine_iterations = (NUM_REFINE == 0) ? 1 : NUM_REFINE;
  // int refine_iterations = NUM_REFINE;
  for (auto k = 0; k < refine_iterations; ++k) {
    if (CLEAR_GOAL_FIRST && !is_at_goal.empty()) {
      for (int i = 0; i < N; ++i) {
        if (!is_at_goal[i]) continue;
        CT.clearPath(i, guide_paths[i]);
      }
    }
    for (auto _i = 0; _i < N; ++_i) {
      const auto i = order[_i];
      if (CLEAR_GOAL_FIRST && !is_at_goal.empty() && is_at_goal[i]) continue;
      if (NUM_REFINE == 0 && guide_paths[i][0] != guide_paths[i].back())
        continue;
      Q_to[i] = nullptr;
      CT.clearPath(i, guide_paths[i]);
      update_guide_path(i);
      CT.enrollPath(i, guide_paths[i]);
    }
    if (CLEAR_GOAL_FIRST && !is_at_goal.empty()) {
      for (int i = 0; i < N; ++i) {
        if (!is_at_goal[i]) continue;
        Q_to[i] = nullptr;
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
  if (!ON || NUM_REFINE < 0 || Q_to[i] == nullptr) return D->get(i, v);
  if (v == Q_to[i]) return 0;
  // return 1;
  return D->get(i, v) + 1;
}

void LocalGuide::set_guide_paths(const std::vector<Path>& paths)
{
  if (paths.size() != static_cast<size_t>(N)) {
    std::cerr << "ERROR: paths size mismatch in set_guide_paths" << std::endl;
    return;
  }
  guide_paths = paths;
}

std::vector<Path> LocalGuide::get_current_guide_paths() const
{
  return guide_paths;
}

void LocalGuide::reconstruct_solution_paths(
    const std::vector<std::vector<Path>>& solution_paths)
{
  guide_paths_history.clear();
  guide_paths_history = solution_paths;
  current_step = solution_paths.size();
}
