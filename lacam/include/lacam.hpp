/*
 * Implementation of LaCAM*
 *
 * references:
 * LaCAM: Search-Based Algorithm for Quick Multi-Agent Pathfinding.
 * Keisuke Okumura.
 * Proc. AAAI Conf. on Artificial Intelligence (AAAI). 2023.
 *
 * Improving LaCAM for Scalable Eventually Optimal Multi-Agent Pathfinding.
 * Keisuke Okumura.
 * Proc. Int. Joint Conf. on Artificial Intelligence (IJCAI). 2023.
 *
 * Engineering LaCAM*: Towards Real-Time, Large-Scale, and Near-Optimal
 * Multi-Agent Pathfinding. Keisuke Okumura. Proc. Int. Conf. on Autonomous
 * Agents and Multiagent Systems. 2024.
 */
#pragma once

#include "dist_table.hpp"
#include "global_guide.hpp"
#include "graph.hpp"
#include "instance.hpp"
#include "local_guide.hpp"
#include "pibt.hpp"
#include "utils.hpp"

// low-level search node
struct LNode {
  std::vector<int> who;
  Vertices where;
  const int depth;
  LNode();
  LNode(LNode *parent, int i, Vertex *v);  // who and where
  ~LNode();
};

using HNodePriority = std::tuple<int, int, float>;

// high-level search node
struct HNode {
  const Config Q;
  HNode *parent;
  const int depth;
  int g;

  std::vector<HNodePriority> priorities;
  std::vector<int> order;
  std::queue<LNode *> search_tree;

  // LocalGuideの参照軌道を保存
  std::vector<Path> local_guide_paths;

  HNode(Config _C, DistTable *D, HNode *_parent = nullptr);
  ~HNode();
};
using HNodes = std::vector<HNode *>;

struct LaCAM {
  const Instance *ins;
  DistTable *D;
  const Deadline *deadline;
  const int seed;
  std::mt19937 MT;
  const int verbose;

  // solver utils
  PIBT pibt;
  GlobalGuide global_guide;
  LocalGuide local_guide;
  int loop_cnt;
  // store last partial solution (backtracked from deepest explored HNode when no goal found)
  Solution last_partial_solution;
  // whether STEP_LIMIT (horizon) was reached during the last solve
  bool reached_horizon;

  // Hyperparameters
  static bool ANYTIME;
  // Maximum allowed high-level depth (solution length). -1 for unlimited.
  static int STEP_LIMIT;

  LaCAM(const Instance *_ins, DistTable *_D, int _verbose = 0,
        const Deadline *_deadline = nullptr, int _seed = 0);
  ~LaCAM();
  Solution solve();
  bool set_new_config(HNode *S, LNode *M, Config &Q_to);
  const Solution& get_last_partial_solution() const { return last_partial_solution; }
  bool was_horizon_reached() const { return reached_horizon; }
  // utilities
  template <typename... Body>
  void solver_info(const int level, Body &&...body)
  {
    if (verbose < level) return;
    std::cout << "elapsed:" << std::setw(6) << elapsed_ms(deadline) << "ms"
              << "  loop_cnt:" << std::setw(8) << loop_cnt << "\t";
    info(level, verbose, (body)...);
  }
};
