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

#include <cstdint>
#include <deque>
#include <memory>
#include <queue>
#include <set>
#include <unordered_map>

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

struct PathCost {
  long long primary = 0;
  long long secondary = 0;
};

struct HNode;
struct CompareHNodePointers {  // for determinism
  bool operator()(const HNode *lhs, const HNode *rhs) const;
};

// high-level search node
struct HNode {
  const Config Q;
  HNode *parent;
  int depth;
  std::set<HNode *, CompareHNodePointers> neighbor;
  PathCost g;

  std::vector<HNodePriority> priorities;
  std::vector<int> order;
  std::queue<LNode *> search_tree;

  // LocalGuideの参照軌道を保存
  std::vector<Path> local_guide_paths;

  // whether each agent has ever reached its goal up to this node
  std::vector<uint8_t> arrived_goal;
  int arrived_goal_cnt;

  HNode(Config _C, DistTable *D, HNode *_parent = nullptr,
        const std::vector<HNodePriority>* initial_priorities = nullptr);
  ~HNode();
};
using HNodes = std::vector<HNode *>;

struct ArrivedKey {
  Config Q;
  std::vector<uint8_t> arrived_goal;
};
struct ArrivedKeyHasher {
  std::size_t operator()(const ArrivedKey &k) const
  {
    std::size_t h = static_cast<std::size_t>(ConfigHasher{}(k.Q));
    for (auto b : k.arrived_goal) {
      h ^= static_cast<std::size_t>(b) + 0x9e3779b9 + (h << 6) + (h >> 2);
    }
    return h;
  }
};
struct ArrivedKeyEq {
  bool operator()(const ArrivedKey &a, const ArrivedKey &b) const
  {
    return is_same_config(a.Q, b.Q) && a.arrived_goal == b.arrived_goal;
  }
};

struct LaCAM {
  const Instance *ins;
  DistTable *D;
  const Deadline *deadline;
  const int seed;
  std::mt19937 MT;
  const int verbose;

  // solver utils
  PIBT pibt;
  std::vector<std::unique_ptr<PIBT>> extra_pibts;
  std::vector<PIBT *> pibt_pool;
  GlobalGuide global_guide;
  LocalGuide local_guide;
  int loop_cnt;
  // search state (initialized in solve)
  std::deque<HNode *> OPEN;
  std::unordered_map<Config, HNode *, ConfigHasher> EXPLORED;
  std::unordered_map<ArrivedKey, HNode *, ArrivedKeyHasher, ArrivedKeyEq>
      EXPLORED_ARRIVED;
  HNodes GC_HNodes;
  HNode *H_init = nullptr;
  HNode *H_goal = nullptr;
  // store last partial solution (backtracked from deepest explored HNode when no goal found)
  Solution last_partial_solution;
  // whether STEP_LIMIT (horizon) was reached during the last solve
  bool reached_horizon;
  // optional override for the next root priorities
  std::vector<HNodePriority> initial_root_priorities;
  bool has_initial_root_priorities = false;
  // cached priorities assigned to the last root node
  std::vector<HNodePriority> last_root_priorities;
  // cached priority sequence for the last reconstructed solution (per depth)
  std::vector<std::vector<HNodePriority>> last_solution_priorities;

  // Hyperparameters
  enum class CostMode {
    // Legacy per-step objective used by LaCAM in this repo:
    // sum_i [ moved_i OR not_at_goal_i ].
    LegacyEdge = 0,
    // Under RELAX_GOAL, counts how many agents have not yet visited their goal
    // (arrived_goal[i]==0) at the *from* node, per timestep.
    // This is equivalent to sum of first-goal-arrival times (up to an additive constant).
    UnreachedCount = 1,
    // Weighted sum: w_u * unreached_count(from) + w_m * move_count(from,to)
    WeightedSumUnreachedMove = 2,
    // Lexicographic minimization of (unreached_count(from), move_count(from,to)).
    LexiUnreachedMove = 3,
    // Count how many agents are NOT on their goal after the transition (to-state).
    // This directly prefers putting more agents onto their current goals sooner.
    NextGoalMissCount = 4,
  };

  static bool ANYTIME;
  static bool REWRITE;
  static bool REWRITE_LOG;
  static bool REWRITE_LOG_GOAL_ONLY;
  static bool REWRITE_LOG_SUMMARY;
  static int REWRITE_LOG_LEVEL;
  static int REWRITE_LOG_MAX;
  static int PIBT_NUM;
  static bool MC_USE_HEURISTIC;
  // Maximum allowed high-level depth (solution length). -1 for unlimited.
  static int STEP_LIMIT;
  // If true, terminate when every agent has reached its goal at least once
  // (not necessarily simultaneously).
  static bool RELAX_GOAL;
  static CostMode COST_MODE;
  static int COST_W_UNREACHED;
  static int COST_W_MOVE;
  // If true, compute the unreached term using the post-transition reached state.
  // This makes reaching a goal "count immediately" in the weighted_sum objective.
  static bool UNREACHED_USE_AFTER;

  // rewrite logging state (per-solve)
  long long rewrite_call_cnt = 0;
  long long rewrite_relax_cnt = 0;
  long long rewrite_relax_printed = 0;

  LaCAM(const Instance *_ins, DistTable *_D, int _verbose = 0,
        const Deadline *_deadline = nullptr, int _seed = 0);
  ~LaCAM();
  Solution solve();
  bool set_new_config(HNode *S, LNode *M, Config &Q_to);
  int get_edge_cost(const Config &from, const Config &to) const;
  int get_move_cost(const Config &from, const Config &to) const;
  int get_unreached_count(const HNode *from) const;
  int get_unreached_count_after(const HNode *from, const Config &to) const;
  int get_next_goal_miss_count(const Config &to) const;
  PathCost get_transition_cost(const HNode *from, const Config &to) const;
  PathCost get_transition_cost(const HNode *from, const HNode *to) const;
  void rewrite(HNode *H_from, HNode *H_to);
  void apply_new_solution(const Solution &plan);
  const Solution& get_last_partial_solution() const { return last_partial_solution; }
  bool was_horizon_reached() const { return reached_horizon; }
  void set_initial_priorities(const std::vector<HNodePriority>& priorities);
  const std::vector<HNodePriority>& get_last_root_priorities() const { return last_root_priorities; }
  const std::vector<std::vector<HNodePriority>>& get_last_solution_priorities() const
  {
    return last_solution_priorities;
  }
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
