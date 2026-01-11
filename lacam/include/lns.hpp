/*
 * Implementation of refiners
 *
 * references:
 * Iterative Refinement for Real-Time Multi-Robot Path Planning.
 * Keisuke Okumura, Yasumasa Tamura, and Xavier Défago.
 * In Proceedings of IEEE/RSJ International Conference on Intelligent Robots and
 * Systems (IROS). 2021.
 *
 * Anytime multi-agent path finding via large neighborhood search.
 * Jiaoyang Li, Zhe Chen, Daniel Harabor, P Stuckey, and Sven Koenig.
 * In Proceedings of International Joint Conference on Artificial Intelligence
 * (IJCAI). 2021.
 */

#pragma once

#include "collision_table.hpp"
#include "dist_table.hpp"
#include "graph.hpp"
#include "instance.hpp"
#include "metrics.hpp"
#include "sipp.hpp"
#include "translator.hpp"
#include "utils.hpp"

struct LNS {
  const Instance *ins;
  DistTable *D;
  Paths solution_paths;
  Solution solution;
  int cost;
  const Deadline *deadline;
  std::mt19937 MT;
  const int verbose;
  const int N;  // number of agents
  const int V_size;

  std::vector<int> order;
  CollisionTable CT;
  int loop_cnt;
  std::vector<int> intersection_vertices;

  // Hyperparametes
  enum class NeighborhoodStrategy {
    RandomBlock,   // legacy: shuffle order then take contiguous blocks
    RandomAgents,  // sample agents uniformly at random
    Intersection,  // collect agents around random intersections
    RandomWalk,    // collect agents by random-walking a delayed agent
  };
  static bool ON;
  static int MAX_LOOP_CNT;
  // If true, refine under the relaxed goal condition:
  // each agent must reach its goal at least once (not necessarily stay there).
  static bool RELAX_GOAL_CONDITION;
  // If true under RELAX_GOAL_CONDITION, prioritize having agents at their goals
  // at t=1 (lexicographically, then time-to-first-goal).
  static bool RELAX_OBJECTIVE_T1;
  static NeighborhoodStrategy NEIGHBOR_STRATEGY;
  // Neighborhood size. If < 0, use the legacy size sampling logic.
  static int NEIGHBOR_SIZE;

  LNS(const Instance *_ins, DistTable *_D, Solution &_solution,
      const Deadline *_deadline, const int seed = 0, const int _verbose = 0);
  ~LNS();
  Solution refine();
  void step();

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
