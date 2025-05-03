/*
 * implementation of PIBT
 *
 * references:
 * Priority Inheritance with Backtracking for Iterative Multi-agent Path
 * Finding. Keisuke Okumura, Manao Machida, Xavier Défago & Yasumasa Tamura.
 * Artificial Intelligence (AIJ). 2022.
 *
 * swap:
 * Improving lacam for scalable eventually optimal multi-agent pathfinding.
 * Keisuke Okumura.
 * IJCAI. 2023.
 */
#pragma once
#include "dist_table.hpp"
#include "global_guide.hpp"
#include "graph.hpp"
#include "instance.hpp"
#include "local_guide.hpp"
#include "utils.hpp"

// lg, tie
using PIBTHeuristic = std::tuple<int, float>;

struct PIBT {
  const Instance *ins;
  std::mt19937 MT;
  std::uniform_real_distribution<float> rrd;  // random, real distribution

  // solver utils
  const int N;  // number of agents
  const int V_size;
  DistTable *D;

  // specific to PIBT
  const int NO_AGENT;
  std::vector<int> occupied_now;                 // for quick collision checking
  std::vector<int> occupied_next;                // for quick collision checking
  std::vector<std::array<Vertex *, 5> > C_next;  // next location candidates
  std::array<PIBTHeuristic, 5> C_cost;
  std::vector<std::array<int, 5> > C_indices;
  std::array<int, 4> neighbor_agents;

  // hyper parameters
  static bool SWAP;

  // guidance
  GlobalGuide *global_guide;
  LocalGuide *local_guide;

  PIBT(const Instance *_ins, DistTable *_D, int seed = 0);
  ~PIBT();

  bool set_new_config(const Config &Q_from, Config &Q_to,
                      const std::vector<int> &order);
  bool funcPIBT(const int i, const Config &Q_from, Config &Q_to);

  int is_swap_required_and_possible(const int ai, const Config &Q_from,
                                    Config &Q_to, Vertex *v_i_target);
  bool is_swap_required(const int pusher, const int puller,
                        Vertex *v_pusher_origin, Vertex *v_puller_origin);
  bool is_swap_possible(Vertex *v_pusher_origin, Vertex *v_puller_origin);
};
