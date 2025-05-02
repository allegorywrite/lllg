/*
 * Implementation of SUO
 *
 * references:
 * Optimizingspaceutilizationformoreeffective multi-robot path planning.
 * Shuai D Han and Jingjin Yu.
 * In Proceedings of IEEE International Conference on Robotics and Automation
 * (ICRA). 2022.
 */
#pragma once

#include "collision_table.hpp"
#include "dist_table.hpp"
#include "graph.hpp"
#include "metrics.hpp"
#include "utils.hpp"

using GGHeuristic = std::pair<int, int>;

struct GlobalGuide {
  const Instance *ins;
  Deadline deadline;
  std::mt19937 MT;
  const int N;
  const int V_size;
  const int T;  // makespan lower bound
  DistTable *D;

  // outcome
  std::vector<Path> paths;

  std::vector<std::queue<Vertex *>> OPEN;             // search queue
  std::vector<std::vector<GGHeuristic>> guide_table;  // who, where -> value

  // Hyperparametes
  static bool ON;
  static int COST_MARGIN;

  void construct();
  void _construct();
  void run_suo();

  GlobalGuide(const Instance *_ins, DistTable *_D, const Deadline *_deadline,
              const int seed = 0);
  ~GlobalGuide();

  GGHeuristic get(const int i, Vertex *v);
};
