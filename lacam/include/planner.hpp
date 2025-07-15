#pragma once

#include "dist_table.hpp"
#include "graph.hpp"
#include "instance.hpp"
#include "lacam.hpp"
#include "lns.hpp"
#include "planner.hpp"
#include "plns.hpp"
#include "post_processing.hpp"
#include "utils.hpp"

struct SolveResult {
  Solution solution;
  LaCAM* lacam;
  double comp_time_init_ms;  // Initial solution computation time (LocalGuide)
};

std::pair<Solution, LaCAM*> solve(const Instance &ins, const int verbose = 0,
               const Deadline *deadline = nullptr, int seed = 0, bool use_sipp = false);

SolveResult solve_with_timing(const Instance &ins, const int verbose = 0,
               const Deadline *deadline = nullptr, int seed = 0, bool use_sipp = false);

Solution solve_lacam2(const Instance &ins, const int verbose,
                      const Deadline *deadline, int seed, DistTable *D);
