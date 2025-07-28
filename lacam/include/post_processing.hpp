/*
 * post processing, e.g., calculating solution quality
 */
#pragma once
#include "dist_table.hpp"
#include "instance.hpp"
#include "local_guide.hpp" 
#include "metrics.hpp"
#include "utils.hpp"

bool is_feasible_solution(const Instance &ins, const Solution &solution,
                          const int verbose = 0);
void print_stats(const int verbose, const Deadline *deadline,
                 const Instance &ins, const Solution &solution,
                 const double comp_time_ms);
void make_log(const Instance &ins, const Solution &solution,
              const std::string &output_name, const double comp_time_ms,
              const std::string &map_name, const int seed,
              const bool log_short, const LocalGuide* local_guide = nullptr,
              const double comp_time_init_ms = -1.0, const Solution& solution_init = {});
