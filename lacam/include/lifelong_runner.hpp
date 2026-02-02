#pragma once

#include "instance.hpp"

// Lifelong initialization source for the LocalGuide
enum class LifelongSeedMode {
  None,
  PrevPlan,
  PrevLocalGuide,
};

// Run Lifelong LaCAM outer loop: replan each cycle and execute one step.
// Returns 0 on success, non-zero on failure.
int run_lifelong(const Instance& base_ins,
                 int verbose,
                 float time_limit_sec,
                 int seed,
                 int steps_limit,
                 const std::string& output_name,
                 const std::string& map_name,
                 bool log_short,
                 LifelongSeedMode seed_mode,
                 bool log_all_step = false,
                 bool log_local_guidance = false);
