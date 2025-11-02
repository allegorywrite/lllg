#pragma once

#include "instance.hpp"

// Run Lifelong LaCAM outer loop: replan each cycle and execute one step.
// Returns 0 on success, non-zero on failure.
int run_lifelong(const Instance& base_ins,
                 int verbose,
                 float time_limit_sec,
                 int seed,
                 bool use_sipp,
                 int steps_limit,
                 const std::string& output_name,
                 const std::string& map_name,
                 bool log_short);

