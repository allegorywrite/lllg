#include "../include/planner.hpp"
#include <chrono>

std::pair<Solution, LaCAM*> solve(const Instance &ins, int verbose, const Deadline *deadline,
               int seed, bool use_sipp)
{
  auto result = solve_with_timing(ins, verbose, deadline, seed, use_sipp);
  return {result.solution, result.lacam};
}

SolveResult solve_with_timing(const Instance &ins, int verbose, const Deadline *deadline,
               int seed, bool use_sipp)
{
  // Measure initial solution computation time
  auto init_start = std::chrono::high_resolution_clock::now();

  // distance table
  auto D = DistTable(ins);
  info(1, verbose, "set distance table");

  // lacam
  auto lacam = new LaCAM(&ins, &D, verbose, deadline, seed, use_sipp);
  info(1, verbose, "start lacam");
  
  auto solution = lacam->solve();
  auto init_end = std::chrono::high_resolution_clock::now();
  auto comp_time_init_ms = std::chrono::duration<double, std::milli>(init_end - init_start).count();
  
  if (solution.empty() || !LNS::ON) {
    return {solution, lacam, comp_time_init_ms};
  }

  // lns refinement
  info(1, verbose, "use lns");
  auto refiner = PLNS(&ins, &D, solution, deadline, verbose);
  solution = refiner.refine();
  return {solution, lacam, comp_time_init_ms};
}
