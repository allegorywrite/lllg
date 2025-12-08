#include "../include/planner.hpp"
#include <chrono>

std::pair<Solution, LaCAM*> solve(const Instance &ins, int verbose, const Deadline *deadline,
               int seed)
{
  auto result = solve_with_timing(ins, verbose, deadline, seed, nullptr);
  return {result.solution, result.lacam};
}

SolveResult solve_with_timing(const Instance &ins, int verbose, const Deadline *deadline,
               int seed, std::function<void(LaCAM&)> init, DistTable *D)
{
  // Measure initial solution computation time
  auto init_start = std::chrono::high_resolution_clock::now();

  // distance table
  std::unique_ptr<DistTable> D_ptr = nullptr;
  if (D == nullptr) {
    D_ptr = std::make_unique<DistTable>(ins);
    D = D_ptr.get();
    info(3, verbose, "set distance table");
  } else {
    info(3, verbose, "use existing distance table");
  }

  // lacam
  auto lacam = new LaCAM(&ins, D, verbose, deadline, seed);
  info(3, verbose, "start lacam");
  if (init) {
    init(*lacam);
  }
  
  auto solution = lacam->solve();
  auto solution_init = solution;
  auto init_end = std::chrono::high_resolution_clock::now();
  auto comp_time_init_ms = std::chrono::duration<double, std::milli>(init_end - init_start).count();
  
  if (solution.empty() || !LNS::ON) {
    return {solution, solution_init, lacam, comp_time_init_ms};
  }

  // lns refinement
  info(1, verbose, "use lns");
  auto refiner = PLNS(&ins, D, solution, deadline, verbose);
  solution = refiner.refine();
  return {solution, solution_init, lacam, comp_time_init_ms};
}
