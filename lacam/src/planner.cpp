#include "../include/planner.hpp"

std::pair<Solution, LaCAM*> solve(const Instance &ins, int verbose, const Deadline *deadline,
               int seed)
{
  // distance table
  auto D = DistTable(ins);
  info(1, verbose, "set distance table");

  // lacam
  auto lacam = new LaCAM(&ins, &D, verbose, deadline, seed);
  info(1, verbose, "start lacam");
  auto solution = lacam->solve();
  if (solution.empty() || !LNS::ON) return {solution, lacam};

  // lns refinement
  info(1, verbose, "use lns");
  auto refiner = PLNS(&ins, &D, solution, deadline, verbose);
  solution = refiner.refine();
  return {solution, lacam};
}
