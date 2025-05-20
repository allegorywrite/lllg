#include <cassert>
#include <planner.hpp>

int main()
{
  {
    const auto scen_filename = "../assets/random-32-32-10-random-1.scen";
    const auto map_filename = "../assets/random-32-32-10.map";
    const auto ins = Instance(scen_filename, map_filename, 3);
    const auto deadline = Deadline(200);
    auto [solution, lacam] = solve(ins, 0, &deadline);
    assert(is_feasible_solution(ins, solution));
    delete lacam;
  }

  {
    const auto scen_filename = "../tests/assets/2x1.scen";
    const auto map_filename = "../tests/assets/2x1.map";
    const auto ins = Instance(scen_filename, map_filename, 2);
    auto [solution, lacam] = solve(ins, 4);
    assert(solution.empty());
    delete lacam;
  }

  return 0;
}
