#include "../include/plns.hpp"

int PLNS::NUM_REFINERS = 8;

constexpr auto TIME_WAIT = std::chrono::milliseconds(0);

static int get_sum_of_first_goal_times(const Instance *ins,
                                       const Solution &solution)
{
  if (ins == nullptr) return 0;
  if (solution.empty()) return 0;

  const int T = (int)solution.size() - 1;
  int total = 0;
  for (int i = 0; i < (int)ins->N; ++i) {
    int first_t = T + 1;
    const int goal_id = ins->goals[i]->id;
    for (int t = 0; t <= T; ++t) {
      if (solution[t][i]->id == goal_id) {
        first_t = t;
        break;
      }
    }
    total += first_t;
  }
  return total;
}

static int get_sum_of_relax_cost(const Instance *ins, const Solution &solution)
{
  if (!LNS::RELAX_OBJECTIVE_T1)
    return get_sum_of_first_goal_times(ins, solution);
  if (ins == nullptr) return 0;
  if (solution.empty()) return 0;

  constexpr int T1_WEIGHT = 1'000'000;
  const int T = (int)solution.size() - 1;
  const bool has_t1 = (int)solution.size() >= 2;

  int total = 0;
  for (int i = 0; i < (int)ins->N; ++i) {
    int first_t = T + 1;
    const int goal_id = ins->goals[i]->id;
    for (int t = 0; t <= T; ++t) {
      if (solution[t][i]->id == goal_id) {
        first_t = t;
        break;
      }
    }
    const bool at_t1 = has_t1 ? (solution[1][i]->id == goal_id)
                              : (solution[0][i]->id == goal_id);
    total += (at_t1 ? 0 : T1_WEIGHT) + first_t;
  }
  return total;
}

PLNS::PLNS(const Instance *_ins, DistTable *_D, Solution &_solution,
           const Deadline *_deadline, const int _verbose, const int seed)
    : ins(_ins),
      D(_D),
      deadline(_deadline),
      verbose(_verbose),
      MT(seed),
      solution(_solution),
      iteration(0),
      cost_best(LNS::RELAX_GOAL_CONDITION ? get_sum_of_relax_cost(ins, solution)
                                          : get_sum_of_costs(solution))
{
}

PLNS::~PLNS() {}

Solution PLNS::refine()
{
  if (!LNS::ON) return solution;
  solver_info(1, "plns refine starts, cost: ", cost_best);

  // set refiner
  LNS::MAX_LOOP_CNT = 1;

  auto step = [&](auto &&sol, const int s) {
    auto refiner = LNS(ins, D, sol, deadline, s, -1);
    return refiner.refine();
  };

  // initialize
  std::list<std::future<Solution> > pool;
  for (auto k = 0; k < NUM_REFINERS; ++k) {
    pool.emplace_back(
        std::async(std::launch::async, step, solution, ++iteration));
  }

  while (!is_expired(deadline)) {
    pool.remove_if([&](auto &proc) {
      if (is_expired(deadline)) return true;
      if (proc.wait_for(TIME_WAIT) != std::future_status::ready) return false;
      auto solution_new = proc.get();
      auto cost = LNS::RELAX_GOAL_CONDITION
                      ? get_sum_of_relax_cost(ins, solution_new)
                      : get_sum_of_costs(solution_new);
      if (cost < cost_best) {
        solver_info(3, "cost update: ", cost_best, " -> ", cost);
        solution = solution_new;
        cost_best = cost;
      }
      pool.emplace_back(
          std::async(std::launch::async, step, solution, ++iteration));
      return true;
    });
  }

  solver_info(1, "plns refine ends, cost: ", cost_best);
  return solution;
}
