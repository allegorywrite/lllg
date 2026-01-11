#include "../include/lifelong_runner.hpp"

#include "../include/planner.hpp"
#include "../include/post_processing.hpp"
#include "../include/metrics.hpp"
#include "../include/graph.hpp"
#include <chrono>
#include <random>
#include <string>
#include <utility>

int run_lifelong(const Instance& base_ins,
                 int verbose,
                 float time_limit_sec,
                 int seed,
                 int steps_limit,
                 const std::string& output_name,
                 const std::string& map_name,
                 bool log_short,
                 LifelongSeedMode seed_mode,
                 bool log_all_step)
{
  // Prepare state
  auto G = base_ins.G;
  Config current_config = base_ins.starts;
  Config current_goals = base_ins.goals;
  Solution executed_solution;
  executed_solution.push_back(current_config);
  std::vector<Config> goals_history;
  goals_history.push_back(current_goals);
  std::mt19937 rng(seed + 1000);
  // Accumulate per-executed-step LocalGuide paths (one window per agent)
  std::vector<std::vector<Path>> local_guidance_history;

  // Local copy of predefined goals to consume
  auto predefined_goals = base_ins.agent_predefined_goals;

  // Helper: generate a new random goal for agent i
  auto generate_random_goal = [&](int agent_id) -> Vertex* {
    // Check predefined goals first
    if (agent_id < (int)predefined_goals.size() && !predefined_goals[agent_id].empty()) {
      auto v = predefined_goals[agent_id].front();
      predefined_goals[agent_id].pop_front();
      return v;
    }

    std::vector<Vertex*> candidates;
    candidates.reserve(G->V.size());
    for (auto v : G->V) {
      if (v == current_config[agent_id]) continue; // avoid current position
      bool conflict = false;
      for (size_t j = 0; j < current_goals.size(); ++j) {
        if ((int)j == agent_id) continue;
        if (current_goals[j] == v) { conflict = true; break; }
      }
      if (!conflict) candidates.push_back(v);
    }
    if (candidates.empty()) return nullptr;
    std::uniform_int_distribution<size_t> dist(0, candidates.size() - 1);
    return candidates[dist(rng)];
  };

  // Initialize DistTable once
  auto D = DistTable(base_ins);

  // Outer loop over execution steps
  const int max_steps = steps_limit;
  int steps_done = 0;
  double total_comp_time_ms = 0.0;
  Solution prev_plan;  // previous cycle's full plan (for seeding guide)
  std::vector<Path> prev_local_guide_paths;  // previous cycle's LocalGuide step-0 paths
  bool prev_local_guide_paths_valid = false;
  std::vector<HNodePriority> prev_step_priorities;

  while (max_steps < 0 || steps_done < max_steps) {
    // Update goals for agents that have reached their goals
    std::vector<bool> goal_updated(base_ins.N, false);
    for (int i = 0; i < (int)base_ins.N; ++i) {
      if (current_config[i] == current_goals[i]) {
        if (auto ng = generate_random_goal(i)) {
          current_goals[i] = ng;
          goal_updated[i] = true;
          // Update DistTable for this agent
          D.update(i, ng);
        }
      }
    }

    // Build per-cycle instance (share graph)
    Instance cyc_ins(G, current_config, current_goals);
    if (!cyc_ins.is_valid(1)) break;

    // Solve current cycle via planner with optional LocalGuide seeding
    auto cyc_start = std::chrono::high_resolution_clock::now();
    auto cyc_deadline = Deadline(time_limit_sec * 1000);
    std::function<void(LaCAM&)> init_cb = nullptr;
    auto append_init_cb = [&](std::function<void(LaCAM&)> extra_cb) {
      if (!extra_cb) return;
      if (!init_cb) init_cb = std::move(extra_cb);
      else {
        auto prev_cb = init_cb;
        init_cb = [prev_cb, extra_cb = std::move(extra_cb)](LaCAM& lacam_ref) {
          prev_cb(lacam_ref);
          extra_cb(lacam_ref);
        };
      }
    };
    std::vector<Path> seed_paths; // keep alive until init_cb executes
    const bool can_seed = LocalGuide::ON && LocalGuide::WINDOW > 0 && seed_mode != LifelongSeedMode::None;
    if (can_seed) {
      if (seed_mode == LifelongSeedMode::PrevPlan && !prev_plan.empty()) {
        seed_paths.assign(cyc_ins.N, Path(LocalGuide::WINDOW, nullptr));
        for (int i = 0; i < (int)cyc_ins.N; ++i) {
          if (goal_updated[i]) continue; // Skip seeding if goal was updated
          // seed_paths[i][0] = current_config[i];
          for (int t = 0; t < LocalGuide::WINDOW; ++t) {
            if (t < (int)prev_plan.size()) seed_paths[i][t] = prev_plan[t][i];
            // else seed_paths[i][t] = current_goals[i];
            else seed_paths[i][t] = prev_plan.back()[i];
          }
        }
        append_init_cb([&seed_paths](LaCAM& lacam_ref) {
          lacam_ref.local_guide.set_guide_paths(seed_paths);
        });
      } else if (seed_mode == LifelongSeedMode::PrevLocalGuide && prev_local_guide_paths_valid) {
        seed_paths = prev_local_guide_paths;
        append_init_cb([&seed_paths](LaCAM& lacam_ref) {
          lacam_ref.local_guide.set_guide_paths(seed_paths);
        });
      }
    }
    if ((int)prev_step_priorities.size() == cyc_ins.N) {
      append_init_cb([&prev_step_priorities](LaCAM& lacam_ref) {
        lacam_ref.set_initial_priorities(prev_step_priorities);
      });
    } else if (!prev_step_priorities.empty()) {
      prev_step_priorities.clear();
    }
    // Pass existing DistTable
    auto result = solve_with_timing(cyc_ins, verbose - 1, &cyc_deadline, seed, init_cb, &D);
    auto sol = result.solution;
    auto lacam = result.lacam;
    auto cyc_end = std::chrono::high_resolution_clock::now();
    double cyc_comp_time_ms = std::chrono::duration<double, std::milli>(cyc_end - cyc_start).count();
    total_comp_time_ms += cyc_comp_time_ms;

    // Determine solved / subsolved
    // Note: in relax-goal mode, the final configuration does not need to match goals.
    const bool relax_goal = LaCAM::RELAX_GOAL;
    const bool cycle_solved =
        !sol.empty() &&
        (relax_goal ? is_feasible_solution_relax_goal(cyc_ins, sol, /*verbose=*/0)
                    : is_feasible_solution(cyc_ins, sol, /*verbose=*/0));
    bool cycle_subsolved = (!cycle_solved && lacam != nullptr && lacam->was_horizon_reached());
    // Per-cycle stats (concise) with label override
    const char* label = cycle_solved ? "solved" : (cycle_subsolved ? "subsolved" : "failed");
    print_stats(verbose, &cyc_deadline, cyc_ins, sol, cyc_comp_time_ms, label);

    // If failed (not reaching goals), dump a one-shot log for this cycle to result_stepX.txt
    // If failed (not reaching goals) or log_all_step is enabled, dump a one-shot log for this cycle to result_stepX.txt
    if (log_all_step || (!cycle_solved && !cycle_subsolved)) {
      // Build filename: <output_name without extension>_step<steps_done>.txt
      auto out_path = output_name;
      auto slash_pos = out_path.find_last_of('/');
      auto dot_pos = out_path.find_last_of('.');
      if (dot_pos == std::string::npos || (slash_pos != std::string::npos && dot_pos < slash_pos)) {
        dot_pos = out_path.size();
      }
      std::string stem = out_path.substr(0, dot_pos);
      std::string ext = out_path.substr(dot_pos);
      if (ext.empty()) ext = ".txt";
      std::string per_cycle_name = stem + "_step" + std::to_string(steps_done) + ext;
      // Backtrack partial solution from LaCAM's deepest explored node if not solved
      const auto& solution_to_log = cycle_solved ? sol : lacam->get_last_partial_solution();
      bool solved_override = cycle_solved;
      make_log(cyc_ins, solution_to_log, per_cycle_name, cyc_comp_time_ms, map_name, seed,
               /*log_short=*/false, /*local_guide=*/(lacam ? &lacam->local_guide : nullptr), /*comp_time_init_ms=*/-1.0,
               /*solution_init=*/{}, /*lifelong_goals_history=*/nullptr, /*goal_change_count=*/0,
               /*local_guidance_history=*/nullptr, /*override_solved=*/&solved_override);
    }

    // Execute exactly one step: solved -> use sol, subsolved -> use partial, failed -> stay
    int priority_index_for_next_config = -1;
    Config next_config = current_config;
    if (cycle_solved && sol.size() >= 2) {
      next_config = sol[1];
      priority_index_for_next_config = 1;
    } else if (cycle_subsolved) {
      const auto& partial = lacam->get_last_partial_solution();
      if (!partial.empty() && partial.size() >= 2) {
        next_config = partial[1];
        priority_index_for_next_config = 1;
      }
    }

    // Record local guidance for this executed step (use step-0 guidance of this cycle)
    bool updated_prev_local_guide = false;
    if (LocalGuide::ON && lacam != nullptr) {
      try {
        if (lacam->local_guide.get_history_size() > 0) {
          const auto& step0_paths = lacam->local_guide.get_paths_at_step(0);
          local_guidance_history.push_back(step0_paths);
          prev_local_guide_paths = step0_paths;
          prev_local_guide_paths_valid = true;
          updated_prev_local_guide = true;
        }
      } catch (...) {
        // ignore logging errors
      }
    }
    if (!updated_prev_local_guide) {
      prev_local_guide_paths_valid = false;
    }

    // Record goals and state at this timestep
    executed_solution.push_back(next_config);
    goals_history.push_back(current_goals);
    steps_done += 1;

    // Advance
    current_config = next_config;
    prev_plan = sol;  // remember last full plan for next cycle seeding
    if (lacam != nullptr) {
      const auto& solution_priorities = lacam->get_last_solution_priorities();
      if (priority_index_for_next_config >= 0 &&
          priority_index_for_next_config < (int)solution_priorities.size() &&
          (int)solution_priorities[priority_index_for_next_config].size() == cyc_ins.N) {
        prev_step_priorities = solution_priorities[priority_index_for_next_config];
      } else {
        prev_step_priorities.clear();
      }
    } else {
      prev_step_priorities.clear();
    }
    delete lacam;

    // Stop if no movement is possible and no steps limit specified
    if (sol.size() < 2 && max_steps < 0) break;
  }

  // Logging (skip feasibility since goals changed over time)
  // print_stats(verbose, nullptr, base_ins, executed_solution, total_comp_time_ms);
  make_log(base_ins, executed_solution, output_name, total_comp_time_ms, map_name, seed,
           log_short, nullptr, -1.0, {}, &goals_history, 0, &local_guidance_history);

  // Compute and print Lifelong summary metrics
  // total_completed_tasks: count goal reassignment events across agents over time
  long long total_completed_tasks = 0;
  if (!goals_history.empty()) {
    for (size_t t = 1; t < goals_history.size(); ++t) {
      const auto& prev = goals_history[t - 1];
      const auto& curr = goals_history[t];
      const size_t n = std::min(prev.size(), curr.size());
      for (size_t i = 0; i < n; ++i) {
        if (curr[i] != prev[i]) ++total_completed_tasks;
      }
    }
  }
  const double comp_time_s = total_comp_time_ms > 0.0 ? (total_comp_time_ms / 1000.0) : 0.0;
  const int ms = get_makespan(executed_solution);
  const double throughput_tasks = comp_time_s > 0.0 ? (static_cast<double>(total_completed_tasks) / static_cast<double>(ms)) : 0.0;
  const double throughput_makespan = comp_time_s > 0.0 ? (ms / comp_time_s) : 0.0;
  info(1, verbose,
       "Lifelong summary:\t",
       "total_completed_tasks=", total_completed_tasks,
       "\tcomp_time_ms=", total_comp_time_ms,
       "\tthroughput_tasks/s=", throughput_tasks,
       "\tthroughput_makespan/s=", throughput_makespan);
  return 0;
}
