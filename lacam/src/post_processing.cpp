#include "../include/post_processing.hpp"

#include "../include/dist_table.hpp"

bool is_feasible_solution(const Instance &ins, const Solution &solution,
                          const int verbose)
{
  if (solution.empty()) return true;

  // check start locations
  if (!is_same_config(solution.front(), ins.starts)) {
    info(1, verbose, "invalid starts");
    return false;
  }

  // check goal locations
  if (!is_same_config(solution.back(), ins.goals)) {
    info(1, verbose, "invalid goals");
    return false;
  }

  for (size_t t = 1; t < solution.size(); ++t) {
    for (size_t i = 0; i < ins.N; ++i) {
      auto v_i_from = solution[t - 1][i];
      auto v_i_to = solution[t][i];
      // check connectivity
      if (v_i_from != v_i_to &&
          std::find(v_i_to->neighbor.begin(), v_i_to->neighbor.end(),
                    v_i_from) == v_i_to->neighbor.end()) {
        info(1, verbose, "invalid move");
        return false;
      }

      // check conflicts
      for (size_t j = i + 1; j < ins.N; ++j) {
        auto v_j_from = solution[t - 1][j];
        auto v_j_to = solution[t][j];
        // vertex conflicts
        if (v_j_to == v_i_to) {
          info(1, verbose, "vertex conflict between agent-", i, " and agent-",
               j, " at vertex-", v_i_to->id, " at timestep ", t);
          return false;
        }
        // swap conflicts
        if (v_j_to == v_i_from && v_j_from == v_i_to) {
          info(1, verbose, "edge conflict");
          return false;
        }
      }
    }
  }

  return true;
}

bool is_feasible_solution_relax_goal(const Instance &ins, const Solution &solution,
                                     const int verbose)
{
  if (solution.empty()) return true;

  // check start locations
  if (!is_same_config(solution.front(), ins.starts)) {
    info(1, verbose, "invalid starts");
    return false;
  }

  std::vector<uint8_t> visited(ins.N, 0);
  int visited_cnt = 0;
  for (size_t t = 0; t < solution.size(); ++t) {
    for (size_t i = 0; i < ins.N; ++i) {
      if (visited[i]) continue;
      if (solution[t][i]->id == ins.goals[i]->id) {
        visited[i] = 1;
        ++visited_cnt;
      }
    }
  }
  if (visited_cnt != static_cast<int>(ins.N)) {
    info(1, verbose, "invalid relaxed goals");
    return false;
  }

  for (size_t t = 1; t < solution.size(); ++t) {
    for (size_t i = 0; i < ins.N; ++i) {
      auto v_i_from = solution[t - 1][i];
      auto v_i_to = solution[t][i];
      // check connectivity
      if (v_i_from != v_i_to &&
          std::find(v_i_to->neighbor.begin(), v_i_to->neighbor.end(),
                    v_i_from) == v_i_to->neighbor.end()) {
        info(1, verbose, "invalid move");
        return false;
      }

      // check conflicts
      for (size_t j = i + 1; j < ins.N; ++j) {
        auto v_j_from = solution[t - 1][j];
        auto v_j_to = solution[t][j];
        // vertex conflicts
        if (v_j_to == v_i_to) {
          info(1, verbose, "vertex conflict between agent-", i, " and agent-",
               j, " at vertex-", v_i_to->id, " at timestep ", t);
          return false;
        }
        // swap conflicts
        if (v_j_to == v_i_from && v_j_from == v_i_to) {
          info(1, verbose, "edge conflict");
          return false;
        }
      }
    }
  }

  return true;
}

void print_stats(const int verbose, const Deadline *deadline,
                 const Instance &ins, const Solution &solution,
                 const double comp_time_ms, const char* state_label)
{
  auto ceil = [](float x) { return std::ceil(x * 100) / 100; };
  auto dist_table = DistTable(ins);
  const auto makespan = get_makespan(solution);
  const auto makespan_lb = get_makespan_lower_bound(ins, dist_table);
  const auto sum_of_costs = get_sum_of_costs(solution);
  const auto sum_of_costs_lb = get_sum_of_costs_lower_bound(ins, dist_table);
  const bool solved = !solution.empty() && is_same_config(solution.back(), ins.goals);
  const char* label = state_label != nullptr ? state_label : (solved ? "solved" : "failed");
  info(2, verbose, deadline,
       label,
       "\tmakespan: ", makespan,
       " (lb=", makespan_lb, ", ub=", ceil((float)makespan / makespan_lb), ")",
       "\tsum_of_costs: ", sum_of_costs,
       " (lb=", sum_of_costs_lb, ", ub=", ceil((float)sum_of_costs / sum_of_costs_lb), ")");
}

// for log of map_name
static const std::regex r_map_name = std::regex(R"(.+/(.+))");

void make_log(const Instance &ins, const Solution &solution,
              const std::string &output_name, const double comp_time_ms,
              const std::string &map_name, const int seed,
              const bool log_short, const LocalGuide* local_guide,
              const double comp_time_init_ms, const Solution& solution_init,
              const std::vector<Config>* lifelong_goals_history,
              const LifelongLbSpMetrics* lifelong_lb_sp_metrics,
              const int goal_change_count,
              const std::vector<std::vector<Path>>* local_guidance_history,
              const bool* override_solved)
{
  // map name
  std::smatch results;
  const auto map_recorded_name =
      (std::regex_match(map_name, results, r_map_name)) ? results[1].str()
                                                        : map_name;

  // for instance-specific values
  auto dist_table = DistTable(ins);

  // log for visualizer
  auto get_x = [&](int k) { return k % ins.G->width; };
  auto get_y = [&](int k) { return k / ins.G->width; };
  std::ofstream log;
  log.open(output_name, std::ios::out);
  log << "agents=" << ins.N << "\n";
  log << "map_file=" << map_recorded_name << "\n";
  log << "solver=planner\n";
  bool solved_flag = override_solved ? *override_solved : !solution.empty();
  log << "solved=" << solved_flag << "\n";
  log << "soc=" << get_sum_of_costs(solution) << "\n";
  log << "soc_lb=" << get_sum_of_costs_lower_bound(ins, dist_table) << "\n";
  log << "makespan=" << get_makespan(solution) << "\n";
  log << "makespan_lb=" << get_makespan_lower_bound(ins, dist_table) << "\n";
  log << "sum_of_loss=" << get_sum_of_loss(solution) << "\n";
  log << "sum_of_loss_lb=" << get_sum_of_costs_lower_bound(ins, dist_table)
      << "\n";
  log << "comp_time=" << comp_time_ms << "\n";
  if (comp_time_init_ms >= 0.0) {
    log << "comp_time_init=" << comp_time_init_ms << "\n";
  }
  if (!solution_init.empty()) {
    log << "soc_init=" << get_sum_of_costs(solution_init) << "\n";
  }
  log << "seed=" << seed << "\n";
  // Realtime-compatible metric from lifelong goals history, if available
  if (lifelong_goals_history != nullptr && !lifelong_goals_history->empty()) {
    long long total_completed_tasks_modified = 0;
    for (size_t t = 1; t < lifelong_goals_history->size(); ++t) {
      const auto& prev_goals = (*lifelong_goals_history)[t - 1];
      const auto& curr_goals = (*lifelong_goals_history)[t];
      const size_t n = std::min(prev_goals.size(), curr_goals.size());
      for (size_t i = 0; i < n; ++i) {
        if (curr_goals[i] != prev_goals[i]) ++total_completed_tasks_modified;
      }
    }
    log << "total_completed_tasks=" << total_completed_tasks_modified << "\n";
  }
  if (lifelong_lb_sp_metrics != nullptr) {
    log << "lb_sp_dist_sum=" << lifelong_lb_sp_metrics->dist_sum << "\n";
    log << "lb_sp_dist_avg_agent=" << lifelong_lb_sp_metrics->dist_avg_agent
        << "\n";
    log << "lb_sp_dist_max_agent=" << lifelong_lb_sp_metrics->dist_max_agent
        << "\n";
    log << "lb_sp_task_count=" << lifelong_lb_sp_metrics->task_count << "\n";
    log << "lb_sp_unreachable_task_count="
        << lifelong_lb_sp_metrics->unreachable_task_count << "\n";
  }
  if (log_short) return;
  log << "starts=";
  for (size_t i = 0; i < ins.N; ++i) {
    auto k = ins.starts[i]->index;
    log << "(" << get_x(k) << "," << get_y(k) << "),";
  }
  log << "\ngoals=";
  // Lifelong mode: output goals history per timestep if provided
  if (lifelong_goals_history != nullptr && !lifelong_goals_history->empty()) {
    log << "\n";
    for (size_t t = 0; t < lifelong_goals_history->size(); ++t) {
      log << t << ":";
      const auto& goals_at_t = (*lifelong_goals_history)[t];
      for (const auto& goal : goals_at_t) {
        auto k = goal->index;
        log << "(" << get_x(k) << "," << get_y(k) << "),";
      }
      log << "\n";
    }
  } else {
    for (size_t i = 0; i < ins.N; ++i) {
      auto k = ins.goals[i]->index;
      log << "(" << get_x(k) << "," << get_y(k) << "),";
    }
  }
  log << "\nsolution=\n";
  for (size_t t = 0; t < solution.size(); ++t) {
    log << t << ":";
    auto C = solution[t];
    for (auto v : C) {
      log << "(" << get_x(v->index) << "," << get_y(v->index) << "),";
    }
    log << "\n";
  }

  if (LocalGuide::ON && (local_guide != nullptr || local_guidance_history != nullptr)) {
    log << "local_guidance=\n";
    if (local_guidance_history != nullptr) {
      log << "history_size=" << local_guidance_history->size() << "\n";
      for (size_t step = 0; step < local_guidance_history->size(); ++step) {
        const auto& paths = (*local_guidance_history)[step];
        log << "step" << step << ":\n";
        for (size_t i = 0; i < ins.N; ++i) {
          log << "agent" << i << ":";
          for (size_t t = 0; t < paths[i].size(); ++t) {
            if (paths[i][t] != nullptr) {
              log << "(" << get_x(paths[i][t]->index) << ","
                  << get_y(paths[i][t]->index) << "),";
            }
          }
          log << "\n";
        }
      }
    } else if (local_guide != nullptr) {
      log << "history_size=" << local_guide->get_history_size() << "\n";
      for (int step = 0; step < local_guide->get_history_size(); ++step) {
        const auto& paths = local_guide->get_paths_at_step(step);
        log << "step" << step << ":\n";
        for (size_t i = 0; i < ins.N; ++i) {
          log << "agent" << i << ":";
          for (size_t t = 0; t < paths[i].size(); ++t) {
            if (paths[i][t] != nullptr) {
              log << "(" << get_x(paths[i][t]->index) << ","
                  << get_y(paths[i][t]->index) << "),";
            }
          }
          log << "\n";
        }
      }
    }
  }

  log.close();
}
