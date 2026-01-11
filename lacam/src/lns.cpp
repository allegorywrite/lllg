#include "../include/lns.hpp"

bool LNS::ON = true;
int LNS::MAX_LOOP_CNT = 100000;
bool LNS::RELAX_GOAL_CONDITION = false;
bool LNS::RELAX_OBJECTIVE_T1 = false;
LNS::NeighborhoodStrategy LNS::NEIGHBOR_STRATEGY =
    LNS::NeighborhoodStrategy::RandomBlock;
int LNS::NEIGHBOR_SIZE = -1;

#include <queue>
#include <unordered_set>

static void ensure_path_length(Path& path, int horizon)
{
  if (path.empty()) return;
  const int target_size = horizon + 1;
  if ((int)path.size() > target_size) {
    path.resize(target_size);
    return;
  }
  auto last = path.back();
  while ((int)path.size() < target_size) path.push_back(last);
}

static Paths make_initial_paths_classic(const Solution& solution)
{
  if (solution.empty()) return Paths();
  return translateConfigsToPaths(solution);
}

static Paths make_initial_paths_relaxed(const Solution& solution)
{
  if (solution.empty()) return Paths();
  const int N = (int)solution.front().size();
  const int T = (int)solution.size() - 1;
  Paths paths(N);
  for (int i = 0; i < N; ++i) {
    paths[i].reserve(T + 1);
    for (int t = 0; t <= T; ++t) paths[i].push_back(solution[t][i]);
  }
  return paths;
}

static int first_goal_time(const Path& path, const Vertex* goal, int horizon)
{
  if (path.empty()) return horizon + 1;
  const int T = std::min((int)path.size() - 1, horizon);
  for (int t = 0; t <= T; ++t) {
    if (path[t]->id == goal->id) return t;
  }
  return horizon + 1;
}

static int get_path_first_goal_time_cost(const Vertex* goal, const Path& path)
{
  if (path.empty()) return 0;
  return first_goal_time(path, goal, (int)path.size() - 1);
}

static bool is_at_goal_at_t1(const Vertex* goal, const Path& path)
{
  if (goal == nullptr) return false;
  if (path.empty()) return false;
  if ((int)path.size() >= 2) return path[1]->id == goal->id;
  return path[0]->id == goal->id;
}

static int get_path_relax_cost(const Vertex* goal, const Path& path)
{
  if (!LNS::RELAX_OBJECTIVE_T1) return get_path_first_goal_time_cost(goal, path);
  // Lexicographic: (not-at-goal-at-t=1 count) first, then time-to-first-goal.
  constexpr int T1_WEIGHT = 1'000'000;
  const int t1_penalty = is_at_goal_at_t1(goal, path) ? 0 : T1_WEIGHT;
  return t1_penalty + get_path_first_goal_time_cost(goal, path);
}

static int get_sum_of_relax_costs(const Instance* ins, const Paths& paths)
{
  int total = 0;
  for (int i = 0; i < (int)ins->N; ++i) {
    total += get_path_relax_cost(ins->goals[i], paths[i]);
  }
  return total;
}

static Path plan_finite_horizon_reach_goal_once(const Instance* ins,
                                               const int agent_id,
                                               Vertex* start, Vertex* goal,
                                               DistTable* D,
                                               CollisionTable* CT,
                                               const Deadline* deadline,
                                               const int horizon)
{
  if (horizon < 0) return Path();
  if (ins == nullptr || ins->G == nullptr) return Path();
  if (start == nullptr || goal == nullptr) return Path();
  if (CT == nullptr || D == nullptr) return Path();

  const int V_size = (int)ins->G->size();
  const int H = horizon;

  auto idx_of = [&](int v_id, int t, int reached) {
    return (t * 2 + reached) * V_size + v_id;
  };

  const int INF = INT_MAX / 4;
  std::vector<int> best_g((H + 1) * 2 * V_size, INF);
  std::vector<int> parent((H + 1) * 2 * V_size, -1);

  struct OpenNode {
    int v_id;
    int t;
    uint8_t reached;
    int g;
    int f;
  };
  auto cmp = [](const OpenNode& a, const OpenNode& b) {
    if (a.f != b.f) return a.f > b.f;
    if (a.g != b.g) return a.g > b.g;
    if (a.t != b.t) return a.t > b.t;
    return a.v_id > b.v_id;
  };
  std::priority_queue<OpenNode, std::vector<OpenNode>, decltype(cmp)> open(cmp);

  const int reached0 = (start->id == goal->id) ? 1 : 0;
  const int h0 = reached0 ? 0 : D->get(agent_id, start);
  if (!reached0 && h0 > H) return Path();  // cannot reach goal within horizon

  const int start_idx = idx_of(start->id, 0, reached0);
  best_g[start_idx] = 0;
  open.push(OpenNode{start->id, 0, (uint8_t)reached0, 0, h0});

  while (!open.empty() && !is_expired(deadline)) {
    const auto cur = open.top();
    open.pop();

    const int cur_idx = idx_of(cur.v_id, cur.t, cur.reached);
    if (cur.g != best_g[cur_idx]) continue;

    if (cur.t == H && cur.reached) {
      Path path(H + 1, nullptr);
      int idx = cur_idx;
      for (int t = H; t >= 0; --t) {
        const int v_id = idx % V_size;
        path[t] = ins->G->V[v_id];
        idx = parent[idx];
      }
      return path;
    }
    if (cur.t == H) continue;

    Vertex* v = ins->G->V[cur.v_id];
    for (auto u : v->actions) {
      if (u == nullptr) continue;
      if (CT->getCollisionCost(v, u, cur.t, agent_id) != 0) continue;

      const int t2 = cur.t + 1;
      const int reached2 = cur.reached || (u->id == goal->id);

      // cost = time-to-first-goal (steps before first arrival)
      const int step_cost = cur.reached ? 0 : 1;
      const int g2 = cur.g + step_cost;

      if (!reached2) {
        const int dist = D->get(agent_id, u);
        if (dist > H - t2) continue;  // cannot reach goal within remaining steps
      }
      const int h2 = reached2 ? 0 : D->get(agent_id, u);
      const int f2 = g2 + h2;

      const int idx2 = idx_of(u->id, t2, reached2 ? 1 : 0);
      if (g2 < best_g[idx2]) {
        best_g[idx2] = g2;
        parent[idx2] = cur_idx;
        open.push(OpenNode{u->id, t2, (uint8_t)(reached2 ? 1 : 0), g2, f2});
      }
    }
  }

  return Path();
}

LNS::LNS(const Instance *_ins, DistTable *_D, Solution &_solution,
         const Deadline *_deadline, const int seed, const int _verbose)
    : ins(_ins),
      D(_D),
      solution_paths(RELAX_GOAL_CONDITION ? make_initial_paths_relaxed(_solution)
                                          : make_initial_paths_classic(_solution)),
      cost(RELAX_GOAL_CONDITION ? get_sum_of_relax_costs(_ins, solution_paths)
                                : get_sum_of_costs_paths(solution_paths)),
      deadline(_deadline),
      MT(std::mt19937(seed)),
      verbose(_verbose),
      N(ins->N),
      V_size(ins->G->size()),
      order(N, 0),
      CT(ins, /*no_use_collision_cnt=*/false,
         /*no_use_goal_occupation=*/RELAX_GOAL_CONDITION),
      loop_cnt(0)
{
  std::iota(order.begin(), order.end(), 0);
  intersection_vertices.reserve(V_size);
  for (auto v : ins->G->V) {
    if (v == nullptr) continue;
    if ((int)v->neighbor.size() > 2) intersection_vertices.push_back(v->id);
  }
  for (auto i = 0; i < N; ++i) CT.enrollPath(i, solution_paths[i]);
}

LNS::~LNS() {}

Solution LNS::refine()
{
  if (!ON) return translatePathsToConfigs(solution_paths);
  if (solution_paths[0].empty()) return Solution();
  solver_info(0, "lns begins, cost: ", cost);
  while (!is_expired(deadline) && loop_cnt < MAX_LOOP_CNT) step();
  solver_info(0, "lns ends,   cost: ", cost);
  return translatePathsToConfigs(solution_paths);
}

void LNS::step()
{
  ++loop_cnt;

  auto cost_before = cost;
  std::shuffle(order.begin(), order.end(), MT);  // used by RandomBlock

  const int legacy_nb_size =
      std::max(1, std::min(get_random_int(MT, 1, 30), int(N / 4)));
  const int nb_size = std::max(1, std::min(NEIGHBOR_SIZE > 0 ? NEIGHBOR_SIZE : legacy_nb_size, N));

  solver_info(5, "size of modif set: ", nb_size);

  auto fill_to_size = [&](std::vector<int>& agents) {
    if ((int)agents.size() >= nb_size) return;
    std::vector<int> pool(N, 0);
    std::iota(pool.begin(), pool.end(), 0);
    std::shuffle(pool.begin(), pool.end(), MT);
    std::unordered_set<int> used(agents.begin(), agents.end());
    for (int a : pool) {
      if ((int)agents.size() >= nb_size) break;
      if (used.insert(a).second) agents.push_back(a);
    }
  };

  auto unique_shuffled_prefix = [&](int size) -> std::vector<int> {
    std::vector<int> pool(N, 0);
    std::iota(pool.begin(), pool.end(), 0);
    std::shuffle(pool.begin(), pool.end(), MT);
    if (size < (int)pool.size()) pool.resize(size);
    return pool;
  };

  auto collect_agents_at_vertex = [&](int v_id, std::unordered_set<int>& out) {
    if (v_id < 0 || v_id >= V_size) return;
    const auto& time_slices = CT.body[v_id];
    if (time_slices.empty()) return;
    int t_max = (int)time_slices.size() - 1;
    while (t_max >= 0 && time_slices[t_max].empty()) --t_max;
    if (t_max <= 0) return;
    const int t0 = get_random_int(MT, 0, t_max);
    for (int delta = 0; delta <= t_max && (int)out.size() < nb_size; ++delta) {
      const int t1 = t0 - delta;
      const int t2 = t0 + delta;
      if (t1 >= 0) {
        for (int a : time_slices[t1]) {
          if ((int)out.size() >= nb_size) break;
          out.insert(a);
        }
      }
      if (t2 != t1 && t2 <= t_max) {
        for (int a : time_slices[t2]) {
          if ((int)out.size() >= nb_size) break;
          out.insert(a);
        }
      }
    }
  };

  auto make_intersection_subset = [&]() -> std::vector<int> {
    if (intersection_vertices.empty()) return unique_shuffled_prefix(nb_size);
    std::unordered_set<int> chosen;
    const int idx = get_random_int(MT, 0, (int)intersection_vertices.size() - 1);
    const int start_v = intersection_vertices[idx];
    collect_agents_at_vertex(start_v, chosen);

    if ((int)chosen.size() < nb_size) {
      std::vector<char> visited(V_size, 0);
      std::queue<int> q;
      visited[start_v] = 1;
      q.push(start_v);
      while (!q.empty() && (int)chosen.size() < nb_size) {
        const int v_id = q.front();
        q.pop();
        auto v = ins->G->V[v_id];
        if (v == nullptr) continue;
        if ((int)v->neighbor.size() > 2) collect_agents_at_vertex(v_id, chosen);
        for (auto u : v->neighbor) {
          if (u == nullptr) continue;
          if (visited[u->id]) continue;
          visited[u->id] = 1;
          q.push(u->id);
        }
      }
    }

    std::vector<int> agents(chosen.begin(), chosen.end());
    if ((int)agents.size() > nb_size) {
      std::shuffle(agents.begin(), agents.end(), MT);
      agents.resize(nb_size);
    } else {
      fill_to_size(agents);
    }
    return agents;
  };

  auto collect_conflicts_for_move = [&](int agent_id, int from_v, int to_v, int t_from,
                                       std::unordered_set<int>& out) {
    const int t_to = t_from + 1;
    if (to_v >= 0 && to_v < V_size && t_to >= 0 &&
        t_to < (int)CT.body[to_v].size()) {
      for (int a : CT.body[to_v][t_to]) {
        if (a != agent_id) out.insert(a);
      }
    }
    if (from_v >= 0 && from_v < V_size && to_v >= 0 && to_v < V_size &&
        t_to >= 0 && t_to < (int)CT.body[from_v].size() && t_from >= 0 &&
        t_from < (int)CT.body[to_v].size()) {
      std::unordered_set<int> from_next;
      from_next.reserve(CT.body[from_v][t_to].size());
      for (int a : CT.body[from_v][t_to]) {
        if (a != agent_id) from_next.insert(a);
      }
      for (int a : CT.body[to_v][t_from]) {
        if (a != agent_id && from_next.count(a)) out.insert(a);
      }
    }
  };

  auto make_randomwalk_subset =
      [&](std::unordered_set<int>& tabu) -> std::vector<int> {
    auto pick_most_delayed_agent = [&]() -> int {
      int best = -1;
      int best_delay = 0;
      for (int i = 0; i < N; ++i) {
        if (tabu.count(i)) continue;
        if (solution_paths[i].empty()) continue;
        const int arrival_t = get_path_cost(solution_paths[i]);
        const int shortest = D->get(i, solution_paths[i].front());
        const int delay = arrival_t - shortest;
        if (delay > best_delay) {
          best_delay = delay;
          best = i;
        }
      }
      if (best_delay <= 0) return -1;
      return best;
    };

    int a = pick_most_delayed_agent();
    if (a < 0) {
      tabu.clear();
      a = pick_most_delayed_agent();
    }
    if (a < 0) return unique_shuffled_prefix(nb_size);
    tabu.insert(a);

    std::unordered_set<int> chosen;
    chosen.insert(a);

    auto random_walk = [&](int agent_id, Vertex* start, int start_t, int upperbound) {
      if (start == nullptr) return;
      Vertex* loc = start;
      for (int t = start_t; t < upperbound && (int)chosen.size() < nb_size; ++t) {
        std::vector<Vertex*> next_locs = loc->actions;
        while (!next_locs.empty()) {
          const int idx = get_random_int(MT, 0, (int)next_locs.size() - 1);
          Vertex* next = next_locs[idx];
          if (next == nullptr) {
            next_locs.erase(next_locs.begin() + idx);
            continue;
          }
          const int dist = D->get(agent_id, next);
          if (t + 1 + dist < upperbound) {
            collect_conflicts_for_move(agent_id, loc->id, next->id, t, chosen);
            loc = next;
            break;
          }
          next_locs.erase(next_locs.begin() + idx);
        }
        if (next_locs.empty()) break;
      }
    };

    int upperbound = std::max(0, get_path_cost(solution_paths[a]));
    if (upperbound > 0) {
      random_walk(a, solution_paths[a].front(), 0, upperbound);
      int count = 0;
      while ((int)chosen.size() < nb_size && count < 10) {
        upperbound = std::max(0, get_path_cost(solution_paths[a]));
        if (upperbound <= 0 || solution_paths[a].empty()) break;
        const int t = get_random_int(MT, 0, upperbound);
        const int idx = std::min(t, (int)solution_paths[a].size() - 1);
        random_walk(a, solution_paths[a][idx], idx, upperbound);
        count++;
        // choose the next agent randomly
        const int pick = get_random_int(MT, 0, (int)chosen.size() - 1);
        int j = 0;
        for (int id : chosen) {
          if (j++ == pick) {
            a = id;
            break;
          }
        }
      }
    }

    std::vector<int> agents(chosen.begin(), chosen.end());
    if ((int)agents.size() > nb_size) {
      std::shuffle(agents.begin(), agents.end(), MT);
      agents.resize(nb_size);
    } else {
      fill_to_size(agents);
    }
    return agents;
  };

  auto select_subset = [&](int attempt, std::unordered_set<int>& rw_tabu) -> std::vector<int> {
    switch (NEIGHBOR_STRATEGY) {
      case NeighborhoodStrategy::RandomBlock: {
        std::vector<int> agents;
        agents.reserve(nb_size);
        const int start = attempt * nb_size;
        for (int i = 0; i < nb_size; ++i) agents.push_back(order[start + i]);
        return agents;
      }
      case NeighborhoodStrategy::RandomAgents:
        return unique_shuffled_prefix(nb_size);
      case NeighborhoodStrategy::Intersection:
        return make_intersection_subset();
      case NeighborhoodStrategy::RandomWalk:
        return make_randomwalk_subset(rw_tabu);
      default:
        return unique_shuffled_prefix(nb_size);
    }
  };

  const int num_attempts = std::max(1, (N - 1) / nb_size);
  std::unordered_set<int> rw_tabu;

  for (int attempt = 0; attempt < num_attempts; ++attempt) {
    auto subset_agents = select_subset(attempt, rw_tabu);
    if (subset_agents.empty()) continue;

    int horizon = -1;
    if (RELAX_GOAL_CONDITION) {
      for (const auto i : subset_agents)
        horizon = std::max(horizon, (int)solution_paths[i].size() - 1);
      horizon = std::max(horizon, 0);
    }

    auto old_cost = 0;
    auto new_cost = 0;

    // compute old cost
    for (const auto i : subset_agents) {
      old_cost += RELAX_GOAL_CONDITION
                      ? get_path_relax_cost(ins->goals[i], solution_paths[i])
                      : get_path_cost(solution_paths[i]);
      CT.clearPath(i, solution_paths[i]);
    }

    // re-planning
    Paths new_paths(subset_agents.size());
    for (int _i = 0; _i < (int)subset_agents.size(); ++_i) {
      const auto i = subset_agents[_i];
      if (!RELAX_GOAL_CONDITION) {
        new_paths[_i] = sipp(i, ins->starts[i], ins->goals[i], D, &CT, deadline,
                             old_cost - new_cost - 1);
      } else {
        new_paths[_i] = plan_finite_horizon_reach_goal_once(
            ins, i, ins->starts[i], ins->goals[i], D, &CT, deadline, horizon);
      }
      if (new_paths[_i].empty()) break;  // failure
      if (RELAX_GOAL_CONDITION) ensure_path_length(new_paths[_i], horizon);
      new_cost += RELAX_GOAL_CONDITION
                      ? get_path_relax_cost(ins->goals[i], new_paths[_i])
                      : get_path_cost(new_paths[_i]);
      CT.enrollPath(i, new_paths[_i]);
    }

    if (!new_paths.back().empty() && new_cost <= old_cost) {
      // success
      for (int _i = 0; _i < (int)subset_agents.size(); ++_i) {
        const auto i = subset_agents[_i];
        solution_paths[i] = new_paths[_i];
      }
      cost = cost - old_cost + new_cost;
    } else {
      // failure
      for (int _i = 0; _i < (int)subset_agents.size(); ++_i) {
        const auto i = subset_agents[_i];
        if (!new_paths[_i].empty()) CT.clearPath(i, new_paths[_i]);
        CT.enrollPath(i, solution_paths[i]);
      }
    }
  }

  solver_info(cost < cost_before ? 3 : 4, "cost: ", cost_before, " -> ", cost);
}
