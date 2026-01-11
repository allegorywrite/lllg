#include "../include/lns.hpp"

bool LNS::ON = true;
int LNS::MAX_LOOP_CNT = 100000;
int LNS::HORIZON = -1;

static void ensure_path_length(Path& path, int horizon)
{
  if (horizon < 0) return;
  if (path.empty()) return;
  const int target_size = horizon + 1;
  if ((int)path.size() > target_size) {
    path.resize(target_size);
    return;
  }
  auto last = path.back();
  while ((int)path.size() < target_size) path.push_back(last);
}

static Paths make_initial_paths(const Instance* ins, const Solution& solution, int horizon)
{
  if (solution.empty()) return Paths();
  if (horizon < 0) return translateConfigsToPaths(solution);

  const int N = (int)ins->N;
  const int H = horizon;
  const int T_in = (int)solution.size() - 1;
  const int T_use = std::min(T_in, H);

  Paths paths(N);
  for (int i = 0; i < N; ++i) {
    paths[i].reserve(H + 1);
    for (int t = 0; t <= T_use; ++t) {
      paths[i].push_back(solution[t][i]);
    }
    ensure_path_length(paths[i], H);
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

static int get_sum_of_first_goal_times(const Instance* ins, const Paths& paths, int horizon)
{
  int total = 0;
  for (int i = 0; i < (int)ins->N; ++i) {
    total += first_goal_time(paths[i], ins->goals[i], horizon);
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
      solution_paths(make_initial_paths(_ins, _solution, HORIZON)),
      cost(HORIZON < 0 ? get_sum_of_costs_paths(solution_paths)
                       : get_sum_of_first_goal_times(_ins, solution_paths, HORIZON)),
      deadline(_deadline),
      MT(std::mt19937(seed)),
      verbose(_verbose),
      N(ins->N),
      V_size(ins->G->size()),
      order(N, 0),
      CT(ins, /*_no_use_collision_cnt=*/false, /*_no_use_goal_occupation=*/HORIZON >= 0),
      loop_cnt(0)
{
  std::iota(order.begin(), order.end(), 0);
  if (HORIZON >= 0) {
    for (int i = 0; i < N; ++i) ensure_path_length(solution_paths[i], HORIZON);
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
  std::shuffle(order.begin(), order.end(), MT);

  const auto num_refine_agents =
      std::max(1, std::min(get_random_int(MT, 1, 30), int(N / 4)));
  solver_info(5, "size of modif set: ", num_refine_agents);
  for (auto k = 0; (k + 1) * num_refine_agents < N; ++k) {
    auto old_cost = 0;
    auto new_cost = 0;

    // compute old cost
    for (auto _i = 0; _i < num_refine_agents; ++_i) {
      const auto i = order[k * num_refine_agents + _i];
      old_cost += (HORIZON < 0) ? get_path_cost(solution_paths[i])
                                : first_goal_time(solution_paths[i], ins->goals[i], HORIZON);
      CT.clearPath(i, solution_paths[i]);
    }

    // re-planning
    Paths new_paths(num_refine_agents);
    for (auto _i = 0; _i < num_refine_agents; ++_i) {
      const auto i = order[k * num_refine_agents + _i];
      if (HORIZON < 0) {
        new_paths[_i] = sipp(i, ins->starts[i], ins->goals[i], D, &CT, deadline,
                             old_cost - new_cost - 1);
      } else {
        new_paths[_i] = plan_finite_horizon_reach_goal_once(
            ins, i, ins->starts[i], ins->goals[i], D, &CT, deadline, HORIZON);
      }
      if (new_paths[_i].empty()) break;  // failure
      if (HORIZON >= 0) ensure_path_length(new_paths[_i], HORIZON);
      new_cost += (HORIZON < 0) ? get_path_cost(new_paths[_i])
                                : first_goal_time(new_paths[_i], ins->goals[i], HORIZON);
      CT.enrollPath(i, new_paths[_i]);
    }

    if (!new_paths[num_refine_agents - 1].empty() && new_cost <= old_cost) {
      // success
      for (auto _i = 0; _i < num_refine_agents; ++_i) {
        const auto i = order[k * num_refine_agents + _i];
        solution_paths[i] = new_paths[_i];
      }
      cost = cost - old_cost + new_cost;
    } else {
      // failure
      for (auto _i = 0; _i < num_refine_agents; ++_i) {
        const auto i = order[k * num_refine_agents + _i];
        if (!new_paths[_i].empty()) CT.clearPath(i, new_paths[_i]);
        CT.enrollPath(i, solution_paths[i]);
      }
    }
  }

  solver_info(cost < cost_before ? 3 : 4, "cost: ", cost_before, " -> ", cost);
}
