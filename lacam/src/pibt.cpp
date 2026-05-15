#include "../include/pibt.hpp"

bool PIBT::SWAP = true;
bool PIBT::DETERMINISTIC = false;
bool PIBT::NEXT_STEP_HINDRANCE = true;
int PIBT::NUM_REGRET_SAMPLING = 0;
float PIBT::NEW_REGRET_WEIGHT = 0.9f;
int PIBT::SWITCH_ORDER = 0;

PIBT::PIBT(const Instance *_ins, DistTable *_D, int seed)
    : ins(_ins),
      MT(std::mt19937(seed)),
      rrd(0, 1),
      N(ins->N),
      V_size(ins->G->size()),
      D(_D),
      NO_AGENT(N),
      occupied_now(V_size, NO_AGENT),
      occupied_next(V_size, NO_AGENT),
      C_next(N),
      C_indices(N),
      R(N),
      global_guide(nullptr),
      local_guide(nullptr)
{
}

PIBT::~PIBT() {}

bool PIBT::set_new_config(const Config &Q_from, Config &Q_to,
                          const std::vector<int> &order)
{
  bool success = true;
  std::vector<int> free_agents;
  // setup cache & constraints check
  for (auto i = 0; i < N; ++i) {
    // set occupied now
    occupied_now[Q_from[i]->id] = i;

    // set occupied next
    if (Q_to[i] != nullptr) {
      // vertex collision
      if (occupied_next[Q_to[i]->id] != NO_AGENT) {
        success = false;
        break;
      }
      // swap collision
      auto j = occupied_now[Q_to[i]->id];
      if (j != NO_AGENT && j != i && Q_to[j] == Q_from[i]) {
        success = false;
        break;
      }
      occupied_next[Q_to[i]->id] = i;
    } else {
      free_agents.push_back(i);
    }

    // reset regret table (only indices that can be used)
    const size_t Kp1 = Q_from[i]->neighbor.size() + 1;
    for (size_t k = 0; k < Kp1 && k < R[i].size(); ++k) R[i][k] = 0.0f;
  }

  if (success) {
    const int samples = std::max(0, NUM_REGRET_SAMPLING);
    for (int s = 0; s < samples + 1; ++s) {
      for (auto i : order) {
        if (Q_to[i] == nullptr && !std::get<0>(funcPIBT(i, Q_from, Q_to))) {
          success = false;
          break;
        }
      }

      // Clear decisions for originally-free agents to resample.
      if (s < samples) {
        for (auto j : free_agents) {
          if (Q_to[j] != nullptr) {
            occupied_next[Q_to[j]->id] = NO_AGENT;
            Q_to[j] = nullptr;
          }
        }
      }

      if (!success) break;
    }
  }

  // cleanup
  for (auto i = 0; i < N; ++i) {
    occupied_now[Q_from[i]->id] = NO_AGENT;
    if (Q_to[i] != nullptr) occupied_next[Q_to[i]->id] = NO_AGENT;
  }

  return success;
}

std::tuple<bool, int> PIBT::funcPIBT(const int i, const Config &Q_from,
                                     Config &Q_to)
{
  // if (DETERMINISTIC) {
  //   std::cout << "funcPIBT called for agent " << i
  //             << " at (" << Q_from[i]->x << "," << Q_from[i]->y << ")"
  //             << " goal=(" << ins->goals[i]->x << "," << ins->goals[i]->y <<
  //             ")" << std::endl;
  // }
  const auto K = Q_from[i]->neighbor.size();

  const bool use_hindrance =
      NEXT_STEP_HINDRANCE && (SWITCH_ORDER == 2 || SWITCH_ORDER == 3);
  int neighbor_agent_idx = 0;
  if (use_hindrance) {
    for (auto u : Q_from[i]->neighbor) {
      if (occupied_now[u->id] != NO_AGENT) {
        neighbor_agents[neighbor_agent_idx] = occupied_now[u->id];
        neighbor_agent_idx += 1;
      }
    }
  }

  auto get_successor_cost = [&](Vertex *u, int u_idx, bool swap = false) {
    auto lg = (local_guide != nullptr) ? local_guide->get(i, u) : D->get(i, u);

    // AAMAS-24 version
    if (!LocalGuide::ON && GlobalGuide::ON) {
      auto &&gg = (global_guide != nullptr) ? global_guide->get(i, u)
                                            : std::make_pair(0, 0);
      auto &&gg_original = global_guide->get(i, Q_from[i]);
      if (gg_original.first == 0 && gg.first == 0 &&
          gg_original.second > gg.second)
        lg = 0;
    }

    float next_step_hindrance = 0.0f;
    if (use_hindrance) {
      for (int k = 0; k < neighbor_agent_idx; ++k) {
        const auto j = neighbor_agents[k];
        if (Q_from[j] != u && D->get(j, u) < D->get(j, Q_from[j]))
          next_step_hindrance += 1.0f;
      }
    }

    float inheri = 0.0f;
    if (occupied_now[u->id] != NO_AGENT) inheri = 1.0f;

    float tie_breaker = DETERMINISTIC ? 0.0f : rrd(MT);
    const int primary_cost = swap ? -D->get(i, u) : lg;
    const float regret_est =
        (u_idx >= 0 && u_idx < static_cast<int>(R[i].size())) ? R[i][u_idx]
                                                              : 0.0f;

    // Switchable tie-breaking (ported from pibt-tiebreaking).
    // 0: legacy (primary, random)
    // 1: prefer unoccupied (primary, inheri, random)
    // 2: prefer low hindrance + regret (primary, hindrance, regret, random)
    // 3: prefer regret + hindrance (primary, regret, hindrance, random)
    if (SWITCH_ORDER == 1) {
      return std::make_tuple(primary_cost, inheri, tie_breaker, 0.0f);
    }
    if (SWITCH_ORDER == 2) {
      return std::make_tuple(primary_cost, next_step_hindrance, regret_est,
                             tie_breaker);
    }
    if (SWITCH_ORDER == 3) {
      return std::make_tuple(primary_cost, regret_est, next_step_hindrance,
                             tie_breaker);
    }
    return std::make_tuple(primary_cost, tie_breaker, 0.0f, 0.0f);
  };

  // set C_next
  for (size_t k = 0; k < K; ++k) {
    auto u = Q_from[i]->neighbor[k];
    C_next[i][k] = u;
    C_cost[k] = get_successor_cost(u, k);
  }
  // stay-in-place action
  C_next[i][K] = Q_from[i];
  C_cost[K] = get_successor_cost(Q_from[i], K);
  std::iota(C_indices[i].begin(), C_indices[i].begin() + K + 1, 0);

  // sort, note: K + 1 is sufficient
  std::sort(C_indices[i].begin(), C_indices[i].begin() + K + 1,
            [&](const int k, const int l) { return C_cost[k] < C_cost[l]; });

  // emulate swap
  const auto swap_agent = is_swap_required_and_possible(
      i, Q_from, Q_to, C_next[i][C_indices[i][0]]);
  if (swap_agent != NO_AGENT) {
    // recompute action cost
    for (size_t k = 0; k < K + 1; ++k) {
      C_cost[k] = get_successor_cost(C_next[i][k], k, true);
      C_indices[i][k] = k;
    }
    std::sort(C_indices[i].begin(), C_indices[i].begin() + K + 1,
              [&](const int k, const int l) { return C_cost[k] < C_cost[l]; });
  }
  auto swap_operation = [&]() {
    if (swap_agent != NO_AGENT &&                 // swap_agent exists
        Q_to[swap_agent] == nullptr &&            // not decided
        occupied_next[Q_from[i]->id] == NO_AGENT  // free
    ) {
      // pull swap_agent
      occupied_next[Q_from[i]->id] = swap_agent;
      Q_to[swap_agent] = Q_from[i];
    }
  };

  // regret baseline: best (minimum) distance-to-goal among all candidates
  int dist_best = INT_MAX;
  for (size_t kk = 0; kk < K; ++kk) {
    dist_best = std::min(dist_best, D->get(i, C_next[i][kk]));
  }
  dist_best = std::min(dist_best, D->get(i, Q_from[i]));  // stay

  // main loop
  // if (DETERMINISTIC && i == 4) {  // Debug agent 5 (index 4)
  //   std::cout << "Agent " << i << " at " << Q_from[i]->index
  //             << " (" << Q_from[i]->x << "," << Q_from[i]->y << ")"
  //             << " goal=" << ins->goals[i]->index
  //             << " (" << ins->goals[i]->x << "," << ins->goals[i]->y << ")"
  //             << " dist=" << D->get(i, ins->goals[i]) << std::endl;
  //   std::cout << "  Neighbors sorted by cost:" << std::endl;
  //   for (size_t k_debug = 0; k_debug < K + 1; ++k_debug) {
  //     auto u_idx_debug = C_indices[i][k_debug];
  //     auto u_debug = C_next[i][u_idx_debug];
  //     auto cost = C_cost[u_idx_debug];
  //     std::cout << "    [" << k_debug << "] vertex=" << u_debug->index
  //               << " (" << u_debug->x << "," << u_debug->y << ")"
  //               << " lg=" << std::get<0>(cost)
  //               << " tie=" << std::get<1>(cost)
  //               << " dist_to_goal=" << D->get(i, u_debug) << std::endl;
  //   }
  // }
  for (size_t k = 0; k < K + 1; ++k) {
    auto u_idx = C_indices[i][k];
    auto u = C_next[i][u_idx];

    // avoid vertex conflicts
    if (occupied_next[u->id] != NO_AGENT) {
      // if (DETERMINISTIC && i == 4) {
      //   std::cout << "    k=" << k << " vertex=" << u->index << " (" << u->x
      //   << "," << u->y
      //             << ") REJECTED: occupied_next by agent " <<
      //             occupied_next[u->id] << std::endl;
      // }
      continue;
    }

    const auto j = occupied_now[u->id];

    // avoid swap conflicts with constraints
    if (j != NO_AGENT && Q_to[j] == Q_from[i]) continue;

    // reserve next location
    occupied_next[u->id] = i;
    Q_to[i] = u;

    // priority inheritance
    int regret = 0;
    if (j != NO_AGENT && u != Q_from[i] && Q_to[j] == nullptr) {
      auto res = funcPIBT(j, Q_from, Q_to);
      const bool validity = std::get<0>(res);
      regret = std::get<1>(res);
      if (u_idx >= 0 && u_idx < static_cast<int>(R[i].size())) {
        R[i][u_idx] = (1.0f - NEW_REGRET_WEIGHT) * R[i][u_idx] +
                      NEW_REGRET_WEIGHT * static_cast<float>(regret);
      }
      if (!validity) continue;
    }
    // if (j != NO_AGENT && u != Q_from[i] && Q_to[j] != nullptr) {
    //   if (DETERMINISTIC) {
    //     std::cout << "  Priority inheritance SUCCESS: agent " << i << "
    //     pushed agent " << j
    //               << " from (" << Q_from[j]->x << "," << Q_from[j]->y << ")"
    //               << " to (" << Q_to[j]->x << "," << Q_to[j]->y << ")" <<
    //               std::endl;
    //   }
    // }

    // success to plan next one step
    if (k == 0) swap_operation();
    // if (DETERMINISTIC && i == 4) {
    //   std::cout << "  -> SELECTED: vertex=" << u->index
    //             << " (" << u->x << "," << u->y << ")"
    //             << " at priority k=" << k << std::endl;
    // }
    const int regret_i = D->get(i, u) - dist_best;
    return {true, regret + regret_i};
  }

  // failed to secure node
  occupied_next[Q_from[i]->id] = i;
  Q_to[i] = Q_from[i];
  const int regret_i = D->get(i, Q_to[i]) - dist_best;
  return {false, regret_i};
}

int PIBT::is_swap_required_and_possible(const int i, const Config &Q_from,
                                        Config &Q_to, Vertex *v_i_target)
{
  if (!SWAP) return NO_AGENT;
  // agent-j occupying the desired vertex for agent-i
  const auto j = occupied_now[v_i_target->id];
  if (j != NO_AGENT && j != i &&  // j exists
      Q_to[j] == nullptr &&       // j does not decide next location
      is_swap_required(i, j, Q_from[i], Q_from[j]) &&  // swap required
      is_swap_possible(Q_from[j], Q_from[i])           // swap possible
  ) {
    return j;
  }

  // for clear operation, c.f., push & swap
  if (v_i_target != Q_from[i]) {
    for (auto u : Q_from[i]->neighbor) {
      const auto k = occupied_now[u->id];
      if (k != NO_AGENT &&            // k exists
          v_i_target != Q_from[k] &&  // this is for clear operation
          is_swap_required(k, i, Q_from[i],
                           v_i_target) &&  // emulating from one step ahead
          is_swap_possible(v_i_target, Q_from[i])) {
        return k;
      }
    }
  }
  return NO_AGENT;
}

bool PIBT::is_swap_required(const int pusher, const int puller,
                            Vertex *v_pusher_origin, Vertex *v_puller_origin)
{
  auto v_pusher = v_pusher_origin;
  auto v_puller = v_puller_origin;
  Vertex *tmp = nullptr;
  while (D->get(pusher, v_puller) < D->get(pusher, v_pusher)) {
    auto n = v_puller->neighbor.size();
    // remove agents who need not to move
    for (auto u : v_puller->neighbor) {
      const auto i = occupied_now[u->id];
      if (u == v_pusher ||
          (u->neighbor.size() == 1 && i != NO_AGENT && ins->goals[i] == u)) {
        --n;
      } else {
        tmp = u;
      }
    }
    if (n >= 2) return false;  // able to swap at v_l
    if (n <= 0) break;
    v_pusher = v_puller;
    v_puller = tmp;
  }

  return (D->get(puller, v_pusher) < D->get(puller, v_puller)) &&
         (D->get(pusher, v_pusher) == 0 ||
          D->get(pusher, v_puller) < D->get(pusher, v_pusher));
}

bool PIBT::is_swap_possible(Vertex *v_pusher_origin, Vertex *v_puller_origin)
{
  // simulate pull
  auto v_pusher = v_pusher_origin;
  auto v_puller = v_puller_origin;
  Vertex *tmp = nullptr;
  while (v_puller != v_pusher_origin) {  // avoid loop
    auto n = v_puller->neighbor.size();
    for (auto u : v_puller->neighbor) {
      const auto i = occupied_now[u->id];
      if (u == v_pusher ||
          (u->neighbor.size() == 1 && i != NO_AGENT && ins->goals[i] == u)) {
        --n;
      } else {
        tmp = u;
      }
    }
    if (n >= 2) return true;  // able to swap at v_next
    if (n <= 0) return false;
    v_pusher = v_puller;
    v_puller = tmp;
  }
  return false;
}
