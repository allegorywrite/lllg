#include "../include/lacam.hpp"

#include <algorithm>
#include <climits>

bool LaCAM::ANYTIME = false;
bool LaCAM::REWRITE = false;
bool LaCAM::REWRITE_LOG = false;
bool LaCAM::REWRITE_LOG_GOAL_ONLY = false;
bool LaCAM::REWRITE_LOG_SUMMARY = false;
int LaCAM::REWRITE_LOG_LEVEL = 2;
int LaCAM::REWRITE_LOG_MAX = 50;
int LaCAM::PIBT_NUM = 1;
bool LaCAM::MC_USE_HEURISTIC = true;
int LaCAM::STEP_LIMIT = -1;
bool LaCAM::RELAX_GOAL = false;
LaCAM::CostMode LaCAM::COST_MODE = LaCAM::CostMode::LegacyEdge;
int LaCAM::COST_W_UNREACHED = 1;
int LaCAM::COST_W_MOVE = 1;
bool LaCAM::UNREACHED_USE_AFTER = false;

static bool cost_less(const PathCost &a, const PathCost &b)
{
  if (LaCAM::COST_MODE == LaCAM::CostMode::LexiUnreachedMove) {
    if (a.primary != b.primary) return a.primary < b.primary;
    return a.secondary < b.secondary;
  }
  return a.primary < b.primary;
}

static bool cost_geq(const PathCost &a, const PathCost &b)
{
  if (LaCAM::COST_MODE == LaCAM::CostMode::LexiUnreachedMove) {
    if (a.primary != b.primary) return a.primary > b.primary;
    return a.secondary >= b.secondary;
  }
  return a.primary >= b.primary;
}

bool CompareHNodePointers::operator()(const HNode *l, const HNode *r) const
{
  if (l == r) return false;
  const int N = static_cast<int>(l->Q.size());
  for (int i = 0; i < N; ++i) {
    if (l->Q[i] != r->Q[i]) return l->Q[i]->id < r->Q[i]->id;
  }
  // Disambiguate by arrived-goal state (needed under RELAX_GOAL).
  return std::lexicographical_compare(
      l->arrived_goal.begin(), l->arrived_goal.end(), r->arrived_goal.begin(),
      r->arrived_goal.end());
}

HNode::HNode(Config _Q, DistTable *D, HNode *_parent,
             const std::vector<HNodePriority> *initial_priorities)
    : Q(_Q),
      parent(_parent),
      depth(parent == nullptr ? 0 : parent->depth + 1),
      neighbor(),
      g(parent == nullptr ? PathCost{0, 0} : parent->g),
      priorities(Q.size()),
      order(Q.size(), 0),
      search_tree(),
      local_guide_paths(),
      arrived_goal(parent == nullptr ? std::vector<uint8_t>(Q.size(), 0)
                                     : parent->arrived_goal),
      arrived_goal_cnt(parent == nullptr ? 0 : parent->arrived_goal_cnt)
{
  search_tree.push(new LNode());
  const int N = static_cast<int>(Q.size());
  if (parent != nullptr) {
    neighbor.insert(parent);
    parent->neighbor.insert(this);
  }

  const bool has_override =
      (parent == nullptr && initial_priorities != nullptr &&
       initial_priorities->size() == static_cast<size_t>(N));

  for (auto i = 0; i < N; ++i) {
    const auto dist_to_goal = D->get(i, Q[i]);

    // set priorities
    if (has_override) {
      priorities[i] = (*initial_priorities)[i];
      std::get<2>(priorities[i]) = (float)dist_to_goal / 10000;
    } else if (parent == nullptr) {
      // initialize
      priorities[i] = std::make_tuple(0, 0, (float)dist_to_goal / 10000);
    } else {
      // dynamic priorities, akin to PIBT
      auto &&pp = parent->priorities[i];
      auto p = (dist_to_goal != 0) ? (std::get<0>(pp) + 1) : 0;
      auto q = (dist_to_goal == 0) ? std::get<1>(pp) - 1 : 0;
      priorities[i] = std::make_tuple(p, q, std::get<2>(pp));
    }

    // compute cost, sum-of-loss
    if (parent != nullptr) {
      const bool moved = (parent->Q[i] != Q[i]);
      const bool not_at_goal = (dist_to_goal > 0);
      const bool unreached = (arrived_goal[i] == 0);
      const bool unreached_after = unreached && not_at_goal;
      switch (LaCAM::COST_MODE) {
        case LaCAM::CostMode::LegacyEdge:
          if (moved || not_at_goal) g.primary += 1;
          break;
        case LaCAM::CostMode::UnreachedCount:
          // Count "still not yet visited goal" agents at the previous timestep.
          if (unreached) g.primary += 1;
          break;
        case LaCAM::CostMode::WeightedSumUnreachedMove:
          if (LaCAM::UNREACHED_USE_AFTER) {
            if (unreached_after) g.primary += LaCAM::COST_W_UNREACHED;
          } else {
            if (unreached) g.primary += LaCAM::COST_W_UNREACHED;
          }
          if (moved) g.primary += LaCAM::COST_W_MOVE;
          break;
        case LaCAM::CostMode::LexiUnreachedMove:
          if (unreached) g.primary += 1;
          if (moved) g.secondary += 1;
          break;
        case LaCAM::CostMode::NextGoalMissCount:
          if (not_at_goal) g.primary += 1;
          break;
      }
    }

    // update arrived-goal flags (monotone)
    if (dist_to_goal == 0 && arrived_goal[i] == 0) {
      arrived_goal[i] = 1;
      ++arrived_goal_cnt;
    }
  }

  // set order
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
            [&](int i, int j) { return priorities[i] > priorities[j]; });

  // Debug: log HNode creation with positions and cost
  // if (PIBT::DETERMINISTIC && parent != nullptr) {
  //   std::cout << "HNode created: depth=" << depth << " g=" << g << "
  //   config="; for (auto i = 0; i < N; ++i) {
  //     std::cout << "(" << Q[i]->x << "," << Q[i]->y << ")";
  //     if (i < N - 1) std::cout << ",";
  //   }
  //   std::cout << std::endl;
  // }
}

HNode::~HNode()
{
  while (!search_tree.empty()) {
    delete search_tree.front();
    search_tree.pop();
  }
}

LNode::LNode() : who(), where(), depth(0) {}

LNode::LNode(LNode *parent, int i, Vertex *v)
    : who(parent->who), where(parent->where), depth(parent->depth + 1)
{
  who.push_back(i);
  where.push_back(v);
}

LNode::~LNode() {};

LaCAM::LaCAM(const Instance *_ins, DistTable *_D, int _verbose,
             const Deadline *_deadline, int _seed)
    : ins(_ins),
      D(_D),
      deadline(_deadline),
      seed(_seed),
      MT(std::mt19937(seed)),
      verbose(_verbose),
      pibt(ins, D, seed),
      extra_pibts(),
      pibt_pool(),
      global_guide(ins, D, deadline, seed),
      local_guide(ins, D, seed, &global_guide),
      loop_cnt(0)
{
}

LaCAM::~LaCAM() {}

void LaCAM::set_initial_priorities(const std::vector<HNodePriority> &priorities)
{
  initial_root_priorities = priorities;
  has_initial_root_priorities = true;
}

Solution LaCAM::solve()
{
  solver_info(2, "LaCAM begins");
  rewrite_call_cnt = 0;
  rewrite_relax_cnt = 0;
  rewrite_relax_printed = 0;
  reached_horizon = false;
  last_partial_solution.clear();
  last_root_priorities.clear();
  last_solution_priorities.clear();

  // construct global guidance
  global_guide.construct();
  // setup PIBT pool (Monte-Carlo configuration generation)
  extra_pibts.clear();
  pibt_pool.clear();
  pibt_pool.reserve(std::max(1, PIBT_NUM));
  pibt_pool.push_back(&pibt);
  for (int k = 1; k < std::max(1, PIBT_NUM); ++k) {
    extra_pibts.emplace_back(std::make_unique<PIBT>(ins, D, seed + k));
    pibt_pool.push_back(extra_pibts.back().get());
  }
  for (auto *p : pibt_pool) {
    p->global_guide = &global_guide;
    p->local_guide = &local_guide;
  }
  if (GlobalGuide::ON) solver_info(2, "constructed global guide");

  // setup search
  OPEN.clear();
  EXPLORED.clear();
  EXPLORED_ARRIVED.clear();
  GC_HNodes.clear();
  H_init = nullptr;
  H_goal = nullptr;

  // insert initial node
  const std::vector<HNodePriority> *root_priority_override = nullptr;
  if (has_initial_root_priorities && initial_root_priorities.size() == ins->N) {
    root_priority_override = &initial_root_priorities;
  }
  has_initial_root_priorities = false;
  H_init = new HNode(ins->starts, D, nullptr, root_priority_override);
  last_root_priorities = H_init->priorities;
  OPEN.push_front(H_init);
  if (!RELAX_GOAL) {
    EXPLORED[H_init->Q] = H_init;
  } else {
    EXPLORED_ARRIVED[ArrivedKey{H_init->Q, H_init->arrived_goal}] = H_init;
  }
  GC_HNodes.push_back(H_init);

  // search loop
  solver_info(2, "search iteration begins");
  while (!OPEN.empty() && !is_expired(deadline)) {
    ++loop_cnt;

    // do not pop here!
    auto H = OPEN.front();  // high-level node
    // if (verbose >= 2) {
    //   std::cout << "Agent priority order (depth " << H->depth << "): ";
    //   for (int idx = 0; idx < static_cast<int>(H->order.size()); ++idx) {
    //     std::cout << H->order[idx];
    //     if (idx + 1 < static_cast<int>(H->order.size())) std::cout << ',';
    //   }
    //   std::cout << "\nAgent priorities: ";
    //   for (size_t idx = 0; idx < H->priorities.size(); ++idx) {
    //     const auto &prio = H->priorities[idx];
    //     std::cout << idx << "=(" << std::get<0>(prio) << ','
    //               << std::get<1>(prio) << ',' << std::get<2>(prio) << ')';
    //     if (idx + 1 < H->priorities.size()) std::cout << ", ";
    //   }
    //   std::cout << std::endl;
    // }

    // check upper bounds
    if (H_goal != nullptr && cost_geq(H->g, H_goal->g)) {
      OPEN.pop_front();
      solver_info(3, "prune, g primary=", H->g.primary,
                  " secondary=", H->g.secondary,
                  " >= goal primary=", H_goal->g.primary,
                  " secondary=", H_goal->g.secondary);
      OPEN.push_front(H_init);
      continue;
    }

    // check goal condition
    const bool is_goal =
        (!RELAX_GOAL) ? is_same_config(H->Q, ins->goals)
                      : (H->arrived_goal_cnt == static_cast<int>(ins->N));
    if (is_goal) {
      if (H_goal == nullptr) {
        H_goal = H;
        solver_info(2, "found solution, g primary=", H->g.primary,
                    " secondary=", H->g.secondary);
        if (!ANYTIME) break;
      } else if (cost_less(H->g, H_goal->g)) {
        solver_info(2, "update solution, g:", H_goal->g.primary, ",",
                    H_goal->g.secondary, " -> ", H->g.primary, ",",
                    H->g.secondary);
        H_goal = H;
      }
    }

    // extract constraints
    if (H->search_tree.empty()) {
      OPEN.pop_front();
      continue;
    }
    auto L = H->search_tree.front();
    H->search_tree.pop();

    // create successors at the high-level search
    auto Q_to = Config(ins->N, nullptr);
    // Helpful debug at verbose>=2 to locate crash point
    // solver_info(2, "about to call set_new_config, L->depth=", L->depth);
    auto res = set_new_config(H, L, Q_to);

    // low level search
    if (res && L->depth < static_cast<int>(H->Q.size())) {
      const auto i = H->order[L->depth];
      auto &&C = H->Q[i]->actions;
      std::shuffle(C.begin(), C.end(), MT);  // randomize
      for (auto u : C) H->search_tree.push(new LNode(L, i, u));
    }
    delete L;
    if (!res) {
      // if (PIBT::DETERMINISTIC) std::cout << "  PIBT failed for this LNode" <<
      // std::endl;
      continue;
    }
    // if (PIBT::DETERMINISTIC) {
    //   std::cout << "  PIBT succeeded, Q_to=";
    //   for (auto i = 0; i < ins->N; ++i) {
    //     std::cout << "(" << Q_to[i]->x << "," << Q_to[i]->y << ")";
    //     if (i < ins->N - 1) std::cout << ",";
    //   }
    //   std::cout << std::endl;
    // }

    // check explored list
    if (!RELAX_GOAL) {
      auto iter = EXPLORED.find(Q_to);
      if (iter != EXPLORED.end()) {
        // known configuration
        auto H_known = iter->second;
        // depth limit check before creating successor (solution size = depth+1)
        if (STEP_LIMIT >= 0 && H->depth + 1 > STEP_LIMIT) {
          reached_horizon = true;
          if (ANYTIME) continue;  // keep searching until deadline/OPEN empty
          break;                  // legacy behavior: stop search at horizon
        }
        // limit depth by STEP_LIMIT (solution size = depth+1)
        if (STEP_LIMIT >= 0 && H->depth + 1 >= STEP_LIMIT) {
          reached_horizon = true;
        }
        if (REWRITE) {
          // Connect and relax costs/parents, as in lacam3.
          rewrite(H, H_known);
          OPEN.push_front(H_known);
        } else {
          // legacy behavior: create a new node if it improves g, otherwise
          // reuse known
          const auto step = get_transition_cost(H, Q_to);
          PathCost g_new{H->g.primary + step.primary,
                         H->g.secondary + step.secondary};
          if (cost_less(H_known->g, g_new)) {
            OPEN.push_front(H_known);
          } else {
            auto H_new = new HNode(Q_to, D, H);
            OPEN.push_front(H_new);
            EXPLORED[H_new->Q] = H_new;
            GC_HNodes.push_back(H_new);
          }
        }
      } else {
        // new one -> insert
        // depth limit check before creating successor
        if (STEP_LIMIT >= 0 && H->depth + 1 > STEP_LIMIT) {
          reached_horizon = true;
          if (ANYTIME) continue;  // keep searching until deadline/OPEN empty
          break;                  // legacy behavior: stop search at horizon
        }
        auto H_new = new HNode(Q_to, D, H);
        if (STEP_LIMIT >= 0 && H_new->depth >= STEP_LIMIT) {
          reached_horizon = true;
        }
        OPEN.push_front(H_new);
        EXPLORED[H_new->Q] = H_new;
        GC_HNodes.push_back(H_new);
      }
    } else {
      // depth limit check before creating successor (solution size = depth+1)
      if (STEP_LIMIT >= 0 && H->depth + 1 > STEP_LIMIT) {
        reached_horizon = true;
        if (ANYTIME) continue;  // keep searching until deadline/OPEN empty
        break;                  // legacy behavior: stop search at horizon
      }
      // limit depth by STEP_LIMIT (solution size = depth+1)
      if (STEP_LIMIT >= 0 && H->depth + 1 >= STEP_LIMIT) {
        reached_horizon = true;
      }

      // compute arrived-goal state without constructing a node
      auto arrived = H->arrived_goal;
      for (int i = 0; i < static_cast<int>(ins->N); ++i) {
        if (arrived[i]) continue;
        if (D->get(i, Q_to[i]) == 0) arrived[i] = 1;
      }
      ArrivedKey key{Q_to, std::move(arrived)};
      auto iter = EXPLORED_ARRIVED.find(key);
      const auto step = get_transition_cost(H, Q_to);
      PathCost g_new{H->g.primary + step.primary,
                     H->g.secondary + step.secondary};
      if (iter != EXPLORED_ARRIVED.end()) {
        auto H_known = iter->second;
        if (REWRITE) {
          rewrite(H, H_known);
          OPEN.push_front(H_known);
        } else if (cost_less(H_known->g, g_new)) {
          OPEN.push_front(H_known);
        } else {
          auto H_new = new HNode(Q_to, D, H);
          OPEN.push_front(H_new);
          iter->second = H_new;  // replace value; key is equivalent
          GC_HNodes.push_back(H_new);
        }
      } else {
        auto H_new = new HNode(Q_to, D, H);
        OPEN.push_front(H_new);
        EXPLORED_ARRIVED.emplace(std::move(key), H_new);
        GC_HNodes.push_back(H_new);
      }
    }
  }

  // backtrack
  Solution solution;
  std::vector<std::vector<Path>>
      solution_local_guide_paths;  // ソリューションに対応するLocalGuideの履歴
  std::vector<std::vector<HNodePriority>> solution_priorities_rev;
  {
    auto H = H_goal;
    if (H == nullptr && !GC_HNodes.empty()) {
      // fallback: use deepest explored node as partial solution source
      int max_depth = -1;
      HNode *best_node = nullptr;
      for (auto node : GC_HNodes) {
        if (node->depth > max_depth) {
          max_depth = node->depth;
          best_node = node;
        }
      }
      H = best_node;
    }
    Solution rev;
    while (H != nullptr) {
      rev.push_back(H->Q);
      solution_priorities_rev.push_back(H->priorities);
      if (!H->local_guide_paths.empty()) {
        solution_local_guide_paths.push_back(H->local_guide_paths);
      }
      H = H->parent;
    }
    std::reverse(rev.begin(), rev.end());
    std::reverse(solution_priorities_rev.begin(),
                 solution_priorities_rev.end());
    solution = rev;
    std::reverse(solution_local_guide_paths.begin(),
                 solution_local_guide_paths.end());
    last_solution_priorities = solution_priorities_rev;
    if (!last_solution_priorities.empty()) {
      last_root_priorities = last_solution_priorities.front();
    } else {
      last_root_priorities.clear();
    }

    if (!solution_local_guide_paths.empty()) {
      local_guide.reconstruct_solution_paths(solution_local_guide_paths);
      solver_info(2, "reconstructed LocalGuide solution paths with ",
                  solution_local_guide_paths.size(), " steps");
    }
    // store partial (or full) for external use when solve() returns empty (no
    // goal found)
    last_partial_solution = solution;
  }

  if (solution.empty() && OPEN.empty()) solver_info(2, "unsolvable instance");

  if (REWRITE && REWRITE_LOG_SUMMARY) {
    solver_info(REWRITE_LOG_LEVEL, "rewrite summary: calls=", rewrite_call_cnt,
                " relaxations=", rewrite_relax_cnt,
                " printed=", rewrite_relax_printed);
  }

  // end processing
  for (auto &&H : GC_HNodes) delete H;  // memory management
  OPEN.clear();
  EXPLORED.clear();
  EXPLORED_ARRIVED.clear();
  GC_HNodes.clear();
  H_init = nullptr;
  H_goal = nullptr;

  return solution;
}

bool LaCAM::set_new_config(HNode *H, LNode *L, Config &Q_to)
{
  local_guide.construct(H->Q, H->order);

  H->local_guide_paths = local_guide.get_current_guide_paths();
  const int trials = std::max(1, PIBT_NUM);

  auto heuristic_cost = [&](const Config &C) -> int {
    int h = 0;
    for (int i = 0; i < static_cast<int>(ins->N); ++i) {
      const int dist = D->get(i, C[i]);
      if (!LocalGuide::ON) {
        h += dist;
        continue;
      }
      // Similar spirit to PIBT's LocalGuide usage: prefer the guided next
      // vertex, but still account for distance-to-goal.
      int guide_penalty = 0;
      if (local_guide.Q_to.size() == static_cast<size_t>(ins->N) &&
          local_guide.Q_to[i] != nullptr && C[i] != local_guide.Q_to[i]) {
        guide_penalty = 1;
      }
      h += dist + guide_penalty;
    }
    return h;
  };

  // Prepare candidate configs with constraints.
  std::vector<Config> Q_cands(trials, Config(ins->N, nullptr));
  for (int k = 0; k < trials; ++k) {
    for (int d = 0; d < L->depth; ++d) Q_cands[k][L->who[d]] = L->where[d];
  }

  int best_idx = -1;
  int best_f = INT_MAX;
  for (int k = 0; k < trials; ++k) {
    auto *pibt_k =
        (k < static_cast<int>(pibt_pool.size())) ? pibt_pool[k] : &pibt;
    const auto ok = pibt_k->set_new_config(H->Q, Q_cands[k], H->order);
    if (!ok) continue;
    const auto step = get_transition_cost(H, Q_cands[k]);
    const int edge_score = (COST_MODE == CostMode::LexiUnreachedMove)
                               ? static_cast<int>(step.secondary)
                               : static_cast<int>(step.primary);
    const int f =
        edge_score + (MC_USE_HEURISTIC ? heuristic_cost(Q_cands[k]) : 0);
    if (f < best_f) {
      best_f = f;
      best_idx = k;
    }
  }

  if (best_idx < 0) {
    return false;
  }

  std::copy(Q_cands[best_idx].begin(), Q_cands[best_idx].end(), Q_to.begin());
  return true;
}

int LaCAM::get_edge_cost(const Config &from, const Config &to) const
{
  const int N = static_cast<int>(to.size());
  auto cost = 0;
  for (int i = 0; i < N; ++i) {
    const auto dist_to_goal = D->get(i, to[i]);
    if (from[i] != to[i] || dist_to_goal > 0) cost += 1;
  }
  return cost;
}

int LaCAM::get_move_cost(const Config &from, const Config &to) const
{
  const int N = static_cast<int>(to.size());
  int cost = 0;
  for (int i = 0; i < N; ++i) cost += (from[i] != to[i]);
  return cost;
}

int LaCAM::get_unreached_count(const HNode *from) const
{
  if (from == nullptr) return 0;
  int cost = 0;
  for (auto b : from->arrived_goal) cost += (b == 0);
  return cost;
}

int LaCAM::get_unreached_count_after(const HNode *from, const Config &to) const
{
  if (from == nullptr) return 0;
  const int N = static_cast<int>(to.size());
  const int n = std::min(N, static_cast<int>(from->arrived_goal.size()));
  int newly_arrived = 0;
  for (int i = 0; i < n; ++i) {
    if (from->arrived_goal[static_cast<size_t>(i)] != 0) continue;
    if (D->get(i, to[i]) == 0) ++newly_arrived;
  }
  return std::max(0, get_unreached_count(from) - newly_arrived);
}

int LaCAM::get_next_goal_miss_count(const Config &to) const
{
  const int N = static_cast<int>(to.size());
  int miss = 0;
  for (int i = 0; i < N; ++i) miss += (D->get(i, to[i]) > 0);
  return miss;
}

PathCost LaCAM::get_transition_cost(const HNode *from, const Config &to) const
{
  if (from == nullptr) return PathCost{0, 0};
  switch (COST_MODE) {
    case CostMode::LegacyEdge:
      return PathCost{get_edge_cost(from->Q, to), 0};
    case CostMode::UnreachedCount:
      return PathCost{get_unreached_count(from), 0};
    case CostMode::WeightedSumUnreachedMove: {
      const long long u = static_cast<long long>(
          UNREACHED_USE_AFTER ? get_unreached_count_after(from, to)
                              : get_unreached_count(from));
      const long long m = static_cast<long long>(get_move_cost(from->Q, to));
      return PathCost{u * static_cast<long long>(COST_W_UNREACHED) +
                          m * static_cast<long long>(COST_W_MOVE),
                      0};
    }
    case CostMode::LexiUnreachedMove:
      return PathCost{get_unreached_count(from), get_move_cost(from->Q, to)};
    case CostMode::NextGoalMissCount:
      return PathCost{get_next_goal_miss_count(to), 0};
  }
  return PathCost{0, 0};
}

PathCost LaCAM::get_transition_cost(const HNode *from, const HNode *to) const
{
  if (from == nullptr || to == nullptr) return PathCost{0, 0};
  return get_transition_cost(from, to->Q);
}

void LaCAM::rewrite(HNode *H_from, HNode *H_to)
{
  if (H_from == nullptr || H_to == nullptr) return;
  ++rewrite_call_cnt;

  // update neighbors (bidirectional)
  H_from->neighbor.insert(H_to);
  H_to->neighbor.insert(H_from);

  // Dijkstra-like relaxation (queue is sufficient, as in lacam3)
  std::queue<HNode *> Q({H_from});
  while (!Q.empty()) {
    auto n_from = Q.front();
    Q.pop();
    for (auto n_to : n_from->neighbor) {
      if (RELAX_GOAL) {
        const auto n =
            std::min(n_from->arrived_goal.size(), n_to->arrived_goal.size());
        bool monotone = true;
        for (size_t i = 0; i < n; ++i) {
          if (n_from->arrived_goal[i] > n_to->arrived_goal[i]) {
            monotone = false;
            break;
          }
        }
        if (!monotone) continue;
      }
      const auto step = get_transition_cost(n_from, n_to);
      PathCost g_val{n_from->g.primary + step.primary,
                     n_from->g.secondary + step.secondary};
      if (cost_less(g_val, n_to->g)) {
        ++rewrite_relax_cnt;
        const auto old = n_to->g;
        const bool is_goal_node = (n_to == H_goal);

        if (REWRITE_LOG && (!REWRITE_LOG_GOAL_ONLY || is_goal_node) &&
            (REWRITE_LOG_MAX <= 0 ||
             rewrite_relax_printed < static_cast<long long>(REWRITE_LOG_MAX))) {
          ++rewrite_relax_printed;
          solver_info(REWRITE_LOG_LEVEL, "rewrite relax: (", old.primary, ",",
                      old.secondary, ") -> (", g_val.primary, ",",
                      g_val.secondary, ")", is_goal_node ? " [goal]" : "",
                      " depth ", n_from->depth, " -> ", (n_from->depth + 1));
        } else if (is_goal_node) {
          // Legacy behavior: announce a goal cost improvement at verbose>=2.
          solver_info(2, "cost update: ", H_goal->g.primary, ",",
                      H_goal->g.secondary, " -> ", g_val.primary, ",",
                      g_val.secondary);
        }
        n_to->g = g_val;
        n_to->parent = n_from;
        n_to->depth = n_from->depth + 1;
        Q.push(n_to);
        if (H_goal != nullptr && cost_less(n_to->g, H_goal->g))
          OPEN.push_front(n_to);
      }
    }
  }
}

void LaCAM::apply_new_solution(const Solution &plan)
{
  if (!REWRITE) return;
  if (plan.empty()) return;

  if (!RELAX_GOAL) {
    auto iter0 = EXPLORED.find(plan.front());
    if (iter0 == EXPLORED.end()) return;

    auto H_from = iter0->second;
    for (size_t t = 1; t < plan.size(); ++t) {
      const auto &Q = plan[t];
      HNode *H_to = nullptr;
      auto iter = EXPLORED.find(Q);
      if (iter != EXPLORED.end()) {
        H_to = iter->second;
        rewrite(H_from, H_to);
      } else {
        H_to = new HNode(Q, D, H_from);
        EXPLORED[H_to->Q] = H_to;
        GC_HNodes.push_back(H_to);
        OPEN.push_front(H_to);
      }
      H_from = H_to;
    }

    if (is_same_config(plan.back(), ins->goals)) {
      auto iter_goal = EXPLORED.find(plan.back());
      if (iter_goal != EXPLORED.end()) {
        auto node = iter_goal->second;
        if (H_goal == nullptr || cost_less(node->g, H_goal->g)) H_goal = node;
      }
    }
    return;
  }

  // RELAX_GOAL: state is (Config, arrived_goal), so reconstruct arrived_goal
  // along plan.
  if (ins == nullptr) return;
  const int N = static_cast<int>(ins->N);
  if (plan.front().size() != static_cast<size_t>(N)) return;

  std::vector<uint8_t> arrived(static_cast<size_t>(N), 0);
  int arrived_cnt = 0;
  for (int i = 0; i < N; ++i) {
    if (D->get(i, plan.front()[i]) == 0) {
      arrived[static_cast<size_t>(i)] = 1;
      ++arrived_cnt;
    }
  }

  auto iter0 = EXPLORED_ARRIVED.find(ArrivedKey{plan.front(), arrived});
  if (iter0 == EXPLORED_ARRIVED.end()) return;

  auto H_from = iter0->second;
  for (size_t t = 1; t < plan.size(); ++t) {
    const auto &Q = plan[t];
    if (Q.size() != static_cast<size_t>(N)) return;
    for (int i = 0; i < N; ++i) {
      if (arrived[static_cast<size_t>(i)]) continue;
      if (D->get(i, Q[i]) == 0) {
        arrived[static_cast<size_t>(i)] = 1;
        ++arrived_cnt;
      }
    }

    auto key = ArrivedKey{Q, arrived};
    HNode *H_to = nullptr;
    auto iter = EXPLORED_ARRIVED.find(key);
    if (iter != EXPLORED_ARRIVED.end()) {
      H_to = iter->second;
      rewrite(H_from, H_to);
    } else {
      H_to = new HNode(Q, D, H_from);
      EXPLORED_ARRIVED.emplace(std::move(key), H_to);
      GC_HNodes.push_back(H_to);
      OPEN.push_front(H_to);
    }

    if (arrived_cnt == N) {
      if (H_goal == nullptr || cost_less(H_to->g, H_goal->g)) H_goal = H_to;
    }
    H_from = H_to;
  }
}
