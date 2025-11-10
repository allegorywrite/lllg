#include "../include/lacam.hpp"

bool LaCAM::ANYTIME = false;
int LaCAM::STEP_LIMIT = -1;

HNode::HNode(Config _Q, DistTable *D, HNode *_parent)
    : Q(_Q),
      parent(_parent),
      depth(parent == nullptr ? 0 : parent->depth + 1),
      g(parent == nullptr ? 0 : parent->g),
      priorities(Q.size()),
      order(Q.size(), 0),
      search_tree(),
      local_guide_paths()
{
  search_tree.push(new LNode());
  const auto N = Q.size();

  local_guide_paths.resize(N);

  for (auto i = 0; i < N; ++i) {
    // set priorities
    if (parent == nullptr) {
      // initialize
      priorities[i] = std::make_tuple(0, 0, (float)D->get(i, Q[i]) / 10000);
    } else {
      // dynamic priorities, akin to PIBT
      auto &&pp = parent->priorities[i];
      auto p = (D->get(i, Q[i]) != 0) ? (std::get<0>(pp) + 1) : 0;
      auto q = (D->get(i, Q[i]) == 0) ? std::get<1>(pp) - 1 : 0;
      priorities[i] = std::make_tuple(p, q, std::get<2>(pp));
    }

    // compute cost, sum-of-loss
    if (parent != nullptr) {
      if (parent->Q[i] != Q[i] || D->get(i, Q[i]) > 0) g += 1;
    }
  }

  // set order
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
            [&](int i, int j) { return priorities[i] > priorities[j]; });
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

LNode::~LNode(){};

LaCAM::LaCAM(const Instance *_ins, DistTable *_D, int _verbose,
             const Deadline *_deadline, int _seed)
    : ins(_ins),
      D(_D),
      deadline(_deadline),
      seed(_seed),
      MT(std::mt19937(seed)),
      verbose(_verbose),
      pibt(ins, D, seed),
      global_guide(ins, D, deadline, seed),
      local_guide(ins, D, seed, &global_guide),
      loop_cnt(0)
{
}

LaCAM::~LaCAM() {}

Solution LaCAM::solve()
{
  solver_info(2, "LaCAM begins");
  reached_horizon = false;
  last_partial_solution.clear();

  // construct global guidance
  global_guide.construct();
  pibt.global_guide = &global_guide;
  pibt.local_guide = &local_guide;
  if (GlobalGuide::ON) solver_info(2, "constructed global guide");

  // setup search
  auto OPEN = std::deque<HNode *>();
  auto EXPLORED = std::unordered_map<Config, HNode *, ConfigHasher>();
  HNodes GC_HNodes;

  // insert initial node
  auto H_init = new HNode(ins->starts, D);
  OPEN.push_front(H_init);
  EXPLORED[H_init->Q] = H_init;
  GC_HNodes.push_back(H_init);

  HNode *H_goal = nullptr;

  // search loop
  solver_info(2, "search iteration begins");
  while (!OPEN.empty() && !is_expired(deadline)) {
    ++loop_cnt;

    // do not pop here!
    auto H = OPEN.front();  // high-level node

    // check uppwer bounds
    if (H_goal != nullptr && H->g >= H_goal->g) {
      OPEN.pop_front();
      solver_info(3, "prune, g=", H->g, " >= ", H_goal->g);
      OPEN.push_front(H_init);
      continue;
    }

    // check goal condition
    if (is_same_config(H->Q, ins->goals)) {
      if (H_goal == nullptr) {
        H_goal = H;
        solver_info(2, "found solution, g=", H->g);
        if (!ANYTIME) break;
      } else if (H->g < H_goal->g) {
        solver_info(2, "update solution, g:", H_goal->g, " -> ", H->g);
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
    solver_info(2, "about to call set_new_config, L->depth=", L->depth);
    auto res = set_new_config(H, L, Q_to);

    // low level search
    if (res && L->depth < H->Q.size()) {
      const auto i = H->order[L->depth];
      auto &&C = H->Q[i]->actions;
      std::shuffle(C.begin(), C.end(), MT);  // randomize
      for (auto u : C) H->search_tree.push(new LNode(L, i, u));
    }
    delete L;
    if (!res) continue;

    // check explored list
    auto iter = EXPLORED.find(Q_to);
    if (iter != EXPLORED.end()) {
      // known configuration
      auto H_known = iter->second;
      // depth limit check before creating successor (solution size = depth+1)
      if (STEP_LIMIT >= 0 && H->depth + 1 > STEP_LIMIT) {
        reached_horizon = true;
        break; // stop search at horizon
      }
      auto H_new = new HNode(Q_to, D, H);
      // limit depth by STEP_LIMIT (solution size = depth+1)
      if (STEP_LIMIT >= 0 && H_new->depth > STEP_LIMIT) {
        reached_horizon = true;
        delete H_new;
        break; // stop search at horizon
      }
      if (H_known->g < H_new->g) {
        OPEN.push_front(H_known);
        delete H_new;
      } else {
        // replace
        OPEN.push_front(H_new);
        EXPLORED[H_new->Q] = H_new;
        GC_HNodes.push_back(H_new);
      }
    } else {
      // new one -> insert
      // depth limit check before creating successor
      if (STEP_LIMIT >= 0 && H->depth + 1 > STEP_LIMIT) {
        reached_horizon = true;
        break; // stop search at horizon
      }
      auto H_new = new HNode(Q_to, D, H);
      if (STEP_LIMIT >= 0 && H_new->depth > STEP_LIMIT) {
        reached_horizon = true;
        delete H_new;
        break; // stop search at horizon
      }
      OPEN.push_front(H_new);
      EXPLORED[H_new->Q] = H_new;
      GC_HNodes.push_back(H_new);
    }
  }

  // backtrack
  Solution solution;
  std::vector<std::vector<Path>> solution_local_guide_paths;  // ソリューションに対応するLocalGuideの履歴
  {
    auto H = H_goal;
    if (H == nullptr && !GC_HNodes.empty()) {
      // fallback: use deepest explored node as partial solution source
      int max_depth = -1;
      HNode* best_node = nullptr;
      for (auto node : GC_HNodes) {
        if (node->depth > max_depth) { max_depth = node->depth; best_node = node; }
      }
      H = best_node;
    }
    Solution rev;
    while (H != nullptr) {
      rev.push_back(H->Q);
      if (!H->local_guide_paths.empty()) {
        solution_local_guide_paths.push_back(H->local_guide_paths);
      }
      H = H->parent;
    }
    std::reverse(rev.begin(), rev.end());
    solution = rev;
    std::reverse(solution_local_guide_paths.begin(), solution_local_guide_paths.end());
    
    if (!solution_local_guide_paths.empty()) {
      local_guide.reconstruct_solution_paths(solution_local_guide_paths);
      solver_info(2, "reconstructed LocalGuide solution paths with ", solution_local_guide_paths.size(), " steps");
    }
    // store partial (or full) for external use when solve() returns empty (no goal found)
    last_partial_solution = solution;
  }

  if (solution.empty() && OPEN.empty()) solver_info(2, "unsolvable instance");

  // end processing
  for (auto &&H : GC_HNodes) delete H;  // memory management

  return solution;
}

bool LaCAM::set_new_config(HNode *H, LNode *L, Config &Q_to)
{
  // Debug: entering LocalGuide construction
  solver_info(3, "enter set_new_config: H->depth=", H->depth,
              " L->depth=", L->depth);
  local_guide.construct(H->Q, H->order);

  H->local_guide_paths = local_guide.get_current_guide_paths();

  for (auto d = 0; d < L->depth; ++d) Q_to[L->who[d]] = L->where[d];
  auto ok = pibt.set_new_config(H->Q, Q_to, H->order);
  solver_info(3, "leave set_new_config: ok=", ok);
  return ok;
}
