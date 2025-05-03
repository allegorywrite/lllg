#include "../include/local_guide.hpp"

bool LocalGuide::ON = true;
int LocalGuide::WINDOW = 10;
int LocalGuide::NUM_REFINE = 1;

LocalGuide::LocalGuide(const Instance* _ins, DistTable* _D, int seed,
                       GlobalGuide* gg)
    : ins(_ins),
      MT(std::mt19937(seed)),
      N(ins->N),
      V_size(ins->G->size()),
      D(_D),
      CT(ins, true),
      guide_paths(N, Path(WINDOW, nullptr)),
      CLOSED(WINDOW, WSPPNodes(V_size, nullptr)),
      Q_to(N, nullptr),
      global_guide(gg)
{
  for (auto k = 0; k < 10000; ++k) wspp_nodes.push_back(new WSPPNode());
}

LocalGuide::~LocalGuide()
{
  for (auto&& n : wspp_nodes)
    if (n != nullptr) delete n;
}

void LocalGuide::construct(const Config& Q_from, const std::vector<int>& order)
{
  if (!ON || WINDOW < 2 || NUM_REFINE <= 0) return;

  auto cmp = [&](WSPPNode* a, WSPPNode* b) {
    if (a->f != b->f) return a->f > b->f;
    if (a->g != b->g) return a->g < b->g;
    return false;
  };

  int wspp_node_idx = 0;
  auto get_node = [&](const int who, Vertex* where, WSPPNode* parent) {
    auto n = wspp_nodes[wspp_node_idx];
    n->when = (parent == nullptr) ? 0 : parent->when + 1;
    n->where = where;
    n->parent = parent;

    // g-value
    auto collision =
        (parent == nullptr)
            ? 0
            : CT.getCollisionCost(parent->where, where, parent->when);
    n->g = (parent == nullptr) ? 0 : parent->g + 1;
    if (collision >= 1) n->g += 1 + collision * 1e-7;

    // h-value
    n->h = D->get(who, where);
    auto&& gg_h = global_guide->get(who, where);
    n->h += gg_h.first * 1e-2 + gg_h.second * 1e-4;

    n->f = n->g + n->h;
    wspp_node_idx += 1;
    return n;
  };
  std::vector<std::pair<int, int> > CLOSED_idx;

  auto update_guide_path = [&](const int i) {
    // special case
    if (Q_from[i] == ins->goals[i]) {
      for (auto t = 0; t < WINDOW; ++t) guide_paths[i][t] = Q_from[i];
      return;
    }

    // initialize search utils
    std::priority_queue<WSPPNode*, WSPPNodes, decltype(cmp)> OPEN(cmp);
    CLOSED_idx.clear();
    wspp_node_idx = 0;

    // initial node
    auto n_init = get_node(i, Q_from[i], nullptr);
    OPEN.push(n_init);

    // serach
    while (!OPEN.empty()) {
      // minimum node
      auto n = OPEN.top();
      OPEN.pop();

      // check closed
      if (CLOSED[n->when][n->where->id] != nullptr) continue;
      CLOSED[n->when][n->where->id] = n;
      CLOSED_idx.emplace_back(n->when, n->where->id);

      // check goal condition
      if (n->when == WINDOW - 1) {
        // register to CT
        while (n != nullptr) {
          guide_paths[i][n->when] = n->where;
          n = n->parent;
        }
        break;
      }

      // expand
      auto&& C = n->where->actions;
      std::shuffle(C.begin(), C.end(), MT);
      for (auto&& v : C) {
        const auto t = n->when + 1;
        if (CLOSED[t][v->id] != nullptr) continue;
        OPEN.push(get_node(i, v, n));
      }
    }

    // clear CLOSED
    for (auto&& st : CLOSED_idx) CLOSED[st.first][st.second] = nullptr;
  };

  // create initial candidate
  for (auto i = 0; i < N; ++i) {
    if (Q_from[i] != guide_paths[i][1]) continue;
    for (auto t = 0; t < WINDOW - 1; ++t) {
      guide_paths[i][t] = guide_paths[i][t + 1];
    }
    CT.enrollPath(i, guide_paths[i]);
  }

  for (auto k = 0; k < NUM_REFINE; ++k) {
    for (auto _i = 0; _i < N; ++_i) {
      const auto i = order[_i];
      Q_to[i] = nullptr;
      CT.clearPath(i, guide_paths[i]);
      update_guide_path(i);
      CT.enrollPath(i, guide_paths[i]);
    }
  }

  // post processing
  for (int i = 0; i < N; ++i) {
    Q_to[i] = guide_paths[i][1];
    CT.clearPath(i, guide_paths[i]);
  }
}

LocalHeuristic LocalGuide::get(const int i, Vertex* v)
{
  if (!ON || WINDOW < 2 || NUM_REFINE <= 0 || Q_to[i] == nullptr)
    return D->get(i, v);
  if (v == Q_to[i]) return 0;
  return 1;
}
