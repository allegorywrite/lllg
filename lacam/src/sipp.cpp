#include "../include/sipp.hpp"

SITable::SITable(CollisionTable *_CT) : CT(_CT) {}

SITable::~SITable() {}

SIs &SITable::get(Vertex *v)
{
  auto &b_v = body[v->id];
  if (!b_v.empty()) return b_v;
  auto &entry = CT->body[v->id];
  auto &entry_last = CT->body_last[v->id];
  auto t_last = entry_last.empty()
                    ? INT_MAX
                    : *std::min_element(entry_last.begin(), entry_last.end());

  // insert safe interval
  auto time_start = 0;
  for (auto t = 0; t < entry.size(); ++t) {
    if (entry[t].empty()) continue;
    auto time_end = t - 1;
    if (time_start <= time_end) {
      b_v.push_back(std::make_pair(time_start, time_end));
    }
    time_start = t + 1;
    if (t_last == t) break;
  }
  // add last safe interval
  if (t_last == INT_MAX) {
    b_v.push_back(std::make_pair(time_start, INT_MAX - 1));
  }
  return b_v;
}

SINode::SINode(const int _uuid, const SI &si, Vertex *_v, int _t, int _g,
               int _f, SINode *_parent)
    : uuid(_uuid),
      time_start(si.first),
      time_end(si.second),
      v(_v),
      t(_t),
      g(_g),
      f(_f),
      parent(_parent)
{
}

bool SINode::operator==(const SINode &other) const
{
  return (other.v->id == v->id && other.time_start == time_start &&
          other.time_end == time_end);
}

uint SINodeHasher::operator()(const SINode &n) const
{
  uint hash = n.v->id;
  hash ^= n.time_start + 0x9e3779b9 + (hash << 6) + (hash >> 2);
  hash ^= n.time_end + 0x9e3779b9 + (hash << 6) + (hash >> 2);
  return hash;
}

// minimizing path-loss - not cost!
Path sipp(const int i, Vertex *s_i, Vertex *g_i, DistTable *D,
          CollisionTable *CT, const Deadline *deadline, const int f_upper_bound)
{
  auto solution_path = Path();
  auto ST = SITable(CT);  // safe interval table

  // setup goal
  auto &intervals_goal = ST.get(g_i);
  if (intervals_goal.empty()) return solution_path;
  const auto t_goal_after = intervals_goal.back().first - 1;

  // setup OPEN lists
  auto cmpNodes = [&](SINode *a, SINode *b) {
    if (a->f != b->f) return a->f > b->f;
    if (a->g != b->g) return a->g < b->g;
    if (a->time_start != b->time_start) return a->time_start > b->time_start;
    return a->uuid < b->uuid;
  };

  int node_id = 0;
  auto OPEN =
      std::priority_queue<SINode *, SINodes, decltype(cmpNodes)>(cmpNodes);
  std::unordered_map<SINode, SINode *, SINodeHasher> EXPLORED;
  OPEN.push(new SINode(++node_id, ST.get(s_i)[0], s_i, 0, 0, D->get(i, s_i),
                       nullptr));

  // main loop
  while (!OPEN.empty() && !is_expired(deadline)) {
    auto n = OPEN.top();
    OPEN.pop();

    // check known node
    auto itr_e = EXPLORED.find(*n);
    if (itr_e != EXPLORED.end() && itr_e->second->g <= n->g) {
      delete n;
      continue;
    }
    EXPLORED[*n] = n;

    // goal check
    if (n->v == g_i && n->t > t_goal_after) {
      // backtrack
      auto t = n->t;
      while (t >= 0) {
        solution_path.push_back(n->v);
        if (t == n->t) n = n->parent;
        --t;
      }
      std::reverse(solution_path.begin(), solution_path.end());
      break;
    }

    // expand neighbors
    for (auto &u : n->v->neighbor) {
      for (auto &si : ST.get(u)) {
        // invalid transition
        if (si.first > n->time_end + 1) break;
        if (si.second <= n->time_start) continue;

        // check existence of t
        auto t_earliest = INT_MAX;
        if (n->v != g_i) {
          for (auto t = std::max(n->t, si.first - 1);
               t <= std::min(n->time_end, si.second - 1); ++t) {
            if (CT->getCollisionCost(n->v, u, t) == 0) {
              t_earliest = t + 1;
              break;
            }
          }
        } else {
          // for goal node -> reverse
          for (auto t = std::min(n->time_end, si.second - 1);
               t >= std::max(n->t, si.first - 1); --t) {
            if (CT->getCollisionCost(n->v, u, t) == 0) {
              t_earliest = t + 1;
              break;
            }
          }
        }
        if (t_earliest >= INT_MAX) continue;

        // valid neighbor
        auto g_val = n->g + (n->v != g_i ? t_earliest - n->t : 1);
        auto f_val = g_val + D->get(i, u);
        auto n_new = new SINode(++node_id, si, u, t_earliest, g_val, f_val, n);

        auto itr = EXPLORED.find(*n_new);
        if (f_val > f_upper_bound ||
            (itr != EXPLORED.end() && g_val >= itr->second->g)) {
          delete n_new;
        } else {
          OPEN.push(n_new);
        }
      }
    }
  }

  // memory management
  while (!OPEN.empty()) {
    delete OPEN.top();
    OPEN.pop();
  }
  for (auto iter : EXPLORED) delete iter.second;
  return solution_path;
}

// WINDOWS[i]サイズのみ探索するSIPP
Path sipp_window(const int i, Vertex *s_i, Vertex *g_i, DistTable *D,
                 CollisionTable *CT, const int window_size,
                 const Deadline *deadline)
{
  auto solution_path = Path();
  auto ST = SITable(CT);  // safe interval table

  // setup OPEN lists
  auto cmpNodes = [&](SINode *a, SINode *b) {
    if (a->f != b->f) return a->f > b->f;
    if (a->g != b->g) return a->g < b->g;
    if (a->time_start != b->time_start) return a->time_start > b->time_start;
    return a->uuid < b->uuid;
  };

  int node_id = 0;
  auto OPEN =
      std::priority_queue<SINode *, SINodes, decltype(cmpNodes)>(cmpNodes);
  std::unordered_map<SINode, SINode *, SINodeHasher> EXPLORED;
  
  // 開始ノードの安全区間を取得
  auto &start_intervals = ST.get(s_i);
  if (start_intervals.empty()) return solution_path;
  
  OPEN.push(new SINode(++node_id, start_intervals[0], s_i, 0, 0, D->get(i, s_i),
                       nullptr));

  SINode* best_node = nullptr;
  int best_distance = INT_MAX;

  // main loop
  while (!OPEN.empty() && !is_expired(deadline)) {
    auto n = OPEN.top();
    OPEN.pop();

    // check known node
    auto itr_e = EXPLORED.find(*n);
    if (itr_e != EXPLORED.end() && itr_e->second->g <= n->g) {
      delete n;
      continue;
    }
    EXPLORED[*n] = n;

    // window_sizeに到達した場合、最良のノードを記録
    if (n->t >= window_size - 1) {
      int distance_to_goal = D->get(i, n->v);
      if (distance_to_goal < best_distance) {
        best_distance = distance_to_goal;
        best_node = n;
      }
      continue;  // これ以上展開しない
    }

    // ゴールに到達した場合（window_size内で）
    if (n->v == g_i) {
      best_node = n;
      break;
    }

    // expand neighbors
    for (auto &u : n->v->neighbor) {
      for (auto &si : ST.get(u)) {
        // invalid transition
        if (si.first > n->time_end + 1) break;
        if (si.second <= n->time_start) continue;

        // check existence of t
        auto t_earliest = INT_MAX;
        for (auto t = std::max(n->t, si.first - 1);
             t <= std::min(n->time_end, si.second - 1); ++t) {
          if (CT->getCollisionCost(n->v, u, t) == 0) {
            t_earliest = t + 1;
            break;
          }
        }
        if (t_earliest >= INT_MAX || t_earliest >= window_size) continue;

        // valid neighbor
        auto g_val = n->g + (t_earliest - n->t);
        auto f_val = g_val + D->get(i, u);
        auto n_new = new SINode(++node_id, si, u, t_earliest, g_val, f_val, n);

        auto itr = EXPLORED.find(*n_new);
        if (itr != EXPLORED.end() && g_val >= itr->second->g) {
          delete n_new;
        } else {
          OPEN.push(n_new);
        }
      }
    }
  }

  // 最良のノードからパスを構築
  if (best_node != nullptr) {
    // backtrack
    std::vector<Vertex*> temp_path;
    auto current = best_node;
    while (current != nullptr) {
      temp_path.push_back(current->v);
      current = current->parent;
    }
    std::reverse(temp_path.begin(), temp_path.end());
    
    // window_sizeに合わせてパスを調整
    solution_path.resize(window_size);
    for (int t = 0; t < window_size; ++t) {
      if (t < static_cast<int>(temp_path.size())) {
        solution_path[t] = temp_path[t];
      } else {
        // パスが短い場合は最後のノードで埋める
        solution_path[t] = temp_path.back();
      }
    }
  }

  // memory management
  while (!OPEN.empty()) {
    delete OPEN.top();
    OPEN.pop();
  }
  for (auto iter : EXPLORED) delete iter.second;
  return solution_path;
}

// SIPPS: SIPP with soft constraints (based on AAAI22 paper)
// This implementation treats collisions as soft constraints with penalties
Path sipps_window(const int i, Vertex *s_i, Vertex *g_i, DistTable *D,
                  CollisionTable *CT, const int window_size,
                  const Deadline *deadline)
{
  auto solution_path = Path();
  auto ST = SITable(CT);  // safe interval table

  // Node structure with collision count for soft constraints
  struct SIPPSNode {
    const int uuid;
    const int time_start;
    const int time_end;
    Vertex *v;
    const int t;  // arrival time
    const int g;  // travel cost
    const int c;  // collision count (soft constraints)
    const int f;  // total cost = g + h
    SIPPSNode *parent;

    SIPPSNode(const int _uuid, const SI &si, Vertex *_v, int _t, int _g, int _c, int _f, SIPPSNode *_parent)
        : uuid(_uuid), time_start(si.first), time_end(si.second), v(_v), t(_t), g(_g), c(_c), f(_f), parent(_parent) {}

    bool operator==(const SIPPSNode &other) const {
      return (other.v->id == v->id && other.time_start == time_start && other.time_end == time_end);
    }
  };

  struct SIPPSNodeHasher {
    uint operator()(const SIPPSNode &n) const {
      uint hash = n.v->id;
      hash ^= n.time_start + 0x9e3779b9 + (hash << 6) + (hash >> 2);
      hash ^= n.time_end + 0x9e3779b9 + (hash << 6) + (hash >> 2);
      return hash;
    }
  };

  // Priority function: zero collision penalty for comparison test
  auto cmpNodes = [&](SIPPSNode *a, SIPPSNode *b) {
    // Zero collision penalty to test equivalence with non-SIPP
    float score_a = a->c * 0.0f + a->f;  // Zero collision penalty
    float score_b = b->c * 0.0f + b->f;
    if (score_a != score_b) return score_a > score_b;
    if (a->g != b->g) return a->g < b->g;
    if (a->time_start != b->time_start) return a->time_start > b->time_start;
    return a->uuid < b->uuid;
  };

  int node_id = 0;
  auto OPEN = std::priority_queue<SIPPSNode *, std::vector<SIPPSNode *>, decltype(cmpNodes)>(cmpNodes);
  std::unordered_map<SIPPSNode, SIPPSNode *, SIPPSNodeHasher> EXPLORED;
  
  // 開始ノードの安全区間を取得
  auto &start_intervals = ST.get(s_i);
  if (start_intervals.empty()) return solution_path;
  
  OPEN.push(new SIPPSNode(++node_id, start_intervals[0], s_i, 0, 0, 0, D->get(i, s_i), nullptr));

  SIPPSNode* best_node = nullptr;
  int best_collision_count = INT_MAX;
  int best_distance = INT_MAX;
  int nodes_expanded = 0;
  const int max_nodes = std::min(window_size * 500, 5000); // More restrictive limit for efficiency

  // main loop
  while (!OPEN.empty() && !is_expired(deadline) && nodes_expanded < max_nodes) {
    auto n = OPEN.top();
    OPEN.pop();

    // check known node with more lenient dominance
    auto itr_e = EXPLORED.find(*n);
    if (itr_e != EXPLORED.end()) {
      auto existing = itr_e->second;
      // More balanced dominance: moderate preference for fewer collisions
      if (existing->c < n->c - 1 || (existing->c <= n->c && existing->g <= n->g - 1)) {
        delete n;
        continue;
      }
    }
    EXPLORED[*n] = n;
    nodes_expanded++;

    // window_sizeに到達した場合、最良のノードを記録
    if (n->t >= window_size - 1) {
      int distance_to_goal = D->get(i, n->v);
      // More balanced evaluation: consider both collisions and distance, but be more lenient
      bool is_better = false;
      if (best_node == nullptr) {
        is_better = true;
      } else if (n->c < best_collision_count - 1) {
        // Prefer if moderately fewer collisions
        is_better = true;
      } else if (n->c <= best_collision_count + 2 && distance_to_goal < best_distance) {
        // Allow slightly more collisions if closer to goal
        is_better = true;
      }
      
      if (is_better) {
        best_collision_count = n->c;
        best_distance = distance_to_goal;
        best_node = n;
      }
      continue;  // これ以上展開しない
    }

    // ゴールに到達した場合（window_size内で）
    if (n->v == g_i) {
      if (n->c < best_collision_count || 
          (n->c == best_collision_count && n->g < best_distance)) {
        best_collision_count = n->c;
        best_distance = n->g;
        best_node = n;
      }
      // Early termination if we find a good enough path to goal
      if (n->c == 0) {
        best_node = n;
        break;
      }
      // Balanced collision acceptance for efficiency
      if (n->c <= 3 && nodes_expanded < max_nodes / 2) { 
        // Continue searching for better solutions if we haven't searched much
        continue;
      } else if (n->c <= 7) {
        // Accept solutions with moderate collisions for better efficiency
        break;
      }
    }

    // expand neighbors
    for (auto &u : n->v->neighbor) {
      for (auto &si : ST.get(u)) {
        // invalid transition
        if (si.first > n->time_end + 1) break;
        if (si.second <= n->time_start) continue;

        // Find earliest valid time with collision checking
        auto t_earliest = INT_MAX;
        int collision_penalty = 0;
        bool found_collision_free = false;
        
        for (auto t = std::max(n->t, si.first - 1);
             t <= std::min(n->time_end, si.second - 1); ++t) {
          auto collision_cost = CT->getCollisionCost(n->v, u, t);
          if (collision_cost == 0) {
            // No collision - prefer this time
            t_earliest = t + 1;
            collision_penalty = 0;
            found_collision_free = true;
            break;
          } else if (!found_collision_free) {
            // Collision exists - allow with penalty if no collision-free option found
            if (t_earliest == INT_MAX) {
              t_earliest = t + 1;
              collision_penalty = 1;  // Count as one collision
            }
          }
        }
        
        if (t_earliest >= INT_MAX || t_earliest >= window_size) continue;

        // valid neighbor with soft constraint handling
        auto g_val = n->g + (t_earliest - n->t);
        auto c_val = n->c + collision_penalty;  // Add collision penalty
        auto f_val = g_val + D->get(i, u);
        auto n_new = new SIPPSNode(++node_id, si, u, t_earliest, g_val, c_val, f_val, n);

        auto itr = EXPLORED.find(*n_new);
        if (itr != EXPLORED.end()) {
          auto existing = itr->second;
          // Skip if existing node dominates new node
          if (existing->c < n_new->c || (existing->c == n_new->c && existing->g <= n_new->g)) {
            delete n_new;
            continue;
          }
        }
        OPEN.push(n_new);
      }
    }
  }

  // 最良のノードからパスを構築
  if (best_node != nullptr) {
    // backtrack
    std::vector<Vertex*> temp_path;
    auto current = best_node;
    while (current != nullptr) {
      temp_path.push_back(current->v);
      current = current->parent;
    }
    std::reverse(temp_path.begin(), temp_path.end());
    
    // window_sizeに合わせてパスを調整
    solution_path.resize(window_size);
    for (int t = 0; t < window_size; ++t) {
      if (t < static_cast<int>(temp_path.size())) {
        solution_path[t] = temp_path[t];
      } else {
        // パスが短い場合は最後のノードで埋める
        solution_path[t] = temp_path.back();
      }
    }
  }

  // memory management
  while (!OPEN.empty()) {
    delete OPEN.top();
    OPEN.pop();
  }
  for (auto iter : EXPLORED) delete iter.second;
  return solution_path;
}

std::ostream &operator<<(std::ostream &os, const SINode *n)
{
  os << "f=" << std::setw(4) << n->f << ", v=" << std::setw(6) << n->v
     << ", t=" << std::setw(4) << n->t << ", si: [" << std::setw(4)
     << n->time_start << ", " << std::setw(4)
     << ((n->time_end < INT_MAX - 1) ? std::to_string(n->time_end) : "inf")
     << "]";
  return os;
}
