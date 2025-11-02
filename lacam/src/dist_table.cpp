#include "../include/dist_table.hpp"

DistTable::DistTable(const Instance &ins)
    : K(ins.G->V.size()), table(ins.N, std::vector<int>(K, K))
{
  setup(&ins);
}

DistTable::DistTable(const Instance *ins)
    : K(ins->G->V.size()), table(ins->N, std::vector<int>(K, K))
{
  setup(ins);
}

void DistTable::setup(const Instance *ins)
{
  // Capture pointer by value to avoid dangling references, and wait for all tasks
  const Instance* local = ins;
  auto bfs = [this, local](const int i) {
    auto g_i = local->goals[i];
    auto Q = std::queue<Vertex *>({g_i});
    table[i][g_i->id] = 0;
    while (!Q.empty()) {
      auto n = Q.front();
      Q.pop();
      const int d_n = table[i][n->id];
      for (auto &m : n->neighbor) {
        const int d_m = table[i][m->id];
        if (d_n + 1 >= d_m) continue;
        table[i][m->id] = d_n + 1;
        Q.push(m);
      }
    }
  };

  auto pool = std::vector<std::future<void>>();
  pool.reserve(local->N);
  for (size_t i = 0; i < local->N; ++i) {
    pool.emplace_back(std::async(std::launch::async, bfs, static_cast<int>(i)));
  }
  // Ensure all BFS computations complete before returning
  for (auto &f : pool) f.get();
}

int DistTable::get(const int i, const int v_id) { return table[i][v_id]; }

int DistTable::get(const int i, const Vertex *v) { return get(i, v->id); }
