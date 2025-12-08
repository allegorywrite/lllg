#include "../include/instance.hpp"

Instance::~Instance()
{
  if (delete_graph_after_used) delete G;
}

Instance::Instance(Graph *_G, const Config &_starts, const Config &_goals)
    : G(_G),
      starts(_starts),
      goals(_goals),
      N(starts.size()),
      delete_graph_after_used(false)
{
  for (auto& v : G->V) {
    v->accessed_by_agents.resize(N, false);
  }
}

Instance::Instance(const std::string &map_filename,
                   const std::vector<int> &start_indexes,
                   const std::vector<int> &goal_indexes)
    : G(new Graph(map_filename)),
      starts(Config()),
      goals(Config()),
      N(start_indexes.size()),
      delete_graph_after_used(true)
{
  for (auto& v : G->V) {
    v->accessed_by_agents.resize(N, false);
  }

  for (auto k : start_indexes) starts.push_back(G->U[k]);
  for (auto k : goal_indexes) goals.push_back(G->U[k]);
}

// for load instance
static const std::regex r_instance =
    std::regex(R"(\d+\t.+\.map\t\d+\t\d+\t(\d+)\t(\d+)\t(\d+)\t(\d+)\t.+)");

Instance::Instance(const std::string &scen_filename,
                   const std::string &map_filename, const int _N)
    : G(new Graph(map_filename)),
      starts(Config()),
      goals(Config()),
      N(_N),
      delete_graph_after_used(true)
{
  for (auto& v : G->V) {
    v->accessed_by_agents.resize(N, false);
  }

  // load start-goal pairs
  std::ifstream file(scen_filename);
  if (!file) {
    info(0, 0, scen_filename, " is not found");
    return;
  }
  std::string line;
  std::smatch results;

  // Resize predefined goals vector
  agent_predefined_goals.resize(N);
  int agent_cnt = 0;

  while (getline(file, line)) {
    // for CRLF coding
    if (*(line.end() - 1) == 0x0d) line.pop_back();

    if (std::regex_match(line, results, r_instance)) {
      auto x_s = std::stoi(results[1].str());
      auto y_s = std::stoi(results[2].str());
      auto x_g = std::stoi(results[3].str());
      auto y_g = std::stoi(results[4].str());
      if (x_s < 0 || G->width <= x_s || x_g < 0 || G->width <= x_g) continue;
      if (y_s < 0 || G->height <= y_s || y_g < 0 || G->height <= y_g) continue;
      auto s = G->U[G->width * y_s + x_s];
      auto g = G->U[G->width * y_g + x_g];
      if (s == nullptr || g == nullptr) continue;
      starts.push_back(s);
      goals.push_back(g);

      // Parse additional goals if present
      // The regex only captures the first start/goal pair. We need to parse the rest of the line.
      // Format: <bucket> <map> <width> <height> <x_s> <y_s> <x_g> <y_g> <dist> <x_g2> <y_g2> ...
      // We can find the position after the first goal and parse remaining numbers.
      
      // Find the position of the first goal coordinates in the line
      // The regex matches the whole line, so we can't easily find "where the rest starts" using just the match results for the suffix.
      // Instead, let's use a stringstream to parse the line manually after the initial match validation.
      
      std::stringstream ss(line);
      std::string segment;
      std::vector<std::string> tokens;
      while(std::getline(ss, segment, '\t')) {
        tokens.push_back(segment);
      }
      
      // Standard format has 9 columns. If more, they are extra goals.
      // Col 4: x_s, Col 5: y_s, Col 6: x_g, Col 7: y_g, Col 8: dist
      // Extra goals start from Col 9 (0-indexed) if we assume standard format ends at 8.
      // Let's check the tokens.
      // 0: bucket, 1: map, 2: w, 3: h, 4: xs, 5: ys, 6: xg, 7: yg, 8: dist
      // 9: xg2, 10: yg2, ...
      
      if (tokens.size() > 9) {
        for (size_t k = 8; k + 1 < tokens.size(); k += 2) {
           try {
             int x = std::stoi(tokens[k]);
             int y = std::stoi(tokens[k+1]);
             if (x >= 0 && x < G->width && y >= 0 && y < G->height) {
               auto v = G->U[G->width * y + x];
               if (v != nullptr) {
                 agent_predefined_goals[agent_cnt].push_back(v);
               }
             }
           } catch (...) {
             break;
           }
        }
      }
      agent_cnt++;
    }

    if (starts.size() == N) break;
  }
}

Instance::Instance(const std::string &map_filename, const int _N,
                   const int seed)
    : G(new Graph(map_filename)),
      starts(Config()),
      goals(Config()),
      N(_N),
      delete_graph_after_used(true)
{
  for (auto& v : G->V) {
    v->accessed_by_agents.resize(N, false);
  }
  
  auto MT = std::mt19937(seed);
  // random assignment
  const auto K = G->size();

  // set starts
  auto s_indexes = std::vector<int>(K);
  std::iota(s_indexes.begin(), s_indexes.end(), 0);
  std::shuffle(s_indexes.begin(), s_indexes.end(), MT);
  int i = 0;
  while (true) {
    if (i >= K) return;
    starts.push_back(G->V[s_indexes[i]]);
    if (starts.size() == N) break;
    ++i;
  }

  // set goals
  auto g_indexes = std::vector<int>(K);
  std::iota(g_indexes.begin(), g_indexes.end(), 0);
  std::shuffle(g_indexes.begin(), g_indexes.end(), MT);
  int j = 0;
  while (true) {
    if (j >= K) return;
    goals.push_back(G->V[g_indexes[j]]);
    if (goals.size() == N) break;
    ++j;
  }
}

bool Instance::is_valid(const int verbose) const
{
  if (N != starts.size() || N != goals.size()) {
    info(1, verbose, "invalid N, check instance");
    return false;
  }
  return true;
}
