/*
 * windowed soft PP
 */

#pragma once
#include "collision_table.hpp"
#include "dist_table.hpp"
#include "global_guide.hpp"
#include "graph.hpp"
#include "instance.hpp"
#include "sipp.hpp"  // Add SIPP header
#include "utils.hpp"

// windowed soft prioritized planning
struct WSPPNode {
  int when;
  Vertex* where;
  float g;
  float h;
  float f;
  WSPPNode* parent;

  WSPPNode(int _when = 0, Vertex* _where = nullptr, int _g = 0, int _h = 0,
           float _f = 0, WSPPNode* _parent = nullptr)
      : when(_when), where(_where), g(_g), h(_h), f(g + h), parent(_parent)
  {
  }
  ~WSPPNode(){};
};
using WSPPNodes = std::vector<WSPPNode*>;

using LocalHeuristic = int;

// Define window size update determination types
enum class WindowUpdateType {
  ACCESS_COUNT,  // Access count based
  OCCUPANCY,     // Occupancy rate based
  COLLISION      // Collision amount based
};

struct LocalGuide {
  const Instance* ins;
  std::mt19937 MT;

  // solver utils
  const int N;  // number of agents
  const int V_size;
  DistTable* D;

  // specific to solver
  CollisionTable CT;
  std::vector<Path> guide_paths;
  
  // // Helper functions for parallel computation
  // Path computeGuidePath(int agent_id, const Config& Q_from);
  // Path computeGuidePathCorrect(int agent_id, const Config& Q_from);
  // Path computeGuidePathWithCT(int agent_id, const Config& Q_from, CollisionTable& ct);

  // Save reference trajectory history
  std::vector<std::vector<Path>> guide_paths_history;  // History of reference trajectories at each step
  int current_step;  // Current step count

  WSPPNodes wspp_nodes;
  std::vector<WSPPNodes> CLOSED;
  Config Q_to;

  // hyper parameters
  static bool ON;
  static int WINDOW;
  static int NUM_REFINE;
  static float COLLISION_COST;  // Collision cost
  static float COLLISION_COST_ORDER;  // Collision cost coefficient
  // guidance
  GlobalGuide* global_guide;
  std::vector<float> cached_collision_costs;  // Cache collision costs during A* search
  std::vector<int> step_counters;          // Step counters for each agent for k-step update

  LocalGuide(const Instance* _ins, DistTable* _D, int seed = 0,
             GlobalGuide* _global_guide = nullptr);
  ~LocalGuide();

  void construct(const Config& Q_from, const std::vector<int>& order);
  LocalHeuristic get(const int i, Vertex* v);

  // History-related methods
  void clear_history();  // Clear history
  void save_current_paths();  // Save current reference trajectory to history
  const std::vector<Path>& get_paths_at_step(int step) const;  // Get reference trajectory at specific step
  int get_history_size() const;  // Get history size

  // Methods for cooperation with HNode
  void set_guide_paths(const std::vector<Path>& paths);  // Set reference trajectory from HNode
  std::vector<Path> get_current_guide_paths() const;     // Get current reference trajectory
  void reconstruct_solution_paths(const std::vector<std::vector<Path>>& solution_paths);  // Reconstruct reference trajectory corresponding to solution

  // Helper function for k-step update determination
  bool should_update_guide_path(int agent_id);
};
