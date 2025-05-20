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

  WSPPNodes wspp_nodes;
  std::vector<WSPPNodes> CLOSED;
  Config Q_to;

  // hyper parameters
  static bool ON;
  static int WINDOW;
  static int NUM_REFINE;

  // guidance
  GlobalGuide* global_guide;
  bool use_sipp_; // Flag to use SIPP

  LocalGuide(const Instance* _ins, DistTable* _D, int seed = 0,
             GlobalGuide* _global_guide = nullptr, bool _use_sipp = false);
  ~LocalGuide();

  void construct(const Config& Q_from, const std::vector<int>& order);
  LocalHeuristic get(const int i, Vertex* v);
};
