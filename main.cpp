#include <argparse/argparse.hpp>
#include <planner.hpp>
#include <post_processing.hpp>
#include <lifelong_runner.hpp>
#include <chrono>
#include <random>

int main(int argc, char *argv[])
{
  // arguments parser
  auto program = argparse::ArgumentParser("lacam", "0.1.0");
  program.add_argument("-m", "--map").help("map file").required();
  program.add_argument("-i", "--scen").help("scenario file").default_value("");
  program.add_argument("-N", "--num")
      .help("number of agents")
      .scan<'d', int>()
      .required();
  program.add_argument("-s", "--seed")
      .help("seed")
      .scan<'d', int>()
      .default_value(0);
  program.add_argument("-v", "--verbose")
      .help("verbose")
      .scan<'d', int>()
      .default_value(0);
  program.add_argument("-t", "--time_limit_sec")
      .help("time limit sec")
      .scan<'g', float>()
      .default_value(3.0f);
  // Horizon control for LaCAM
  program.add_argument("--lacam_horizon")
      .help("limit LaCAM high-level depth (solution steps); -1 for unlimited")
      .scan<'d', int>()
      .default_value(-1);
  program.add_argument("--relax_goal")
      .help("relax goal condition: each agent must visit its goal at least once (not necessarily simultaneously)")
      .default_value(false)
      .implicit_value(true);
  // Lifelong control
  program.add_argument("-S", "--steps")
      .help("number of execution steps (lifelong mode)")
      .scan<'d', int>()
      .default_value(-1);
  program.add_argument("--lifelong")
      .help("enable Lifelong LaCAM (replan each step and execute one step)")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("-o", "--output")
      .help("output file")
      .default_value("./build/result.txt");
  program.add_argument("-l", "--log_short")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("--log-all-step")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("--lifelong_seed_mode")
      .help("initialization source for Lifelong LaCAM local guide: plan, local_guide, none")
      .default_value(std::string("local_guide"));

  // solver parameters
  program.add_argument("--no_pibt_swap")
      .default_value(false)
      .implicit_value(true);

  // PIBT tie-breaking (ported from pibt-tiebreaking)
  program.add_argument("--no_hindrance")
      .help("disable PIBT next-step hindrance tie-break")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("--pibt_sort")
      .help("PIBT tie-break mode: 0=legacy(random), 1=prefer-free, 2=hindrance, 3=random+hindrance")
      .scan<'d', int>()
      .default_value(0);

  program.add_argument("--lg").default_value(false).implicit_value(true);
  program.add_argument("--lg_num_refine").scan<'d', int>().default_value(1);
  program.add_argument("--lg_window").scan<'d', int>().default_value(10);
  program.add_argument("--lg_collision_cost")
      .help("collision cost for dynamic window adjustment")
      .scan<'g', float>()
      .default_value(2.0f);
  program.add_argument("--lg_collision_cost_order")
      .help("collision cost order for dynamic window adjustment")
      .scan<'g', float>()
      .default_value(1e-7f);

  program.add_argument("--gg_margin").scan<'d', int>().default_value(10);
  program.add_argument("--gg").default_value(false).implicit_value(true);

  program.add_argument("--lns").default_value(false).implicit_value(true);
  program.add_argument("--plns_num_refiners").scan<'d', int>().default_value(8);
  program.add_argument("--lns_nb_strategy")
      .help("LNS neighborhood selection: block, random, intersection, randomwalk")
      .default_value(std::string("block"));
  program.add_argument("--lns_nb_size")
      .help("LNS neighborhood size; -1 uses legacy size sampling")
      .scan<'d', int>()
      .default_value(-1);
  program.add_argument("--lns_relax_objective")
      .help("LNS objective under --relax_goal: first_goal_time, t1")
      .default_value(std::string("first_goal_time"));

  // deterministic
  program.add_argument("-d", "--deterministic")
      .help("disable randomness in solver (PIBT/LocalGuide)")
      .default_value(false)
      .implicit_value(true);

  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    std::exit(1);
  }

  // setup instance
  const auto verbose = program.get<int>("verbose");
  const auto time_limit_sec = program.get<float>("time_limit_sec");
  const auto scen_name = program.get<std::string>("scen");
  const auto seed = program.get<int>("seed");
  const auto map_name = program.get<std::string>("map");
  const auto output_name = program.get<std::string>("output");
  const auto log_short = program.get<bool>("log_short");
  const auto N = program.get<int>("num");
  const auto seed_mode_str = program.get<std::string>("lifelong_seed_mode");
  LifelongSeedMode lifelong_seed_mode = LifelongSeedMode::PrevLocalGuide;
  if (seed_mode_str == "plan") {
    lifelong_seed_mode = LifelongSeedMode::PrevPlan;
  } else if (seed_mode_str == "local_guide") {
    lifelong_seed_mode = LifelongSeedMode::PrevLocalGuide;
  } else if (seed_mode_str == "none") {
    lifelong_seed_mode = LifelongSeedMode::None;
  } else {
    std::cerr << "Unknown --lifelong_seed_mode '" << seed_mode_str
              << "'. Falling back to 'plan'." << std::endl;
  }
  const auto ins = scen_name.size() > 0 ? Instance(scen_name, map_name, N)
                                        : Instance(map_name, N, seed);
  if (!ins.is_valid(1)) return 1;

  // set hyper parameters

  // local guide
  LocalGuide::ON = program.get<bool>("lg");
//   LocalGuide::WINDOWS.resize(ins.N, program.get<int>("lg_window"));
  LocalGuide::WINDOW = program.get<int>("lg_window");
  LocalGuide::NUM_REFINE = program.get<int>("lg_num_refine");

  LocalGuide::COLLISION_COST = program.get<float>("lg_collision_cost");
  LocalGuide::COLLISION_COST_ORDER = program.get<float>("lg_collision_cost_order");
  LocalGuide::CLEAR_GOAL_FIRST = program.get<bool>("relax_goal");

  // global guide
  GlobalGuide::ON = program.get<bool>("gg");
  GlobalGuide::COST_MARGIN = program.get<int>("gg_margin");

  // pibt
  PIBT::SWAP = !program.get<bool>("no_pibt_swap");
  PIBT::NEXT_STEP_HINDRANCE = !program.get<bool>("no_hindrance");
  PIBT::SWITCH_ORDER = program.get<int>("pibt_sort");
  // LaCAM horizon
  LaCAM::STEP_LIMIT = program.get<int>("lacam_horizon");
  LaCAM::RELAX_GOAL = program.get<bool>("relax_goal");

  // lns, plns
  LNS::ON = program.get<bool>("lns");
  LNS::RELAX_GOAL_CONDITION = program.get<bool>("relax_goal");
  {
    // Only meaningful when RELAX_GOAL_CONDITION is enabled.
    const auto obj = program.get<std::string>("lns_relax_objective");
    if (!LNS::RELAX_GOAL_CONDITION || obj == "first_goal_time") {
      LNS::RELAX_OBJECTIVE_T1 = false;
    } else if (obj == "t1") {
      LNS::RELAX_OBJECTIVE_T1 = true;
    } else {
      std::cerr << "Unknown --lns_relax_objective '" << obj
                << "'. Falling back to 'first_goal_time'." << std::endl;
      LNS::RELAX_OBJECTIVE_T1 = false;
    }
  }
  PLNS::NUM_REFINERS = program.get<int>("plns_num_refiners");
  LNS::NEIGHBOR_SIZE = program.get<int>("lns_nb_size");
  {
    const auto nb = program.get<std::string>("lns_nb_strategy");
    if (nb == "block") {
      LNS::NEIGHBOR_STRATEGY = LNS::NeighborhoodStrategy::RandomBlock;
    } else if (nb == "random") {
      LNS::NEIGHBOR_STRATEGY = LNS::NeighborhoodStrategy::RandomAgents;
    } else if (nb == "intersection") {
      LNS::NEIGHBOR_STRATEGY = LNS::NeighborhoodStrategy::Intersection;
    } else if (nb == "randomwalk") {
      LNS::NEIGHBOR_STRATEGY = LNS::NeighborhoodStrategy::RandomWalk;
    } else {
      std::cerr << "Unknown --lns_nb_strategy '" << nb
                << "'. Falling back to 'block'." << std::endl;
      LNS::NEIGHBOR_STRATEGY = LNS::NeighborhoodStrategy::RandomBlock;
    }
  }

  // deterministic
  const auto deterministic = program.get<bool>("deterministic");
  PIBT::DETERMINISTIC = deterministic;
  LocalGuide::DETERMINISTIC = deterministic;

  // solve
//   const auto use_sipp = program.get<bool>("use_sipp");
  const auto lifelong = program.get<bool>("lifelong");
  const auto steps_limit = program.get<int>("steps");

  if (!lifelong) {
    const auto deadline = Deadline(time_limit_sec * 1000);
    auto result = solve_with_timing(ins, verbose - 1, &deadline, seed);
    auto solution = result.solution;
    auto solution_init = result.solution_init;
    auto lacam = result.lacam;
    const auto comp_time_ms = deadline.elapsed_ms();
    const auto comp_time_init_ms = result.comp_time_init_ms;

    // failure
    if (solution.empty()) {
      info(1, verbose, &deadline, "failed to solve");
      delete lacam;
      return 1;
    }

    // check feasibility
    const auto relax_goal = program.get<bool>("relax_goal");
    const bool feasible = relax_goal
                              ? is_feasible_solution_relax_goal(ins, solution, verbose)
                              : is_feasible_solution(ins, solution, verbose);
    if (!feasible) {
      info(0, verbose, &deadline, "invalid solution");
      delete lacam;
      return 1;
    }

    // post processing
    const char* solved_label = relax_goal ? "solved" : nullptr;
    print_stats(verbose, &deadline, ins, solution, comp_time_ms, solved_label);
    // Only pass comp_time_init_ms if LNS was used
    const auto comp_time_init_to_log = LNS::ON ? comp_time_init_ms : -1.0;
    const auto empty_solution = Solution();
    const auto& solution_init_to_log = LNS::ON ? solution_init : empty_solution;
    if (relax_goal) {
      const bool solved_override = true;
      make_log(ins, solution, output_name, comp_time_ms, map_name, seed, log_short,
               &lacam->local_guide, comp_time_init_to_log, solution_init_to_log,
               /*lifelong_goals_history=*/nullptr, /*goal_change_count=*/0,
               /*local_guidance_history=*/nullptr, &solved_override);
    } else {
      make_log(ins, solution, output_name, comp_time_ms, map_name, seed, log_short,
               &lacam->local_guide, comp_time_init_to_log, solution_init_to_log);
    }
    delete lacam;
    return 0;
  }

  // Lifelong LaCAM mode (outer loop: replan each step)
  const auto log_all_step = program.get<bool>("log-all-step");
  return run_lifelong(ins, verbose, time_limit_sec, seed, steps_limit,
                      output_name, map_name, log_short, lifelong_seed_mode, log_all_step);
}
