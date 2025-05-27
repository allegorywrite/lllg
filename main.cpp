#include <argparse/argparse.hpp>
#include <planner.hpp>

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
  program.add_argument("-o", "--output")
      .help("output file")
      .default_value("./build/result.txt");
  program.add_argument("-l", "--log_short")
      .default_value(false)
      .implicit_value(true);

  // solver parameters
  program.add_argument("--no_pibt_swap")
      .default_value(false)
      .implicit_value(true);

  program.add_argument("--lg").default_value(false).implicit_value(true);
  program.add_argument("--lg_num_refine").scan<'d', int>().default_value(1);
  program.add_argument("--lg_window").scan<'d', int>().default_value(10);
  program.add_argument("--lg_dynamic_window")
      .help("enable dynamic window size adjustment")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("--lg_window_update_type")
      .help("window update type (0: ACCESS_COUNT, 1: OCCUPANCY, 2: COLLISION)")
      .scan<'d', int>()
      .default_value(1);  // デフォルトは占有率ベース
  program.add_argument("--lg_min_window")
      .help("minimum window size for dynamic window")
      .scan<'d', int>()
      .default_value(5);
  program.add_argument("--lg_max_window")
      .help("maximum window size for dynamic window")
      .scan<'d', int>()
      .default_value(10);
  program.add_argument("--lg_collision_cost")
      .help("collision cost for dynamic window adjustment")
      .scan<'g', float>()
      .default_value(1.0f);
  program.add_argument("--lg_collision_cost_order")
      .help("collision cost order for dynamic window adjustment")
      .scan<'g', float>()
      .default_value(1e-7f);
  program.add_argument("--lg_global_guide_first_order")
      .help("global guide first order for dynamic window adjustment")
      .scan<'g', float>()
      .default_value(1e-2f);
  program.add_argument("--lg_global_guide_second_order")
      .help("global guide second order for dynamic window adjustment")
      .scan<'g', float>()
      .default_value(1e-4f);
  program.add_argument("--lg_occupancy_threshold")
      .help("occupancy threshold for dynamic window adjustment")
      .scan<'g', float>()
      .default_value(0.25f);
  program.add_argument("--lg_collision_threshold")
      .help("collision threshold for dynamic window adjustment")
      .scan<'g', float>()
      .default_value(0.5f);
  program.add_argument("--lg_access_count_threshold")
      .help("access count threshold for dynamic window adjustment")
      .scan<'g', float>()
      .default_value(8.0f);

  program.add_argument("--gg_margin").scan<'d', int>().default_value(10);
  program.add_argument("--gg").default_value(false).implicit_value(true);

  program.add_argument("--lns").default_value(false).implicit_value(true);
  program.add_argument("--plns_num_refiners").scan<'d', int>().default_value(8);
  program.add_argument("--use_sipp")
      .help("use SIPP for local guide")
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
  const auto ins = scen_name.size() > 0 ? Instance(scen_name, map_name, N)
                                        : Instance(map_name, N, seed);
  if (!ins.is_valid(1)) return 1;

  // set hyper parameters

  // local guide
  LocalGuide::ON = program.get<bool>("lg");
  LocalGuide::WINDOWS.resize(ins.N, program.get<int>("lg_window"));
  LocalGuide::NUM_REFINE = program.get<int>("lg_num_refine");
  LocalGuide::DYNAMIC_WINDOW = program.get<bool>("lg_dynamic_window");
  
  // ウィンドウ更新タイプの設定
  int window_update_type = program.get<int>("lg_window_update_type");
  switch (window_update_type) {
    case 0:
      LocalGuide::WINDOW_UPDATE_TYPE = WindowUpdateType::ACCESS_COUNT;
      break;
    case 1:
      LocalGuide::WINDOW_UPDATE_TYPE = WindowUpdateType::OCCUPANCY;
      break;
    case 2:
      LocalGuide::WINDOW_UPDATE_TYPE = WindowUpdateType::COLLISION;
      break;
    default:
      std::cerr << "Invalid window update type: " << window_update_type << std::endl;
      std::exit(1);
  }
  
  LocalGuide::MIN_WINDOW = program.get<int>("lg_min_window");
  LocalGuide::MAX_WINDOW = program.get<int>("lg_max_window");
  LocalGuide::COLLISION_COST = program.get<float>("lg_collision_cost");
  LocalGuide::COLLISION_COST_ORDER = program.get<float>("lg_collision_cost_order");
  LocalGuide::GLOBAL_GUIDE_FIRST_ORDER = program.get<float>("lg_global_guide_first_order");
  LocalGuide::GLOBAL_GUIDE_SECOND_ORDER = program.get<float>("lg_global_guide_second_order");
  LocalGuide::OCCUPANCY_THRESHOLD = program.get<float>("lg_occupancy_threshold");
  LocalGuide::COLLISION_THRESHOLD = program.get<float>("lg_collision_threshold");
  LocalGuide::ACCESS_COUNT_THRESHOLD = program.get<float>("lg_access_count_threshold");

  // global guide
  GlobalGuide::ON = program.get<bool>("gg");
  GlobalGuide::COST_MARGIN = program.get<int>("gg_margin");

  // pibt
  PIBT::SWAP = !program.get<bool>("no_pibt_swap");

  // lns, plns
  LNS::ON = program.get<bool>("lns");
  PLNS::NUM_REFINERS = program.get<int>("plns_num_refiners");

  // solve
  const auto use_sipp = program.get<bool>("use_sipp");
  const auto deadline = Deadline(time_limit_sec * 1000);
  auto [solution, lacam] = solve(ins, verbose - 1, &deadline, seed, use_sipp);
  const auto comp_time_ms = deadline.elapsed_ms();

  // failure
  if (solution.empty()) {
    info(1, verbose, &deadline, "failed to solve");
    delete lacam;
    return 1;
  }

  // check feasibility
  if (!is_feasible_solution(ins, solution, verbose)) {
    info(0, verbose, &deadline, "invalid solution");
    delete lacam;
    return 1;
  }

  // post processing
  print_stats(verbose, &deadline, ins, solution, comp_time_ms);
  make_log(ins, solution, output_name, comp_time_ms, map_name, seed, log_short, &lacam->local_guide);
  delete lacam;
  return 0;
}
