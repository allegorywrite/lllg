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
      .help("enable dynamic window size based on occupancy")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("--lg_min_window")
      .help("minimum window size for dynamic window")
      .scan<'d', int>()
      .default_value(5);
  program.add_argument("--lg_max_window")
      .help("maximum window size for dynamic window")
      .scan<'d', int>()
      .default_value(20);
  program.add_argument("--lg_occupancy_threshold")
      .help("occupancy threshold for dynamic window adjustment")
      .scan<'g', float>()
      .default_value(0.3f);

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
  LocalGuide::MIN_WINDOW = program.get<int>("lg_min_window");
  LocalGuide::MAX_WINDOW = program.get<int>("lg_max_window");
  LocalGuide::OCCUPANCY_THRESHOLD = program.get<float>("lg_occupancy_threshold");

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
