# LLLG: Lifelong LaCAM with Local Guidance for Lifelong MAPF

[<img src="https://img.shields.io/badge/arxiv-2605.16855-990000" alt="Arxiv">](http://arxiv.org/abs/2605.16855)

**LLLG is the lifelong version of LG-LaCAM (AAAI-26)**, bringing local guidance to a receding-horizon planning framework for **Lifelong Multi-Agent Pathfinding (LMAPF)**. Local guidance supplies each agent with informative spatiotemporal cues that help mitigate congestion, reduce waiting, and improve short-horizon coordination in dense multi-agent environments. While local guidance has recently shown strong empirical benefits in one-shot MAPF, this work lifts the same idea to the lifelong setting, where agents continuously receive new tasks and must replan under strict real-time constraints. Our method scales effectively and maintains strong performance even in compact, dense environments with many tightly packed agents, yielding higher throughput and surpassing the prior state-of-the-art, thereby pushing the frontier for real-time lifelong MAPF.


The paper will appear at SoCS-26.


<table>
  <tr>
    <td align="center">
      <img alt="LaCAM baseline" src="media/lifelong_lacam.gif" width="95%"/> <p><b>LaCAM</b> — The baseline configuration-based LMAPF solver.
      </p>
    </td>
    <td align="center">
      <img alt="LLLG enhanced" src="media/LLLG_guidance.gif" width="100%"/> <p><b>LLLG</b> — Lifelong LaCAM with Local Guidance for LMAPF.
      </p>
    </td>
  </tr>
</table>
<p align="center">
  <i>Visualization of 400 agents navigating a multi-room environment.<br> LLLG visibly alleviates local congestion and accelerates overall throughput compared to lifelong LaCAM.</i>
</p>

## Citation
If you find this work to be useful in your research, please consider citing:

```bibtex
@article{arita2026local,
  title={Lifelong LaCAM with Local Guidance for Lifelong MAPF},
  author={Arita, Tomoki and Okumura, Keisuke},
  journal={arXiv preprint arxiv-2605.16855},
  year={2026}
}
```

## Building

All you need is [CMake](https://cmake.org/) (≥v3.16).
The code is written in C++(17).

First, clone this repo with submodules.

```sh
git clone --recursive {this repo}
```

Then, build the project.

```sh
cmake -B build && make -C build -j4
```

## Usage

```sh
build/main -i assets/random-32-32-10-random-1.scen -m assets/random-32-32-10.map -N 400 -v 2 --lg --lg_window 20 --lacam_horizon 10 --lifelong -S 10
```

The result will be saved in `build/result.txt`.

You can find details of all parameters with:

```sh
build/main --help
```

## Visualizer

This repository is compatible with [allegorywrite@mapf-visualizer](https://github.com/allegorywrite/mapf-visualizer).
For example,

```sh
mapf-visualizer assets/random-32-32-10.map build/result.txt --lifelong
```

## Notes

### install pre-commit for formatting

```sh
pre-commit install
```

### simple test

```sh
ctest --test-dir ./build
```
