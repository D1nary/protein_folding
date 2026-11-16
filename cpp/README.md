# `cpp/`

C++17 core of the project: HP model on a 2D lattice, moves, REMC, exact enumeration and the conformation-graph oracle. It does not depend on Python and neither reads nor writes files: it receives data from Python and returns results in memory. All the code lives in the `hp` namespace.

| Folder | Contents |
|---|---|
| `include/hp/` | Headers: they declare the interface of each component. |
| `src/` | Implementations, one `.cpp` for each header with the same name. |
| `bindings/` | `bindings.cpp`: pybind11 wrapper that creates the Python module `hpfold._core`. |

## Components

| Header | Role |
|---|---|
| `conformation.hpp` | `Conformation` class (sequence and positions), validity, energy (−1 for each non-consecutive H-H contact), application of a `Move`. |
| `vshd_moves.hpp` | VSHD moves: end, corner and crankshaft. |
| `pull_moves.hpp` | Pull moves from Lesh et al. (2003), internal and at the ends. |
| `remc.hpp` | `run_remc`: Replica Exchange Monte Carlo, `REMC_m` variant of Thachuk et al. (2007); `RemcParams`, `RemcResult`, temperature ladder. |
| `exact.hpp` | `exact_ground_state`: exact minimum energy by enumeration of self-avoiding walks (DFS with branch and bound). |
| `sample_writer.hpp` | `SampleRecorder`: records replica trajectories and snapshots in memory during the REMC (used for the datasets). |
| `canonical.hpp` | Canonical form (turns) and 64-bit key, invariant under translations, rotations and reflections. |
| `actions.hpp` | Action space with 6N + 11 slots (residue, move type). |
| `state_graph.hpp` | `StateGraph` and `Oracle`: state enumeration, move graph, BFS distances from the native state, optimal paths, trajectory cleanup. |

## Build and linking

It is not compiled by hand: [`CMakeLists.txt`](../CMakeLists.txt) builds the static library `hp_core` from `src/` and links it into the `_core` module; it is invoked by `pip install -e .` through scikit-build-core. After every change to the C++ code:

```bash
pip install -e ".[dev,ml]"
```

From Python it is used as `from hpfold import _core` (for example `_core.run_remc` in `python/hpfold/simulation.py`).
