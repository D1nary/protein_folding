#pragma once

// Replica Exchange Monte Carlo (Thachuk et al., 2007), variant REMC_m: the
// local search mixes pull moves and VSHD moves.
// See notes/samplings.md.

#include <cstdint>
#include <vector>

#include "hp/conformation.hpp"
#include "hp/sample_writer.hpp"

namespace hp {

// Simulation parameters, filled from Python with the values of the
// configuration file (no default values here).
struct RemcParams {
    int n_replicas;      // M: number of replicas (at least 2)
    double t_min;        // T_1: temperature of the coldest replica (k_B = 1)
    double t_max;        // T_M: temperature of the hottest replica
    int local_steps;     // phi: local Monte Carlo steps between two swap attempts
    int n_cycles;        // number of cycles (local sampling + swap)
    double rho;          // probability that a local step uses a pull move
    std::uint64_t seed;  // seed of the random number generator
};

struct RemcResult {
    Conformation best;                // lowest-energy conformation visited
    int best_energy;                  // its energy
    std::vector<double> temperatures; // T_1 < ... < T_M
    std::vector<int> final_energies;  // energy of each replica at the end
    long long swap_attempts;          // proposed exchanges between replicas
    long long swap_accepted;          // accepted exchanges
};

// Temperatures of the replicas, linearly spaced between t_min and t_max:
// T_k = t_min + k * (t_max - t_min) / (n_replicas - 1), k = 0 ... n_replicas-1.
std::vector<double> temperature_ladder(int n_replicas, double t_min, double t_max);

// Runs the REMC simulation: all the replicas start from `initial`.
//
// Each cycle performs `local_steps` Metropolis steps on every replica, then
// proposes swaps between replicas at adjacent temperatures, alternating
// between the pairs (1,2), (3,4), ... and (2,3), (4,5), ... (even-odd scheme).
//
// Local step: the neighbourhood is drawn first (pull moves with probability
// rho, VSHD otherwise), then a residue uniformly at random, then one of the
// feasible moves at that residue uniformly at random. If there is none, the
// step leaves the conformation unchanged.
//
// The best conformation is updated after every accepted move, not only at the
// end of the local search.
//
// If `recorder` is given, it receives the trajectories of the walkers and the
// snapshots of the replicas (sample_writer.hpp). Recording does not use the
// random generator: the simulation is the same with or without it.
//
// Throws std::invalid_argument if the parameters are not valid.
RemcResult run_remc(const Conformation& initial, const RemcParams& params,
                    SampleRecorder* recorder = nullptr);

}  // namespace hp
