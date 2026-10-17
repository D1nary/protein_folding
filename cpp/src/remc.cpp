#include "hp/remc.hpp"

#include <cmath>
#include <random>
#include <stdexcept>
#include <utility>

#include "hp/pull_moves.hpp"
#include "hp/vshd_moves.hpp"

namespace hp {

namespace {

struct Replica {
    Conformation conformation;
    int energy;
    int walker;  // follows the conformation through the swaps
};

void check_params(const RemcParams& p) {
    if (p.n_replicas < 2) {
        throw std::invalid_argument("n_replicas must be at least 2");
    }
    if (!(p.t_min > 0.0) || !(p.t_max > p.t_min)) {
        throw std::invalid_argument("temperatures must satisfy 0 < t_min < t_max");
    }
    if (p.local_steps < 1 || p.n_cycles < 1) {
        throw std::invalid_argument("local_steps and n_cycles must be at least 1");
    }
    if (!(p.rho >= 0.0 && p.rho <= 1.0)) {
        throw std::invalid_argument("rho must be between 0 and 1");
    }
}

}  // namespace

std::vector<double> temperature_ladder(int n_replicas, double t_min, double t_max) {
    if (n_replicas < 2) {
        throw std::invalid_argument("n_replicas must be at least 2");
    }
    std::vector<double> temperatures(n_replicas);
    for (int k = 0; k < n_replicas; ++k) {
        temperatures[k] = t_min + k * (t_max - t_min) / (n_replicas - 1);
    }
    return temperatures;
}

RemcResult run_remc(const Conformation& initial, const RemcParams& params,
                    SampleRecorder* recorder) {
    check_params(params);

    // A single generator for the whole simulation: the replicas are advanced
    // one after the other, so the same seed gives the same run.
    std::mt19937_64 rng(params.seed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::uniform_int_distribution<int> random_residue(0, initial.size() - 1);

    const std::vector<double> temperatures =
        temperature_ladder(params.n_replicas, params.t_min, params.t_max);

    // replicas[i] is always the conformation at temperature temperatures[i]:
    // a swap exchanges the conformations, the temperatures stay in place.
    std::vector<Replica> replicas(params.n_replicas, Replica{initial, initial.energy(), 0});
    for (int r = 0; r < params.n_replicas; ++r) {
        replicas[r].walker = r;
    }
    if (recorder != nullptr) {
        recorder->start(params.n_replicas, initial);
    }

    RemcResult result{initial, initial.energy(), temperatures, {}, 0, 0};

    int offset = 0;
    for (int cycle = 0; cycle < params.n_cycles; ++cycle) {
        // Local Monte Carlo search on every replica.
        for (int r = 0; r < params.n_replicas; ++r) {
            Replica& replica = replicas[r];
            const double temperature = temperatures[r];

            for (int step = 0; step < params.local_steps; ++step) {
                const bool use_pull = uniform(rng) < params.rho;
                const int k = random_residue(rng);
                const std::vector<Move> moves = use_pull ? pull_moves(replica.conformation, k)
                                                         : vshd_moves(replica.conformation, k);
                if (moves.empty()) {
                    continue;
                }
                std::uniform_int_distribution<std::size_t> pick(0, moves.size() - 1);
                const Move undo = replica.conformation.apply(moves[pick(rng)]);

                // Metropolis criterion.
                const int new_energy = replica.conformation.energy();
                const int delta = new_energy - replica.energy;
                if (delta <= 0 || uniform(rng) < std::exp(-delta / temperature)) {
                    replica.energy = new_energy;
                    if (recorder != nullptr) {
                        recorder->record_move(replica.walker, replica.conformation);
                    }
                    if (new_energy < result.best_energy) {
                        result.best_energy = new_energy;
                        result.best = replica.conformation;
                    }
                } else {
                    replica.conformation.apply(undo);
                }
            }
        }

        if (recorder != nullptr && recorder->wants_snapshot(cycle)) {
            for (int r = 0; r < params.n_replicas; ++r) {
                recorder->record_snapshot(cycle, r, replicas[r].walker, replicas[r].energy,
                                          replicas[r].conformation);
            }
        }

        // Replica exchange between adjacent temperatures.
        for (int i = offset; i + 1 < params.n_replicas; i += 2) {
            const int j = i + 1;
            const double beta_i = 1.0 / temperatures[i];
            const double beta_j = 1.0 / temperatures[j];
            const double delta = (beta_j - beta_i) * (replicas[i].energy - replicas[j].energy);
            ++result.swap_attempts;
            if (delta <= 0.0 || uniform(rng) <= std::exp(-delta)) {
                std::swap(replicas[i], replicas[j]);
                ++result.swap_accepted;
            }
        }
        offset = 1 - offset;
    }

    for (const Replica& replica : replicas) {
        result.final_energies.push_back(replica.energy);
    }
    return result;
}

}  // namespace hp
