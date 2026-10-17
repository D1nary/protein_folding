#pragma once

// Recording of the configurations visited by the REMC, for the dataset
// (phase 2). See notes/dataset.md.
//
// The data stays in memory: Python reads it through the wrapper and writes
// the files, so the C++ code reads and writes no file.

#include <cstdint>
#include <vector>

#include "hp/conformation.hpp"

namespace hp {

// State of one replica at the end of a cycle.
struct Snapshot {
    int cycle;
    int replica;  // temperature index: 0 is the coldest replica
    int walker;   // which trajectory the conformation belongs to
    int energy;
    std::uint64_t key;  // canonical key (canonical.hpp)
};

// Passed to run_remc(), it records two things:
//
// - the trajectory of every walker. A walker is a conformation followed
//   through the swaps: a swap exchanges the temperatures of two walkers, not
//   their conformations, so every trajectory is a chain of single moves. The
//   trajectory lists the canonical key of the initial state and of the state
//   after every accepted move that changes it, up to `max_steps` moves;
// - every `save_every` cycles, a snapshot of each replica.
class SampleRecorder {
public:
    // Throws std::invalid_argument if a parameter is not positive.
    SampleRecorder(int max_steps, int save_every);

    int max_steps() const { return max_steps_; }
    int save_every() const { return save_every_; }
    // trajectories()[w] is the trajectory of walker w.
    const std::vector<std::vector<std::uint64_t>>& trajectories() const { return trajectories_; }
    const std::vector<Snapshot>& snapshots() const { return snapshots_; }

    // Called by run_remc(). Throws std::invalid_argument if the chain is too
    // long for a canonical key.
    void start(int walkers, const Conformation& initial);
    void record_move(int walker, const Conformation& conformation);
    bool wants_snapshot(int cycle) const { return (cycle + 1) % save_every_ == 0; }
    void record_snapshot(int cycle, int replica, int walker, int energy,
                         const Conformation& conformation);

private:
    int max_steps_;
    int save_every_;
    std::vector<std::vector<std::uint64_t>> trajectories_;
    std::vector<Snapshot> snapshots_;
};

}  // namespace hp
