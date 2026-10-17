#include "hp/sample_writer.hpp"

#include <stdexcept>

#include "hp/canonical.hpp"

namespace hp {

SampleRecorder::SampleRecorder(int max_steps, int save_every)
    : max_steps_(max_steps), save_every_(save_every) {
    if (max_steps_ < 1 || save_every_ < 1) {
        throw std::invalid_argument("max_steps and save_every must be at least 1");
    }
}

void SampleRecorder::start(int walkers, const Conformation& initial) {
    if (initial.size() > MAX_KEY_LENGTH) {
        throw std::invalid_argument("the chain is too long to be recorded");
    }
    trajectories_.assign(walkers, std::vector<std::uint64_t>{canonical_key(initial)});
    snapshots_.clear();
}

void SampleRecorder::record_move(int walker, const Conformation& conformation) {
    std::vector<std::uint64_t>& trajectory = trajectories_[walker];
    if (static_cast<int>(trajectory.size()) > max_steps_) {
        return;  // full
    }
    const std::uint64_t key = canonical_key(conformation);
    // A move can turn the chain into its own mirror image: same state.
    if (key != trajectory.back()) {
        trajectory.push_back(key);
    }
}

void SampleRecorder::record_snapshot(int cycle, int replica, int walker, int energy,
                                     const Conformation& conformation) {
    snapshots_.push_back({cycle, replica, walker, energy, canonical_key(conformation)});
}

}  // namespace hp
