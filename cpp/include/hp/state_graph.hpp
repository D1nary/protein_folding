#pragma once

// Graph of the conformations connected by one action, and the oracle that
// labels the dataset (phase 2): distance of every state from the native
// state, optimal actions, paths to the native state and cleaning of the REMC
// trajectories. See notes/policy_dataset.md.
//
// States are canonical conformations (canonical.hpp), identified by their
// key; actions are the slots of actions.hpp, in the frame of the canonical
// conformation.

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "hp/conformation.hpp"

namespace hp {

// Value of next_distance for a slot that is not feasible.
constexpr int INVALID_ACTION = -1;
// Distance of a state whose distance from the native state is not known:
// the state is not in the graph, or the native state cannot be reached
// from it inside the graph.
constexpr int UNKNOWN_DISTANCE = -2;

// One feasible action of a state and the key of the state it leads to.
struct Successor {
    int action;
    std::uint64_t key;
};

// Successors of the canonical state `key` of a chain of n residues.
std::vector<Successor> successors(int n, std::uint64_t key);

// All the canonical states of a chain of n residues (self-avoiding walks up
// to translations, rotations and reflections), in increasing order of key.
// Their number grows exponentially with n (about 800 000 for n = 16).
std::vector<std::uint64_t> enumerate_states(int n);

// Directed graph on a set of canonical states: there is an edge u -> v when
// one action of u leads to v. Edges that leave the set are not kept.
class StateGraph {
public:
    StateGraph(int n, std::vector<std::uint64_t> keys);

    int length() const { return n_; }
    int size() const { return static_cast<int>(keys_.size()); }
    long long edge_count() const { return static_cast<long long>(predecessors_.size()); }
    // Keys of the nodes, in increasing order: node i has key keys()[i].
    const std::vector<std::uint64_t>& keys() const { return keys_; }
    // Node of the key, or -1 if the state is not in the graph.
    int index(std::uint64_t key) const;

    // Moves needed to reach one of the target nodes from every node (breadth
    // first search along the reversed edges), or -1 if no target can be
    // reached.
    std::vector<int> distances_to(const std::vector<int>& targets) const;

private:
    int n_;
    std::vector<std::uint64_t> keys_;
    // The edges are stored reversed, in compressed rows: the predecessors of
    // node v are predecessors_[offsets_[v]] ... predecessors_[offsets_[v+1]-1].
    std::vector<long long> offsets_;
    std::vector<int> predecessors_;
};

// Labels of a list of states, arrays of shape (states) or (states, actions)
// flattened in row-major order.
struct Labels {
    int states;
    int actions;
    std::vector<int> energy;
    // Moves to the native state, or UNKNOWN_DISTANCE.
    std::vector<int> distance;
    // 1 if the slot is feasible.
    std::vector<std::uint8_t> mask;
    // Distance of the state reached by the action, INVALID_ACTION or
    // UNKNOWN_DISTANCE.
    std::vector<int> next_distance;
    // Energy change of the action (0 for slots that are not feasible).
    std::vector<int> next_energy_delta;
};

// Path of states joined by actions: actions[i] leads from keys[i] to keys[i+1].
struct StatePath {
    std::vector<std::uint64_t> keys;
    std::vector<int> actions;
};

// Distances from the native state of one sequence, on a graph.
class Oracle {
public:
    // The native states are the nodes with energy `target_energy`, or with
    // the lowest energy of the graph if it is not given. Throws
    // std::invalid_argument if no node has that energy.
    Oracle(std::shared_ptr<const StateGraph> graph, std::string sequence,
           std::optional<int> target_energy);

    const StateGraph& graph() const { return *graph_; }
    const std::string& sequence() const { return sequence_; }
    int target_energy() const { return target_energy_; }
    const std::vector<int>& native_nodes() const { return native_nodes_; }
    // Distance of every node of the graph, -1 if it cannot reach the native
    // state.
    const std::vector<int>& distances() const { return distances_; }

    int energy(std::uint64_t key) const;
    bool is_native(std::uint64_t key) const { return energy(key) == target_energy_; }
    // Distance of a state of the graph, or UNKNOWN_DISTANCE.
    int distance(std::uint64_t key) const;

    // Labels of the states. A state outside the graph gets the distance
    // 1 + (smallest known distance of its successors), if there is one.
    // The optimal actions of a state at distance d > 0 are those with
    // next_distance == d - 1.
    Labels label(const std::vector<std::uint64_t>& keys) const;

    // Path from `start` to a native state, taking at every step one of the
    // optimal actions chosen uniformly at random (generator seeded with
    // `seed`). Throws std::invalid_argument if the distance of `start` is not
    // known.
    StatePath path(std::uint64_t start, std::uint64_t seed) const;

private:
    std::shared_ptr<const StateGraph> graph_;
    std::string sequence_;
    int target_energy_;
    std::vector<int> native_nodes_;
    std::vector<int> distances_;
};

// A REMC trajectory before and after cleaning.
struct PrunedTrajectory {
    // False if the trajectory never reaches a native state of the oracle:
    // then only raw_length is meaningful (the others are -1).
    bool reached_native;
    // Moves of the trajectory up to the first native state.
    int raw_length;
    // Moves left after removing the loops (a return to a state already
    // visited cancels the part of the trajectory in between).
    int loop_erased_length;
    // Moves of the shortest path from the first to the last state, using
    // only the states visited by the trajectory but every action between
    // them, also the ones that the REMC never made (shortcuts).
    int shortest_length;
    // That shortest path.
    StatePath path;
};

// Cleans the trajectory of a walker: `trajectory` lists the keys of the
// states visited, one action apart. It is cut at the first native state of
// the oracle.
PrunedTrajectory prune_trajectory(const std::vector<std::uint64_t>& trajectory,
                                  const Oracle& oracle);

}  // namespace hp
