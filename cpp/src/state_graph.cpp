#include "hp/state_graph.hpp"

#include <algorithm>
#include <climits>
#include <deque>
#include <random>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "hp/actions.hpp"
#include "hp/canonical.hpp"

namespace hp {

namespace {

// Successors of a conformation; the conformation is moved and put back.
std::vector<Successor> successors_of(Conformation& c) {
    std::vector<Successor> result;
    const std::vector<Move> moves = action_moves(c);
    for (int a = 0; a < static_cast<int>(moves.size()); ++a) {
        if (moves[a].empty()) {
            continue;
        }
        const Move undo = c.apply(moves[a]);
        result.push_back({a, canonical_key(c)});
        c.apply(undo);
    }
    return result;
}

void check_length(int n) {
    if (n < 3 || n > MAX_KEY_LENGTH) {
        throw std::invalid_argument("the chain must have between 3 and " +
                                    std::to_string(MAX_KEY_LENGTH) + " residues");
    }
}

// Depth-first search over the self-avoiding walks, as in exact.cpp: the
// first bond goes to the right and the first turn is LEFT, so every
// canonical state is built once.
class Enumeration {
public:
    explicit Enumeration(int n)
        : n_(n), side_(2 * n + 1), occupied_(static_cast<std::size_t>(side_) * side_, false),
          turns_(n - 2) {}

    std::vector<std::uint64_t> run() {
        const Coord first{n_, n_};
        const Coord second{n_ + 1, n_};
        occupied_[cell(first)] = true;
        occupied_[cell(second)] = true;
        extend(2, second, {1, 0}, true);
        return std::move(keys_);
    }

private:
    int cell(Coord c) const { return c.second * side_ + c.first; }

    // Places residue i after the residue on `last`, reached with `direction`.
    void extend(int i, Coord last, Coord direction, bool straight) {
        if (i == n_) {
            keys_.push_back(turns_key(turns_));
            return;
        }
        // FORWARD, LEFT, RIGHT: the keys come out in increasing order.
        for (const int turn : {FORWARD, LEFT, RIGHT}) {
            if (straight && turn == RIGHT) {
                continue;  // mirror image of a walk whose first turn is LEFT
            }
            const Coord d = turn == FORWARD ? direction
                            : turn == LEFT  ? turn_left(direction)
                                            : turn_right(direction);
            const Coord next{last.first + d.first, last.second + d.second};
            if (occupied_[cell(next)]) {
                continue;
            }
            occupied_[cell(next)] = true;
            turns_[i - 2] = turn;
            extend(i + 1, next, d, straight && turn == FORWARD);
            occupied_[cell(next)] = false;
        }
    }

    int n_;
    int side_;
    std::vector<bool> occupied_;
    std::vector<int> turns_;
    std::vector<std::uint64_t> keys_;
};

// What one state looks like to the oracle: its distance and, for every slot,
// where the action leads.
struct Expansion {
    int energy;
    int distance;
    std::vector<std::uint8_t> mask;
    std::vector<std::uint64_t> next_key;
    std::vector<int> next_distance;
    std::vector<int> next_energy_delta;
};

}  // namespace

std::vector<Successor> successors(int n, std::uint64_t key) {
    check_length(n);
    Conformation c = conformation_from_key(std::string(n, 'P'), key);
    return successors_of(c);
}

std::vector<std::uint64_t> enumerate_states(int n) {
    check_length(n);
    return Enumeration(n).run();
}

// ----------------------------------------------------------------------
// StateGraph

StateGraph::StateGraph(int n, std::vector<std::uint64_t> keys) : n_(n), keys_(std::move(keys)) {
    check_length(n);
    std::sort(keys_.begin(), keys_.end());
    keys_.erase(std::unique(keys_.begin(), keys_.end()), keys_.end());

    // Forward edges first, one row per node, without repetitions or loops.
    // The nodes are split into contiguous blocks handled by parallel threads
    // (the graph of all the states of a long chain has millions of nodes);
    // each block has its own rows, joined in order at the end.
    const int threads = std::max(1, std::min<int>(std::thread::hardware_concurrency(),
                                                  size() / 1000 + 1));
    std::vector<std::vector<int>> block_targets(threads);
    std::vector<std::vector<int>> block_degrees(threads);
    auto build_block = [&](int b) {
        const std::string sequence(n_, 'P');
        const int first = static_cast<int>(static_cast<long long>(size()) * b / threads);
        const int last = static_cast<int>(static_cast<long long>(size()) * (b + 1) / threads);
        std::vector<int> row;
        for (int u = first; u < last; ++u) {
            Conformation c = conformation_from_key(sequence, keys_[u]);
            row.clear();
            for (const Successor& s : successors_of(c)) {
                const int v = index(s.key);
                if (v != -1 && v != u) {
                    row.push_back(v);
                }
            }
            std::sort(row.begin(), row.end());
            row.erase(std::unique(row.begin(), row.end()), row.end());
            block_targets[b].insert(block_targets[b].end(), row.begin(), row.end());
            block_degrees[b].push_back(static_cast<int>(row.size()));
        }
    };
    std::vector<std::thread> workers;
    for (int b = 1; b < threads; ++b) {
        workers.emplace_back(build_block, b);
    }
    build_block(0);
    for (std::thread& w : workers) {
        w.join();
    }

    std::vector<long long> forward_offsets(keys_.size() + 1, 0);
    std::vector<int> forward;
    int node = 0;
    for (int b = 0; b < threads; ++b) {
        forward.insert(forward.end(), block_targets[b].begin(), block_targets[b].end());
        std::vector<int>().swap(block_targets[b]);
        for (const int degree : block_degrees[b]) {
            forward_offsets[node + 1] = forward_offsets[node] + degree;
            ++node;
        }
    }

    // Then reversed: count the predecessors of every node, then fill.
    offsets_.assign(keys_.size() + 1, 0);
    for (const int v : forward) {
        ++offsets_[v + 1];
    }
    for (std::size_t v = 0; v < keys_.size(); ++v) {
        offsets_[v + 1] += offsets_[v];
    }
    predecessors_.resize(forward.size());
    std::vector<long long> fill(offsets_.begin(), offsets_.end() - 1);
    for (int u = 0; u < size(); ++u) {
        for (long long e = forward_offsets[u]; e < forward_offsets[u + 1]; ++e) {
            predecessors_[fill[forward[e]]++] = u;
        }
    }
}

int StateGraph::index(std::uint64_t key) const {
    const auto it = std::lower_bound(keys_.begin(), keys_.end(), key);
    return (it != keys_.end() && *it == key) ? static_cast<int>(it - keys_.begin()) : -1;
}

std::vector<int> StateGraph::distances_to(const std::vector<int>& targets) const {
    std::vector<int> distance(keys_.size(), -1);
    std::deque<int> queue;
    for (const int t : targets) {
        if (t < 0 || t >= size()) {
            throw std::invalid_argument("target node out of range");
        }
        if (distance[t] == -1) {
            distance[t] = 0;
            queue.push_back(t);
        }
    }
    while (!queue.empty()) {
        const int v = queue.front();
        queue.pop_front();
        for (long long e = offsets_[v]; e < offsets_[v + 1]; ++e) {
            const int u = predecessors_[e];
            if (distance[u] == -1) {
                distance[u] = distance[v] + 1;
                queue.push_back(u);
            }
        }
    }
    return distance;
}

// ----------------------------------------------------------------------
// Oracle

namespace {

// Distance of the state reached by an action: known if the state is in the
// graph, 0 if it is outside the graph but native.
int known_distance(const Oracle& oracle, std::uint64_t key, int energy) {
    const int node = oracle.graph().index(key);
    if (node != -1) {
        const int d = oracle.distances()[node];
        return d == -1 ? UNKNOWN_DISTANCE : d;
    }
    return energy == oracle.target_energy() ? 0 : UNKNOWN_DISTANCE;
}

Expansion expand(const Oracle& oracle, std::uint64_t key) {
    Conformation c = conformation_from_key(oracle.sequence(), key);
    const std::vector<Move> moves = action_moves(c);
    const std::size_t count = moves.size();

    Expansion x{c.energy(), UNKNOWN_DISTANCE, std::vector<std::uint8_t>(count, 0),
                std::vector<std::uint64_t>(count, 0), std::vector<int>(count, INVALID_ACTION),
                std::vector<int>(count, 0)};
    int best = INT_MAX;
    for (std::size_t a = 0; a < count; ++a) {
        if (moves[a].empty()) {
            continue;
        }
        const Move undo = c.apply(moves[a]);
        const int energy = c.energy();
        x.mask[a] = 1;
        x.next_key[a] = canonical_key(c);
        x.next_energy_delta[a] = energy - x.energy;
        x.next_distance[a] = known_distance(oracle, x.next_key[a], energy);
        c.apply(undo);
        if (x.next_distance[a] >= 0) {
            best = std::min(best, x.next_distance[a]);
        }
    }

    x.distance = known_distance(oracle, key, x.energy);
    if (x.distance == UNKNOWN_DISTANCE && oracle.graph().index(key) == -1 && best != INT_MAX) {
        // Outside the graph: one move more than the best successor.
        x.distance = best + 1;
    }
    return x;
}

}  // namespace

Oracle::Oracle(std::shared_ptr<const StateGraph> graph, std::string sequence,
               std::optional<int> target_energy)
    : graph_(std::move(graph)), sequence_(std::move(sequence)) {
    if (!graph_) {
        throw std::invalid_argument("the oracle needs a graph");
    }
    if (static_cast<int>(sequence_.size()) != graph_->length()) {
        throw std::invalid_argument("the sequence and the graph have different lengths");
    }
    if (graph_->size() == 0) {
        throw std::invalid_argument("the graph is empty");
    }

    std::vector<int> energies(graph_->size());
    for (int v = 0; v < graph_->size(); ++v) {
        energies[v] = conformation_from_key(sequence_, graph_->keys()[v]).energy();
    }
    target_energy_ =
        target_energy ? *target_energy : *std::min_element(energies.begin(), energies.end());
    for (int v = 0; v < graph_->size(); ++v) {
        if (energies[v] == target_energy_) {
            native_nodes_.push_back(v);
        }
    }
    if (native_nodes_.empty()) {
        throw std::invalid_argument("no state of the graph has the target energy");
    }
    distances_ = graph_->distances_to(native_nodes_);
}

int Oracle::energy(std::uint64_t key) const {
    return conformation_from_key(sequence_, key).energy();
}

int Oracle::distance(std::uint64_t key) const {
    const int node = graph_->index(key);
    if (node != -1) {
        return distances_[node] == -1 ? UNKNOWN_DISTANCE : distances_[node];
    }
    return expand(*this, key).distance;
}

Labels Oracle::label(const std::vector<std::uint64_t>& keys) const {
    const int actions = action_count(graph_->length());
    Labels labels{static_cast<int>(keys.size()), actions, {}, {}, {}, {}, {}};
    labels.energy.reserve(keys.size());
    labels.distance.reserve(keys.size());
    labels.mask.reserve(keys.size() * actions);
    labels.next_distance.reserve(keys.size() * actions);
    labels.next_energy_delta.reserve(keys.size() * actions);
    for (const std::uint64_t key : keys) {
        const Expansion x = expand(*this, key);
        labels.energy.push_back(x.energy);
        labels.distance.push_back(x.distance);
        labels.mask.insert(labels.mask.end(), x.mask.begin(), x.mask.end());
        labels.next_distance.insert(labels.next_distance.end(), x.next_distance.begin(),
                                    x.next_distance.end());
        labels.next_energy_delta.insert(labels.next_energy_delta.end(),
                                        x.next_energy_delta.begin(), x.next_energy_delta.end());
    }
    return labels;
}

StatePath Oracle::path(std::uint64_t start, std::uint64_t seed) const {
    std::mt19937_64 rng(seed);
    StatePath path{{start}, {}};
    Expansion x = expand(*this, start);
    if (x.distance < 0) {
        throw std::invalid_argument("the distance of the start state is not known");
    }
    std::vector<int> optimal;
    while (x.distance > 0) {
        optimal.clear();
        for (int a = 0; a < static_cast<int>(x.next_distance.size()); ++a) {
            if (x.next_distance[a] == x.distance - 1) {
                optimal.push_back(a);
            }
        }
        if (optimal.empty()) {
            throw std::logic_error("a state at distance > 0 has no optimal action");
        }
        std::uniform_int_distribution<std::size_t> pick(0, optimal.size() - 1);
        const int a = optimal[pick(rng)];
        path.actions.push_back(a);
        path.keys.push_back(x.next_key[a]);
        x = expand(*this, x.next_key[a]);
    }
    return path;
}

// ----------------------------------------------------------------------
// Cleaning of the REMC trajectories

PrunedTrajectory prune_trajectory(const std::vector<std::uint64_t>& trajectory,
                                  const Oracle& oracle) {
    const int n = oracle.graph().length();
    PrunedTrajectory result{false, static_cast<int>(trajectory.size()) - 1, -1, -1, {}};
    std::size_t end = 0;
    while (end < trajectory.size() && !oracle.is_native(trajectory[end])) {
        ++end;
    }
    if (end == trajectory.size()) {
        return result;
    }
    result.reached_native = true;
    result.raw_length = static_cast<int>(end);

    // Loop truncation: going back to a state already on the path removes
    // everything visited since then.
    std::vector<std::uint64_t> path;
    std::unordered_map<std::uint64_t, std::size_t> position;
    for (std::size_t i = 0; i <= end; ++i) {
        const std::uint64_t key = trajectory[i];
        const auto it = position.find(key);
        if (it == position.end()) {
            position.emplace(key, path.size());
            path.push_back(key);
        } else {
            for (std::size_t j = it->second + 1; j < path.size(); ++j) {
                position.erase(path[j]);
            }
            path.resize(it->second + 1);
        }
    }
    result.loop_erased_length = static_cast<int>(path.size()) - 1;

    // Shortest path from the first to the last state through the visited
    // states, with every action between them (breadth-first search).
    const std::unordered_set<std::uint64_t> visited(trajectory.begin(),
                                                    trajectory.begin() + end + 1);
    const std::uint64_t source = trajectory.front();
    const std::uint64_t target = trajectory[end];
    // For every state reached: the state it was reached from and the action.
    std::unordered_map<std::uint64_t, std::pair<std::uint64_t, int>> parent;
    parent.emplace(source, std::make_pair(source, -1));
    std::deque<std::uint64_t> queue{source};
    while (!queue.empty() && parent.count(target) == 0) {
        const std::uint64_t u = queue.front();
        queue.pop_front();
        for (const Successor& s : successors(n, u)) {
            if (visited.count(s.key) != 0 && parent.count(s.key) == 0) {
                parent.emplace(s.key, std::make_pair(u, s.action));
                queue.push_back(s.key);
            }
        }
    }
    if (parent.count(target) == 0) {
        throw std::logic_error("the states of the trajectory are not one action apart");
    }
    for (std::uint64_t key = target; key != source; key = parent.at(key).first) {
        result.path.keys.push_back(key);
        result.path.actions.push_back(parent.at(key).second);
    }
    result.path.keys.push_back(source);
    std::reverse(result.path.keys.begin(), result.path.keys.end());
    std::reverse(result.path.actions.begin(), result.path.actions.end());
    result.shortest_length = static_cast<int>(result.path.actions.size());
    return result;
}

}  // namespace hp
