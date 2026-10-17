// pybind11 wrapper: exposes the C++ code as the Python module `hpfold._core`.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

#include "hp/actions.hpp"
#include "hp/canonical.hpp"
#include "hp/conformation.hpp"
#include "hp/exact.hpp"
#include "hp/pull_moves.hpp"
#include "hp/remc.hpp"
#include "hp/sample_writer.hpp"
#include "hp/state_graph.hpp"
#include "hp/vshd_moves.hpp"

namespace py = pybind11;

namespace {

// Copies a vector into a new numpy array of the given shape. Large results
// (keys, labels) are returned as arrays: a Python list of a million ints
// would be slow to build and to use.
template <typename Out, typename In>
py::array_t<Out> to_array(const std::vector<In>& values, std::vector<py::ssize_t> shape) {
    py::array_t<Out> array(shape);
    Out* data = array.mutable_data();
    for (std::size_t i = 0; i < values.size(); ++i) {
        data[i] = static_cast<Out>(values[i]);
    }
    return array;
}

template <typename Out, typename In>
py::array_t<Out> to_array(const std::vector<In>& values) {
    return to_array<Out>(values, {static_cast<py::ssize_t>(values.size())});
}

using KeyArray = py::array_t<std::uint64_t, py::array::c_style | py::array::forcecast>;

std::vector<std::uint64_t> to_keys(const KeyArray& keys) {
    return std::vector<std::uint64_t>(keys.data(), keys.data() + keys.size());
}

py::dict path_dict(const hp::StatePath& path) {
    py::dict d;
    d["keys"] = to_array<std::uint64_t>(path.keys);
    d["actions"] = to_array<std::int32_t>(path.actions);
    return d;
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "C++ core of hpfold: REMC for HP proteins on a 2D lattice.";

    // pybind11/stl.h converts the containers by itself: a Coord is a tuple
    // (x, y), a Move is a list of (residue index, (x, y)).

    py::class_<hp::Conformation>(m, "Conformation")
        .def(py::init<std::string, std::vector<hp::Coord>, int, int>(),
             py::arg("sequence"), py::arg("coordinates"), py::arg("width"), py::arg("height"),
             "Chain of H/P residues on a width x height lattice. Raises ValueError if the "
             "conformation is not a self-avoiding walk inside the lattice.")
        .def_property_readonly("sequence", &hp::Conformation::sequence)
        .def_property_readonly("coordinates", &hp::Conformation::coordinates)
        .def_property_readonly("width", &hp::Conformation::width)
        .def_property_readonly("height", &hp::Conformation::height)
        .def("__len__", &hp::Conformation::size)
        .def("is_valid", &hp::Conformation::is_valid)
        .def("energy", &hp::Conformation::energy,
             "Minus the number of H-H contacts between residues not consecutive in the chain.")
        .def("apply", &hp::Conformation::apply, py::arg("move"),
             "Applies a move returned by vshd_moves or pull_moves; returns the move that "
             "undoes it.");

    m.def("vshd_moves", &hp::vshd_moves, py::arg("conformation"), py::arg("k"),
          "Feasible VSHD moves (end, corner, crankshaft) at residue k.");
    m.def("pull_moves", &hp::pull_moves, py::arg("conformation"), py::arg("k"),
          "Feasible pull moves at residue k.");

    py::class_<hp::RemcParams>(m, "RemcParams")
        .def(py::init([](int n_replicas, double t_min, double t_max, int local_steps,
                         int n_cycles, double rho, std::uint64_t seed) {
                 return hp::RemcParams{n_replicas, t_min, t_max, local_steps,
                                       n_cycles,   rho,   seed};
             }),
             py::kw_only(), py::arg("n_replicas"), py::arg("t_min"), py::arg("t_max"),
             py::arg("local_steps"), py::arg("n_cycles"), py::arg("rho"), py::arg("seed"))
        .def_readwrite("n_replicas", &hp::RemcParams::n_replicas)
        .def_readwrite("t_min", &hp::RemcParams::t_min)
        .def_readwrite("t_max", &hp::RemcParams::t_max)
        .def_readwrite("local_steps", &hp::RemcParams::local_steps)
        .def_readwrite("n_cycles", &hp::RemcParams::n_cycles)
        .def_readwrite("rho", &hp::RemcParams::rho)
        .def_readwrite("seed", &hp::RemcParams::seed);

    py::class_<hp::RemcResult>(m, "RemcResult")
        .def_readonly("best", &hp::RemcResult::best)
        .def_readonly("best_energy", &hp::RemcResult::best_energy)
        .def_readonly("temperatures", &hp::RemcResult::temperatures)
        .def_readonly("final_energies", &hp::RemcResult::final_energies)
        .def_readonly("swap_attempts", &hp::RemcResult::swap_attempts)
        .def_readonly("swap_accepted", &hp::RemcResult::swap_accepted);

    m.def("temperature_ladder", &hp::temperature_ladder, py::arg("n_replicas"),
          py::arg("t_min"), py::arg("t_max"),
          "Temperatures of the replicas, linearly spaced between t_min and t_max.");
    py::class_<hp::SampleRecorder>(m, "SampleRecorder")
        .def(py::init<int, int>(), py::arg("max_steps"), py::arg("save_every"),
             "Records the walkers' trajectories (up to max_steps moves each) and a snapshot "
             "of every replica each save_every cycles, when passed to run_remc.")
        .def_property_readonly("max_steps", &hp::SampleRecorder::max_steps)
        .def_property_readonly("save_every", &hp::SampleRecorder::save_every)
        .def_property_readonly(
            "trajectories",
            [](const hp::SampleRecorder& r) {
                py::list result;
                for (const auto& t : r.trajectories()) {
                    result.append(to_array<std::uint64_t>(t));
                }
                return result;
            },
            "One array of canonical keys per walker.")
        .def_property_readonly(
            "snapshots",
            [](const hp::SampleRecorder& r) {
                const auto& s = r.snapshots();
                std::vector<int> cycle, replica, walker, energy;
                std::vector<std::uint64_t> key;
                for (const hp::Snapshot& x : s) {
                    cycle.push_back(x.cycle);
                    replica.push_back(x.replica);
                    walker.push_back(x.walker);
                    energy.push_back(x.energy);
                    key.push_back(x.key);
                }
                py::dict d;
                d["cycle"] = to_array<std::int32_t>(cycle);
                d["replica"] = to_array<std::int32_t>(replica);
                d["walker"] = to_array<std::int32_t>(walker);
                d["energy"] = to_array<std::int32_t>(energy);
                d["key"] = to_array<std::uint64_t>(key);
                return d;
            },
            "Dictionary of arrays: cycle, replica (temperature index), walker, energy, key.");

    m.def("run_remc", &hp::run_remc, py::arg("initial"), py::arg("params"),
          py::arg("recorder") = nullptr,
          "Runs the REMC simulation starting all the replicas from `initial`; the optional "
          "SampleRecorder receives the visited states.");

    // ------------------------------------------------------------------
    // Dataset (phase 2): canonical form, action space, oracle.

    m.attr("MAX_KEY_LENGTH") = hp::MAX_KEY_LENGTH;
    m.attr("INVALID_ACTION") = hp::INVALID_ACTION;
    m.attr("UNKNOWN_DISTANCE") = hp::UNKNOWN_DISTANCE;

    m.def("turns", &hp::turns, py::arg("conformation"),
          "The N-2 turns of the chain: 0 forward, 1 left, 2 right.");
    m.def("mirror_turns", &hp::mirror_turns, py::arg("turns"));
    m.def("canonical_turns", &hp::canonical_turns, py::arg("conformation"));
    m.def("turns_key", &hp::turns_key, py::arg("turns"));
    m.def("key_turns", &hp::key_turns, py::arg("key"), py::arg("n"));
    m.def("canonical_key", &hp::canonical_key, py::arg("conformation"),
          "Key that is the same for conformations related by translations, rotations and "
          "reflections.");
    m.def("free_lattice_side", &hp::free_lattice_side, py::arg("n"));
    m.def("conformation_from_turns", &hp::conformation_from_turns, py::arg("sequence"),
          py::arg("turns"));
    m.def("conformation_from_key", &hp::conformation_from_key, py::arg("sequence"),
          py::arg("key"));

    py::list type_names;
    for (int t = 0; t < hp::N_MOVE_TYPES; ++t) {
        type_names.append(hp::move_type_name(t));
    }
    m.attr("MOVE_TYPE_NAMES") = py::tuple(type_names);
    m.def("mirror_move_type", &hp::mirror_move_type, py::arg("type"));
    m.def("action_count", &hp::action_count, py::arg("n"));
    m.def(
        "action_table",
        [](int n) {
            std::vector<std::pair<int, int>> table;
            for (const hp::Action& a : hp::action_table(n)) {
                table.emplace_back(a.residue, a.type);
            }
            return table;
        },
        py::arg("n"), "List of (residue, move type) for every slot.");
    m.def("action_index", &hp::action_index, py::arg("n"), py::arg("residue"), py::arg("type"));
    m.def("mirror_permutation", &hp::mirror_permutation, py::arg("n"));
    m.def("classify_move", &hp::classify_move, py::arg("conformation"), py::arg("k"),
          py::arg("move"), py::arg("pull"));
    m.def("action_moves", &hp::action_moves, py::arg("conformation"),
          "Feasible move of every slot (empty list if the slot is not feasible).");
    m.def(
        "successors",
        [](int n, std::uint64_t key) {
            std::vector<std::pair<int, std::uint64_t>> result;
            for (const hp::Successor& s : hp::successors(n, key)) {
                result.emplace_back(s.action, s.key);
            }
            return result;
        },
        py::arg("n"), py::arg("key"), "List of (slot, key of the next state).");
    m.def(
        "enumerate_states", [](int n) { return to_array<std::uint64_t>(hp::enumerate_states(n)); },
        py::arg("n"), "Keys of all the canonical states of a chain of n residues.");

    py::class_<hp::StateGraph, std::shared_ptr<hp::StateGraph>>(m, "StateGraph")
        .def(py::init([](int n, const KeyArray& keys) {
                 return std::make_shared<hp::StateGraph>(n, to_keys(keys));
             }),
             py::arg("n"), py::arg("keys"),
             "Graph on the given canonical states, with an edge for every action between them.")
        .def_property_readonly("length", &hp::StateGraph::length)
        .def("__len__", &hp::StateGraph::size)
        .def_property_readonly("edge_count", &hp::StateGraph::edge_count)
        .def_property_readonly(
            "keys", [](const hp::StateGraph& g) { return to_array<std::uint64_t>(g.keys()); })
        .def("index", &hp::StateGraph::index, py::arg("key"))
        .def(
            "distances_to",
            [](const hp::StateGraph& g, const std::vector<int>& targets) {
                return to_array<std::int32_t>(g.distances_to(targets));
            },
            py::arg("targets"));

    py::class_<hp::Oracle>(m, "Oracle")
        .def(py::init([](std::shared_ptr<hp::StateGraph> graph, std::string sequence,
                         std::optional<int> target_energy) {
                 return hp::Oracle(std::move(graph), std::move(sequence), target_energy);
             }),
             py::arg("graph"), py::arg("sequence"), py::arg("target_energy") = py::none(),
             "Distances from the native states (the nodes with target_energy, or with the "
             "lowest energy of the graph).")
        .def_property_readonly("sequence", &hp::Oracle::sequence)
        .def_property_readonly("target_energy", &hp::Oracle::target_energy)
        .def_property_readonly(
            "native_keys",
            [](const hp::Oracle& o) {
                std::vector<std::uint64_t> keys;
                for (const int v : o.native_nodes()) {
                    keys.push_back(o.graph().keys()[v]);
                }
                return to_array<std::uint64_t>(keys);
            })
        .def_property_readonly(
            "distances", [](const hp::Oracle& o) { return to_array<std::int32_t>(o.distances()); },
            "Distance of every node of the graph (-1 if the native state cannot be reached).")
        .def("energy", &hp::Oracle::energy, py::arg("key"))
        .def("is_native", &hp::Oracle::is_native, py::arg("key"))
        .def("distance", &hp::Oracle::distance, py::arg("key"))
        .def(
            "label",
            [](const hp::Oracle& o, const KeyArray& keys) {
                const hp::Labels l = o.label(to_keys(keys));
                py::dict d;
                d["energy"] = to_array<std::int16_t>(l.energy);
                d["distance"] = to_array<std::int16_t>(l.distance);
                d["mask"] = to_array<bool>(l.mask, {l.states, l.actions});
                d["next_distance"] = to_array<std::int16_t>(l.next_distance, {l.states, l.actions});
                d["next_energy_delta"] =
                    to_array<std::int8_t>(l.next_energy_delta, {l.states, l.actions});
                return d;
            },
            py::arg("keys"),
            "Dictionary of arrays: energy, distance (S,), mask, next_distance, "
            "next_energy_delta (S, A).")
        .def(
            "path",
            [](const hp::Oracle& o, std::uint64_t start, std::uint64_t seed) {
                return path_dict(o.path(start, seed));
            },
            py::arg("start"), py::arg("seed"),
            "Path to a native state along optimal actions: dictionary with keys and actions.");

    m.def(
        "prune_trajectory",
        [](const KeyArray& trajectory, const hp::Oracle& oracle) {
            const hp::PrunedTrajectory p = hp::prune_trajectory(to_keys(trajectory), oracle);
            py::dict d;
            d["reached_native"] = p.reached_native;
            d["raw_length"] = p.raw_length;
            d["loop_erased_length"] = p.loop_erased_length;
            d["shortest_length"] = p.shortest_length;
            d["path"] = path_dict(p.path);
            return d;
        },
        py::arg("trajectory"), py::arg("oracle"),
        "Cleans a walker trajectory: lengths before and after loop truncation and shortest "
        "path through the visited states.");

    py::class_<hp::ExactResult>(m, "ExactResult")
        .def_readonly("best", &hp::ExactResult::best)
        .def_readonly("energy", &hp::ExactResult::energy)
        .def_readonly("explored_nodes", &hp::ExactResult::explored_nodes);

    m.def("exact_ground_state", &hp::exact_ground_state, py::arg("sequence"),
          py::arg("width"), py::arg("height"),
          "Exact minimum energy of a short sequence, by exhaustive enumeration of the "
          "self-avoiding walks that fit in the lattice.");
}
