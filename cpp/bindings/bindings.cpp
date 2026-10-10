// pybind11 wrapper: exposes the C++ code as the Python module `hpfold._core`.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "hp/conformation.hpp"
#include "hp/pull_moves.hpp"
#include "hp/remc.hpp"
#include "hp/vshd_moves.hpp"

namespace py = pybind11;

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
    m.def("run_remc", &hp::run_remc, py::arg("initial"), py::arg("params"),
          "Runs the REMC simulation starting all the replicas from `initial`.");
}
