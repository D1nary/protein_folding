#pragma once

// Exact ground state of a short HP sequence, by exhaustive enumeration of the
// self-avoiding walks (depth-first search with branch and bound).
// Used to verify the minimum found by the REMC.

#include <string>

#include "hp/conformation.hpp"

namespace hp {

struct ExactResult {
    Conformation best;         // one of the lowest-energy conformations
    int energy;                // the exact minimum energy
    long long explored_nodes;  // partial chains visited by the search
};

// Finds the minimum energy of `sequence` over all the self-avoiding walks that
// fit in a width x height lattice.
//
// The chain is grown one residue at a time, trying the free adjacent nodes
// and going back when none is left. Two things keep the search small:
//
// - symmetry: the first bond is fixed to the right and the first turn is
//   upwards, so only one of the 8 rotated/mirrored copies of every walk is
//   built;
// - bound: a partial chain is abandoned when, even with the largest number of
//   contacts its missing residues can still form (2 for every H, 3 for the
//   last residue), it cannot beat the best conformation found so far.
//
// The cost still grows exponentially with the length of the chain: the
// function is meant for short sequences.
//
// Throws std::invalid_argument if the sequence is not made only of H and P or
// if no walk of that length fits in the lattice.
ExactResult exact_ground_state(const std::string& sequence, int width, int height);

}  // namespace hp
