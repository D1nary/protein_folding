#pragma once

// Move set: VSHD moves (Verdier-Stockmayer, Hilhorst-Deutch), as described by
// Thachuk et al. (2007): end moves, corner moves and crankshaft moves.

#include <vector>

#include "hp/conformation.hpp"

namespace hp {

// All the feasible VSHD moves at residue `k` (numbered from 0). A move is
// feasible if the target nodes are free and inside the lattice.
//
// - End move (k is the first or the last residue): k goes to a free node
//   adjacent to its only chain neighbour.
// - Corner move (k inside the chain): if k-1, k, k+1 form a corner, k goes to
//   the opposite corner of the square, if free.
// - Crankshaft move: if k is one of the two middle residues of a U-shaped
//   bend, the two of them are flipped to the other side of the bend.
std::vector<Move> vshd_moves(const Conformation& conformation, int k);

}  // namespace hp
