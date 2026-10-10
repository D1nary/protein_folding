#pragma once

// Move set: pull moves by Lesh, Mitzenmacher and Whitesides (2003).
// See notes/move_set.md.

#include <vector>

#include "hp/conformation.hpp"

namespace hp {

// All the feasible pull moves at residue `k` (numbered from 0). A move is
// feasible if the target nodes are free and inside the lattice.
//
// Residue inside the chain. Let m be one of its two chain neighbours, L a
// free node adjacent to m and diagonal to k, and C the fourth corner of the
// square (k, m, L, C). On the other side of k the chain continues with p.
// - If C holds p, k goes to L and the move ends (it is a corner move).
// - If C is free, k goes to L and p goes to C. Then the rest of the chain on
//   the side of p is pulled: each residue takes the node left two places
//   ahead, until one is already adjacent to the residue before it or the end
//   of the chain is reached.
// Both choices of m are considered, so the chain can be pulled towards
// either end.
//
// First or last residue. Let A be a free node adjacent to k and B a free node
// adjacent to A: k goes to B, its chain neighbour goes to A, and the rest of
// the chain is pulled as above.
std::vector<Move> pull_moves(const Conformation& conformation, int k);

}  // namespace hp
