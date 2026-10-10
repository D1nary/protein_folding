#include "hp/vshd_moves.hpp"

namespace hp {

namespace {

// Crankshaft on the pair (j, j+1), whose chain neighbours j-1 and j+2 stay
// still. The four residues form a U when j-1 and j+2 are adjacent; the flip
// mirrors j and j+1 across the line through j-1 and j+2.
void add_crankshaft(const Conformation& c, int j, std::vector<Move>& moves) {
    if (j < 1 || j + 2 >= c.size()) {
        return;
    }
    const Coord before = c.position(j - 1);
    const Coord after = c.position(j + 2);
    if (!are_adjacent(before, after)) {
        return;
    }
    const Coord a = c.position(j);
    const Coord b = c.position(j + 1);
    const Coord new_a{2 * before.first - a.first, 2 * before.second - a.second};
    const Coord new_b{2 * after.first - b.first, 2 * after.second - b.second};
    if (c.is_free(new_a) && c.is_free(new_b)) {
        moves.push_back({{j, new_a}, {j + 1, new_b}});
    }
}

}  // namespace

std::vector<Move> vshd_moves(const Conformation& c, int k) {
    std::vector<Move> moves;
    const int n = c.size();
    if (n < 2 || k < 0 || k >= n) {
        return moves;
    }

    if (k == 0 || k == n - 1) {
        // End move: pivot around the only chain neighbour.
        const Coord anchor = c.position(k == 0 ? 1 : n - 2);
        for (const Coord target : neighbours(anchor)) {
            if (c.is_free(target)) {
                moves.push_back({{k, target}});
            }
        }
        return moves;
    }

    // Corner move: k-1 and k+1 are diagonal, so they share two nodes. One is
    // the node of k, the other is the target.
    const Coord before = c.position(k - 1);
    const Coord after = c.position(k + 1);
    if (are_diagonal(before, after)) {
        const Coord here = c.position(k);
        const Coord target{before.first + after.first - here.first,
                           before.second + after.second - here.second};
        if (c.is_free(target)) {
            moves.push_back({{k, target}});
        }
    }

    // Crankshaft moves: k can be the first or the second residue of the pair.
    add_crankshaft(c, k, moves);
    add_crankshaft(c, k - 1, moves);

    return moves;
}

}  // namespace hp
