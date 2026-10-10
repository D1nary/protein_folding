#include "hp/pull_moves.hpp"

namespace hp {

namespace {

// Completes a pull move. `move` already holds the two residues that open the
// move: `last`, on its new node `last_target`, is the one closest to the part
// of the chain still to be pulled. `step` (+1 or -1) is the direction in
// which that part is visited.
//
// Each residue j takes the node that residue j - 2*step had before the move,
// which is free because that residue has already moved. The pull stops at the
// first residue that is already adjacent to the new node of the one before.
void pull_rest(const Conformation& c, int last, Coord last_target, int step, Move& move) {
    Coord previous_target = last_target;
    for (int j = last + step; j >= 0 && j < c.size(); j += step) {
        if (are_adjacent(c.position(j), previous_target)) {
            break;
        }
        previous_target = c.position(j - 2 * step);
        move.emplace_back(j, previous_target);
    }
}

// Pull moves of a residue inside the chain. `side` (+1 or -1) selects the
// chain neighbour m = k + side next to which k is moved; the chain is pulled
// on the opposite side.
void add_inner_pulls(const Conformation& c, int k, int side, std::vector<Move>& moves) {
    const Coord here = c.position(k);
    const Coord m = c.position(k + side);
    const int p = k - side;

    for (const Coord l : neighbours(m)) {
        if (!c.is_free(l) || !are_diagonal(l, here)) {
            continue;
        }
        const Coord corner{here.first + l.first - m.first, here.second + l.second - m.second};
        if (c.residue_at(corner) == p) {
            // Corner move. It is the same move for both values of `side`, so
            // it is listed only once.
            if (side == 1) {
                moves.push_back({{k, l}});
            }
        } else if (c.is_free(corner)) {
            Move move{{k, l}, {p, corner}};
            pull_rest(c, p, corner, -side, move);
            moves.push_back(std::move(move));
        }
    }
}

// Pull moves of the first or the last residue: the end of the chain advances
// by two nodes and drags the rest behind it.
void add_end_pulls(const Conformation& c, int k, std::vector<Move>& moves) {
    const int step = (k == 0) ? 1 : -1;  // from k towards the rest of the chain
    for (const Coord a : neighbours(c.position(k))) {
        if (!c.is_free(a)) {
            continue;
        }
        for (const Coord b : neighbours(a)) {
            if (!c.is_free(b)) {
                continue;
            }
            Move move{{k, b}, {k + step, a}};
            pull_rest(c, k + step, a, step, move);
            moves.push_back(std::move(move));
        }
    }
}

}  // namespace

std::vector<Move> pull_moves(const Conformation& c, int k) {
    std::vector<Move> moves;
    const int n = c.size();
    if (n < 2 || k < 0 || k >= n) {
        return moves;
    }

    if (k == 0 || k == n - 1) {
        add_end_pulls(c, k, moves);
    } else {
        add_inner_pulls(c, k, 1, moves);
        add_inner_pulls(c, k, -1, moves);
    }
    return moves;
}

}  // namespace hp
