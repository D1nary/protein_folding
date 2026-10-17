#include "hp/actions.hpp"

#include <algorithm>
#include <stdexcept>

#include "hp/canonical.hpp"
#include "hp/pull_moves.hpp"
#include "hp/vshd_moves.hpp"

namespace hp {

namespace {

constexpr int BACKWARDS = -1;

Coord difference(Coord a, Coord b) { return {a.first - b.first, a.second - b.second}; }

// Direction `d` seen from the direction `reference`: FORWARD, LEFT, RIGHT or
// BACKWARDS. Both are unit vectors of the lattice.
int relative_direction(Coord reference, Coord d) {
    if (d == reference) {
        return FORWARD;
    }
    if (d == turn_left(reference)) {
        return LEFT;
    }
    if (d == turn_right(reference)) {
        return RIGHT;
    }
    return BACKWARDS;
}

int mirror_letter(int letter) {
    return letter == LEFT ? RIGHT : (letter == RIGHT ? LEFT : letter);
}

[[noreturn]] void wrong_move(int k) {
    throw std::logic_error("the move of residue " + std::to_string(k) +
                           " does not match its move type");
}

bool same_move(Move a, Move b) {
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    return a == b;
}

int classify_vshd(const Conformation& c, int k, const Move& move) {
    const int n = c.size();
    if (move.size() == 1 && move[0].first == k) {
        if (k != 0 && k != n - 1) {
            return action_index(n, k, CORNER);
        }
        const Coord anchor = c.position(k == 0 ? 1 : n - 2);
        switch (relative_direction(difference(c.position(k), anchor),
                                   difference(move[0].second, anchor))) {
            case LEFT:
                return action_index(n, k, END_L);
            case RIGHT:
                return action_index(n, k, END_R);
            case BACKWARDS:
                return action_index(n, k, END_B);
            default:
                wrong_move(k);
        }
    }
    if (move.size() == 2) {
        const int j = move[0].first;
        if ((j == k || j == k - 1) && move[1].first == j + 1) {
            return action_index(n, j, CRANKSHAFT);
        }
    }
    wrong_move(k);
}

int classify_pull(const Conformation& c, int k, const Move& move) {
    const int n = c.size();
    if (move.empty() || move[0].first != k) {
        wrong_move(k);
    }
    if (k == 0 || k == n - 1) {
        // move = {k -> B, neighbour -> A, ...}
        const int neighbour = (k == 0) ? 1 : n - 2;
        if (move.size() < 2 || move[1].first != neighbour) {
            wrong_move(k);
        }
        const Coord a = move[1].second;
        const Coord b = move[0].second;
        const Coord towards_a = difference(a, c.position(k));
        const int first =
            relative_direction(difference(c.position(k), c.position(neighbour)), towards_a);
        const int second = relative_direction(towards_a, difference(b, a));
        if (first == BACKWARDS || second == BACKWARDS) {
            wrong_move(k);
        }
        return action_index(n, k, PULL_END_FF + 3 * first + second);
    }
    if (move.size() == 1) {
        // The pull move that ends at once is the corner move.
        return action_index(n, k, CORNER);
    }
    // move = {k -> L, p -> C, ...}, with m = k + side on the other side of k.
    const int side = k - move[1].first;
    if (side != 1 && side != -1) {
        wrong_move(k);
    }
    const Coord m = c.position(k + side);
    const int letter = relative_direction(difference(m, c.position(k)), difference(move[0].second, m));
    if (letter == LEFT) {
        return action_index(n, k, side == 1 ? PULL_NEXT_L : PULL_PREV_L);
    }
    if (letter == RIGHT) {
        return action_index(n, k, side == 1 ? PULL_NEXT_R : PULL_PREV_R);
    }
    wrong_move(k);
}

}  // namespace

std::string move_type_name(int type) {
    static const char* const names[N_MOVE_TYPES] = {
        "END_L",       "END_R",       "END_B",       "CORNER",      "CRANKSHAFT",
        "PULL_NEXT_L", "PULL_NEXT_R", "PULL_PREV_L", "PULL_PREV_R", "PULL_END_FF",
        "PULL_END_FL", "PULL_END_FR", "PULL_END_LF", "PULL_END_LL", "PULL_END_LR",
        "PULL_END_RF", "PULL_END_RL", "PULL_END_RR",
    };
    if (type < 0 || type >= N_MOVE_TYPES) {
        throw std::invalid_argument("unknown move type " + std::to_string(type));
    }
    return names[type];
}

int mirror_move_type(int type) {
    switch (type) {
        case END_L:
            return END_R;
        case END_R:
            return END_L;
        case PULL_NEXT_L:
            return PULL_NEXT_R;
        case PULL_NEXT_R:
            return PULL_NEXT_L;
        case PULL_PREV_L:
            return PULL_PREV_R;
        case PULL_PREV_R:
            return PULL_PREV_L;
        default:
            break;
    }
    if (type >= PULL_END_FF && type < N_MOVE_TYPES) {
        const int first = (type - PULL_END_FF) / 3;
        const int second = (type - PULL_END_FF) % 3;
        return PULL_END_FF + 3 * mirror_letter(first) + mirror_letter(second);
    }
    return type;  // END_B, CORNER, CRANKSHAFT
}

int action_count(int n) {
    if (n < 3) {
        throw std::invalid_argument("the action space needs at least 3 residues");
    }
    return 6 * n + 11;
}

int action_index(int n, int residue, int type) {
    if (residue < 0 || residue >= n || type < 0 || type >= N_MOVE_TYPES) {
        return -1;
    }
    if (residue == 0 || residue == n - 1) {
        // 12 slots: END_L, END_R, END_B, then the 9 end pulls.
        const int base = (residue == 0) ? 0 : 6 * n - 1;
        if (type <= END_B) {
            return base + type;
        }
        if (type >= PULL_END_FF) {
            return base + 3 + (type - PULL_END_FF);
        }
        return -1;
    }
    // Inner residue: CORNER, CRANKSHAFT (only up to N-3), then the 4 pulls.
    const int base = 12 + 6 * (residue - 1);
    const bool has_crankshaft = residue <= n - 3;
    if (type == CORNER) {
        return base;
    }
    if (type == CRANKSHAFT) {
        return has_crankshaft ? base + 1 : -1;
    }
    if (type >= PULL_NEXT_L && type <= PULL_PREV_R) {
        return base + (has_crankshaft ? 2 : 1) + (type - PULL_NEXT_L);
    }
    return -1;
}

std::vector<Action> action_table(int n) {
    std::vector<Action> table(action_count(n));
    for (int residue = 0; residue < n; ++residue) {
        for (int type = 0; type < N_MOVE_TYPES; ++type) {
            const int a = action_index(n, residue, type);
            if (a != -1) {
                table[a] = {residue, type};
            }
        }
    }
    return table;
}

std::vector<int> mirror_permutation(int n) {
    const std::vector<Action> table = action_table(n);
    std::vector<int> permutation(table.size());
    for (std::size_t a = 0; a < table.size(); ++a) {
        permutation[a] = action_index(n, table[a].residue, mirror_move_type(table[a].type));
    }
    return permutation;
}

int classify_move(const Conformation& c, int k, const Move& move, bool pull) {
    return pull ? classify_pull(c, k, move) : classify_vshd(c, k, move);
}

std::vector<Move> action_moves(const Conformation& c) {
    const int n = c.size();
    std::vector<Move> moves(action_count(n));
    auto put = [&](int k, const Move& move, bool pull) {
        Move& slot = moves[classify_move(c, k, move, pull)];
        if (slot.empty()) {
            slot = move;
        } else if (!same_move(slot, move)) {
            throw std::logic_error("two different moves of residue " + std::to_string(k) +
                                   " fall in the same slot");
        }
    };
    for (int k = 0; k < n; ++k) {
        for (const Move& move : vshd_moves(c, k)) {
            put(k, move, false);
        }
        for (const Move& move : pull_moves(c, k)) {
            put(k, move, true);
        }
    }
    return moves;
}

}  // namespace hp
