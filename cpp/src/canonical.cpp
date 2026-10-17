#include "hp/canonical.hpp"

#include <algorithm>
#include <stdexcept>

namespace hp {

namespace {

Coord bond(const Conformation& c, int i) {
    return {c.position(i + 1).first - c.position(i).first,
            c.position(i + 1).second - c.position(i).second};
}

}  // namespace

Coord turn_left(Coord d) { return {-d.second, d.first}; }

Coord turn_right(Coord d) { return {d.second, -d.first}; }

std::vector<int> turns(const Conformation& c) {
    std::vector<int> t;
    if (c.size() < 3) {
        return t;
    }
    t.reserve(c.size() - 2);
    for (int i = 1; i + 1 < c.size(); ++i) {
        const Coord before = bond(c, i - 1);
        const Coord after = bond(c, i);
        if (after == before) {
            t.push_back(FORWARD);
        } else if (after == turn_left(before)) {
            t.push_back(LEFT);
        } else {
            t.push_back(RIGHT);
        }
    }
    return t;
}

std::vector<int> mirror_turns(std::vector<int> t) {
    for (int& turn : t) {
        if (turn == LEFT) {
            turn = RIGHT;
        } else if (turn == RIGHT) {
            turn = LEFT;
        }
    }
    return t;
}

std::vector<int> canonical_turns(const Conformation& c) {
    std::vector<int> t = turns(c);
    const auto first_turn = std::find_if(t.begin(), t.end(), [](int x) { return x != FORWARD; });
    if (first_turn != t.end() && *first_turn == RIGHT) {
        t = mirror_turns(std::move(t));
    }
    return t;
}

std::uint64_t turns_key(const std::vector<int>& t) {
    if (static_cast<int>(t.size()) > MAX_KEY_LENGTH - 2) {
        throw std::invalid_argument("the chain is too long for a 64-bit key (at most " +
                                    std::to_string(MAX_KEY_LENGTH) + " residues)");
    }
    std::uint64_t key = 0;
    for (const int turn : t) {
        key = 3 * key + static_cast<std::uint64_t>(turn);
    }
    return key;
}

std::vector<int> key_turns(std::uint64_t key, int n) {
    std::vector<int> t(std::max(n - 2, 0));
    for (int i = static_cast<int>(t.size()) - 1; i >= 0; --i) {
        t[i] = static_cast<int>(key % 3);
        key /= 3;
    }
    return t;
}

std::uint64_t canonical_key(const Conformation& c) {
    if (c.size() > MAX_KEY_LENGTH) {
        throw std::invalid_argument("the chain is too long for a 64-bit key (at most " +
                                    std::to_string(MAX_KEY_LENGTH) + " residues)");
    }
    // Same result as turns_key(canonical_turns(c)), without building the
    // lists: keys of equal length compare as the turns do, so the canonical
    // key is the smaller of the key of the chain and the key of its mirror.
    std::uint64_t key = 0;
    std::uint64_t mirror = 0;
    for (int i = 1; i + 1 < c.size(); ++i) {
        const Coord before = bond(c, i - 1);
        const Coord after = bond(c, i);
        const std::uint64_t turn = after == before ? FORWARD
                                   : after == turn_left(before) ? LEFT
                                                                : RIGHT;
        key = 3 * key + turn;
        mirror = 3 * mirror + (turn == FORWARD ? FORWARD : 3 - turn);
    }
    return std::min(key, mirror);
}

int free_lattice_side(int n) { return n + 4; }

Conformation conformation_from_turns(const std::string& sequence, const std::vector<int>& t) {
    const int n = static_cast<int>(sequence.size());
    if (n < 2 || static_cast<int>(t.size()) != n - 2) {
        throw std::invalid_argument("a chain of n residues needs n - 2 turns (n >= 2)");
    }

    // Walk from the origin, first bond to the right.
    std::vector<Coord> coordinates{{0, 0}, {1, 0}};
    Coord direction{1, 0};
    for (const int turn : t) {
        if (turn == LEFT) {
            direction = turn_left(direction);
        } else if (turn == RIGHT) {
            direction = turn_right(direction);
        } else if (turn != FORWARD) {
            throw std::invalid_argument("a turn must be FORWARD, LEFT or RIGHT");
        }
        const Coord last = coordinates.back();
        coordinates.push_back({last.first + direction.first, last.second + direction.second});
    }

    // Centre the box that contains the chain.
    int min_x = 0, max_x = 0, min_y = 0, max_y = 0;
    for (const Coord c : coordinates) {
        min_x = std::min(min_x, c.first);
        max_x = std::max(max_x, c.first);
        min_y = std::min(min_y, c.second);
        max_y = std::max(max_y, c.second);
    }
    const int side = free_lattice_side(n);
    const int shift_x = (side - (max_x - min_x + 1)) / 2 - min_x;
    const int shift_y = (side - (max_y - min_y + 1)) / 2 - min_y;
    for (Coord& c : coordinates) {
        c = {c.first + shift_x, c.second + shift_y};
    }
    // The constructor rejects walks that are not self-avoiding.
    return Conformation(sequence, std::move(coordinates), side, side);
}

Conformation conformation_from_key(const std::string& sequence, std::uint64_t key) {
    return conformation_from_turns(sequence,
                                   key_turns(key, static_cast<int>(sequence.size())));
}

}  // namespace hp
