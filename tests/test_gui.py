"""Tests of the parts of the GUI module that do not need a window."""

import pytest

from hpfold import _core
from hpfold.conformation import load_conformation, validate_conformation

# The module imports Tkinter: skip these tests where it is not installed.
pytest.importorskip("tkinter")

from hpfold import gui  # noqa: E402


def test_pixel_to_cell_snaps_to_the_closest_node():
    origin = (60, 60)
    assert gui.pixel_to_cell(60, 60, 60, origin) == (0, 0)
    assert gui.pixel_to_cell(85, 40, 60, origin) == (0, 0)
    assert gui.pixel_to_cell(95, 60, 60, origin) == (1, 0)
    assert gui.pixel_to_cell(5, 5, 60, origin) == (-1, -1)  # outside the grid


def test_pixel_and_cell_conversions_are_inverse():
    for cell in [(0, 0), (3, 5), (7, 7)]:
        pixel = gui.cell_to_pixel(cell, 37.5, (12.5, 80))
        assert gui.pixel_to_cell(*pixel, 37.5, (12.5, 80)) == cell


def test_fit_grid_fills_a_canvas_with_the_same_proportions():
    # 8x8 nodes plus one cell of margin per side: 9 cells in each direction.
    assert gui.fit_grid(540, 540, 8, 8) == (60, (60, 60))
    assert gui.fit_grid(270, 270, 8, 8) == (30, (30, 30))


def test_fit_grid_is_limited_by_the_shorter_side_and_centred():
    cell_size, origin = gui.fit_grid(900, 540, 8, 8)
    assert cell_size == 60
    # The grid spans 7 cells (420 px): the rest is split between the two sides.
    assert origin == (240, 60)

    cell_size, origin = gui.fit_grid(540, 900, 8, 8)
    assert cell_size == 60
    assert origin == (60, 240)


def test_fit_grid_keeps_the_whole_grid_inside_the_canvas():
    cell_size, origin = gui.fit_grid(333, 211, 5, 3)
    right, bottom = gui.cell_to_pixel((4, 2), cell_size, origin)
    assert origin[0] >= cell_size / 2 and origin[1] >= cell_size / 2
    assert right <= 333 - cell_size / 2 and bottom <= 211 - cell_size / 2


def test_hh_contacts_skip_consecutive_and_polar_residues():
    # A square of 4 residues closed by a contact between 0 and 3, and a P at
    # the end of the chain next to residue 0.
    coordinates = [(0, 0), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]
    assert gui.hh_contacts("HHHHHP", coordinates) == [(0, 3)]
    assert gui.hh_contacts("HHHHHH", coordinates) == [(0, 3), (0, 5)]
    assert gui.hh_contacts("PHHPHP", coordinates) == []


def test_interpolate_moves_the_residues_in_a_straight_line():
    a = [(0, 0), (1, 0)]
    b = [(0, 0), (1, 2)]
    assert gui.interpolate(a, b, 0) == [(0, 0), (1, 0)]
    assert gui.interpolate(a, b, 0.5) == [(0, 0), (1, 1)]
    assert gui.interpolate(a, b, 1) == [(0, 0), (1, 2)]


def make_config(tmp_path, sequence="HPHPPH", length=6):
    return {
        # The GUI does not use [lattice]: a small grid shows that it is ignored.
        "lattice": {"width": 4, "height": 4},
        "protein": {
            "length": length,
            "sequence": sequence,
            "initial_conformation": str(tmp_path / "initial.json"),
        },
    }


def test_grid_side_is_the_lattice_of_the_dataset(tmp_path):
    for length in (3, 6, 16):
        config = make_config(tmp_path, sequence="H" * length, length=length)
        assert gui.grid_side(config) == length + 4 == _core.free_lattice_side(length)


def test_write_default_conformation(tmp_path):
    path = gui.write_default_conformation(make_config(tmp_path))

    sequence, coordinates = load_conformation(path)
    assert sequence == "HPHPPH"
    validate_conformation(sequence, coordinates, 10, 10)  # 6 residues + 4


def test_write_default_conformation_checks_the_length(tmp_path):
    config = make_config(tmp_path, length=7)
    with pytest.raises(ValueError):
        gui.write_default_conformation(config)
    assert not (tmp_path / "initial.json").exists()


def test_save_folding_gif(tmp_path):
    from PIL import Image

    frames = [
        [(0, 0), (1, 0), (2, 0), (3, 0)],
        [(0, 0), (1, 0), (2, 0), (2, 1)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
    ]
    path = tmp_path / "folding.gif"
    gui.save_folding_gif(
        path, "HPPH", frames, ["Start", "Move 1/2", "Move 2/2"],
        {"H": "#E53935", "P": "#1E88E5"}, 16 / 60, move_ms=600, pause_ms=700,
    )
    with Image.open(path) as image:
        assert image.size == (
            gui.GIF_SIDE, gui.GIF_LEGEND_HEIGHT + gui.GIF_SIDE + gui.GIF_CAPTION_HEIGHT
        )
        # Every frame held, then the intermediate positions of each move.
        assert image.n_frames == len(frames) + (len(frames) - 1) * (gui.GIF_TWEEN_FRAMES - 1)
        assert image.info["loop"] == 0
        assert image.info["duration"] == 700
        image.seek(image.n_frames - 1)
        assert image.info["duration"] == 700 * gui.GIF_END_PAUSES
