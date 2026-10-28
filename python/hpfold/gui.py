"""Phase 0 — GUI (Tkinter) to draw the initial configuration by hand.

The drawn configuration is saved to the file given by
``protein.initial_conformation``.

This module contains only the window: the state of the chain, the validation
of the moves and the JSON file are handled by ``hpfold.conformation``. It also
holds the window that animates the path to the native state (phase
``discriminator_score``, FoldingAnimation) and saves it as an animated GIF
(save_folding_gif).
"""

import functools
import math
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, messagebox
from tkinter import font as tkfont

from PIL import Image, ImageDraw, ImageFont

from hpfold import _core
from hpfold.conformation import (
    RESIDUE_TYPES,
    Chain,
    Coord,
    MoveError,
    default_coordinates,
    save_conformation,
    validate_conformation,
)

# Text shown in the status bar for each refused move.
MOVE_ERROR_MESSAGES = {
    MoveError.CHAIN_COMPLETE: "The chain is already complete: no more residues can be added.",
    MoveError.OUTSIDE_GRID: "Click outside the grid.",
    MoveError.OCCUPIED: "Cell already occupied by another residue.",
    MoveError.NOT_ADJACENT: "The cell must be adjacent (up/down/left/right) to the last residue.",
}

# Colours of the windows. The colours of the residues are in [gui] of the
# configuration file.
THEME = {
    "window": "#F0F2F5",       # background of the windows and of the panels
    "grid": "#F8F9FA",         # background of the grid
    "surface": "#FFFFFF",      # boxes and status bar
    "border": "#E4E7EC",       # thin border around the grid and the boxes
    "text": "#1D2939",
    "text_muted": "#667085",
    "value": "#101828",        # numbers in the boxes
    "value_label": "#475467",  # labels of the numbers
    "button": "#E2E8F0",
    "button_active": "#CBD5E1",
    "button_text": "#334155",
    "button_primary": "#2563EB",  # main action of the window
    "button_primary_active": "#1D4ED8",
    "button_primary_text": "#FFFFFF",
    "button_disabled": "#F1F5F9",
    "text_disabled": "#94A3B8",
    "lattice": "#D0D5DD",      # lines and nodes of the grid
    "bond": "#475467",         # bonds of the chain and H-H contacts
    # Same colour as the grid: a thin gap between a residue and its bonds.
    "residue_outline": "#F8F9FA",
    "residue_label": "#FFFFFF",
}
# Size in points of the text of every widget (the default of Tk is 10).
FONT_SIZE = 12
TITLE_FONT = ("TkDefaultFont", FONT_SIZE + 2, "bold")
VALUE_FONT = ("TkDefaultFont", FONT_SIZE, "bold")
CARD_TITLE_FONT = ("TkDefaultFont", FONT_SIZE + 1, "bold")
# Width in pixels of the explanations in the side panels, and of the
# labels of KeyValueList: longer ones go on more lines.
PANEL_TEXT_WIDTH = 270
LABEL_WIDTH = 170
# Names of the residue types in the buttons of step 2 and in the legend.
RESIDUE_NAMES = {"H": "H — hydrophobic", "P": "P — polar"}


def apply_theme(root: tk.Tk) -> None:
    """Flat light theme for every widget created afterwards under `root`.

    Uses the option database of Tk, so the widgets do not need to set their
    colours one by one; the ones that differ (grid, boxes) set them explicitly.
    """
    t = THEME
    root.configure(background=t["window"])
    for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
        tkfont.nametofont(name, root=root).configure(size=FONT_SIZE)
    for pattern, value in (
        ("*background", t["window"]),
        ("*foreground", t["text"]),
        ("*activeBackground", t["button_active"]),
        ("*activeForeground", t["text"]),
        ("*disabledForeground", t["text_disabled"]),
        ("*highlightThickness", 0),
        # Also for the buttons of the Tk dialogs (messagebox).
        ("*Button.background", t["button"]),
        ("*Button.foreground", t["button_text"]),
        ("*Button.relief", "flat"),
        ("*Button.borderWidth", 0),
        ("*Button.padX", 10),
        ("*Button.padY", 6),
    ):
        root.option_add(pattern, value)


class FlatButton(tk.Button):
    """Flat button; `primary` for the main action of the window.

    Tk changes only the colour of the text of a disabled button:
    set_enabled() changes also its background.
    """

    def __init__(self, master: tk.Misc, primary: bool = False, **options) -> None:
        t = THEME
        if primary:
            self.enabled_background = t["button_primary"]
            colors = dict(
                foreground=t["button_primary_text"],
                activebackground=t["button_primary_active"],
                activeforeground=t["button_primary_text"],
                font=VALUE_FONT,
            )
        else:
            self.enabled_background = t["button"]
            colors = dict(
                foreground=t["button_text"],
                activebackground=t["button_active"],
                activeforeground=t["button_text"],
            )
        super().__init__(
            master,
            background=self.enabled_background,
            disabledforeground=t["text_disabled"],
            cursor="hand2",
            **colors,
            **options,
        )

    def set_enabled(self, enabled: bool) -> None:
        self.config(
            state="normal" if enabled else "disabled",
            background=self.enabled_background if enabled else THEME["button_disabled"],
            cursor="hand2" if enabled else "",
        )


class KeyValueList(tk.Frame):
    """Lines "label: value" in two columns, the value in bold.

    A line without ": " takes both columns, in grey.
    """

    def __init__(self, master: tk.Misc, background: str = THEME["window"]) -> None:
        super().__init__(master, background=background)
        self.text = ""

    def set_text(self, text: str) -> None:
        self.text = text
        for child in self.winfo_children():
            child.destroy()
        background = self["background"]
        for row, line in enumerate(text.splitlines()):
            label, separator, value = line.partition(": ")
            if not separator:
                tk.Label(
                    self, text=line, background=background, foreground=THEME["text_muted"]
                ).grid(row=row, column=0, columnspan=2, sticky="w")
                continue
            tk.Label(
                self, text=label, background=background, foreground=THEME["value_label"],
                wraplength=LABEL_WIDTH, justify="left",
            ).grid(row=row, column=0, sticky="nw", padx=(0, 12))
            tk.Label(
                self, text=value, background=background, foreground=THEME["value"],
                font=VALUE_FONT,
            ).grid(row=row, column=1, sticky="nw")


def card(master: tk.Misc, title: str) -> tk.Frame:
    """White box with a bold title; the caller packs it and puts the content
    in it (with background THEME["surface"])."""
    t = THEME
    box = tk.Frame(
        master, background=t["surface"], padx=16, pady=14,
        highlightthickness=1, highlightbackground=t["border"], highlightcolor=t["border"],
    )
    tk.Label(
        box, text=title, background=t["surface"],
        foreground=t["text"], font=CARD_TITLE_FONT,
    ).pack(anchor="w", pady=(0, 10))
    return box


def hint(master: tk.Misc, text: str) -> tk.Label:
    """Grey explanation inside a box, wrapped to the width of the panel."""
    return tk.Label(
        master, text=text, justify="left", wraplength=PANEL_TEXT_WIDTH,
        background=THEME["surface"], foreground=THEME["text_muted"],
    )


def pack_box(box: tk.Frame, side: str) -> None:
    """Packs a box at the bottom of its panel ("bottom"), above the boxes
    already packed there, or under the ones at the top ("top")."""
    box.pack(side=side, fill="x", pady=(12, 0) if side == "bottom" else (0, 10))


def prediction_box(master: tk.Misc, side: str = "bottom") -> KeyValueList:
    """Box "Model prediction" in `master`, packed as pack_box.

    Returns the list inside it, which receives the text of the outputs.
    """
    box = card(master, "Model prediction")
    pack_box(box, side)
    values = KeyValueList(box, background=THEME["surface"])
    values.pack(fill="x")
    return values


def swatch(master: tk.Misc, color: str) -> tk.Canvas:
    """Small circle of the colour of a residue, on the background of a box."""
    canvas = tk.Canvas(
        master, width=16, height=16, highlightthickness=0, background=THEME["surface"]
    )
    canvas.create_oval(2, 2, 14, 14, fill=color, outline=THEME["surface"])
    return canvas


def legend_box(master: tk.Misc, colors: dict[str, str], side: str = "bottom") -> tk.Frame:
    """Box "Legend" with the colour of each residue type, packed as pack_box.

    At the bottom it takes the free space under the controls. The types are
    on one line, to take as little height as possible.
    """
    box = card(master, "Legend")
    pack_box(box, side)
    row = tk.Frame(box, background=THEME["surface"])
    row.pack(anchor="w")
    for k, residue_type in enumerate(RESIDUE_TYPES):
        swatch(row, colors[residue_type]).pack(side="left", padx=(0 if k == 0 else 16, 0))
        tk.Label(
            row, text=RESIDUE_NAMES[residue_type], background=THEME["surface"],
            foreground=THEME["value_label"],
        ).pack(side="left", padx=(6, 0))
    return box


def status_bar(master: tk.Misc, textvariable: tk.StringVar) -> tk.Label:
    """Line of text at the bottom of a window."""
    return tk.Label(
        master, textvariable=textvariable, anchor="w", padx=8, pady=4,
        background=THEME["surface"], foreground=THEME["text_muted"],
    )


def grid_canvas(master: tk.Misc, width: float, height: float) -> tk.Canvas:
    """Canvas with a thin border, on which the lattice is drawn."""
    return tk.Canvas(
        master,
        width=width,
        height=height,
        background=THEME["grid"],
        highlightthickness=1,
        highlightbackground=THEME["border"],
        highlightcolor=THEME["border"],
    )


def fit_grid(
    canvas_width: float, canvas_height: float, width: int, height: int
) -> tuple[float, tuple[float, float]]:
    """Largest grid that fits in a canvas of the given size, centred in it.

    Returns the distance in pixels between two adjacent nodes and the canvas
    coordinates of the node (0, 0). One cell of margin is left on every side,
    so that the residues on the border are drawn entirely.
    """
    cell_size = min(canvas_width / (width + 1), canvas_height / (height + 1))
    origin = (
        (canvas_width - (width - 1) * cell_size) / 2,
        (canvas_height - (height - 1) * cell_size) / 2,
    )
    return cell_size, origin


def pixel_to_cell(
    px: float, py: float, cell_size: float, origin: tuple[float, float]
) -> Coord:
    """Grid node closest to the canvas point (px, py).

    `origin` is the canvas position of the node (0, 0). The result can lie
    outside the grid: it is up to the caller to check it.
    """
    return round((px - origin[0]) / cell_size), round((py - origin[1]) / cell_size)


def grid_side(config: dict) -> int:
    """Side of the square grid of the GUI: N + 4 nodes for a chain of N residues.

    It is the lattice on which the dataset (phase 2) places every state
    (free_lattice_side in cpp/include/hp/canonical.hpp), so the chain is drawn
    on a grid of the same size. It does not depend on [lattice], which is the
    grid of the REMC.
    """
    return _core.free_lattice_side(config["protein"]["length"])


def cell_to_pixel(
    cell: Coord, cell_size: float, origin: tuple[float, float]
) -> tuple[float, float]:
    """Canvas coordinates of the centre of a grid node."""
    x, y = cell
    return origin[0] + x * cell_size, origin[1] + y * cell_size


class ConformationEditor:
    """Window with the grid, the controls of the two steps and the status bar.

    By default the drawn conformation is saved to ``protein.initial_conformation``
    (phase 0). With `on_done` the last button calls ``on_done(sequence,
    coordinates)`` instead and the window stays open, so that more
    conformations can be drawn one after the other (phase
    ``discriminator_score``); `length` then replaces ``protein.length``, and
    a box in a column on the right shows the text given to show_result().
    """

    def __init__(
        self,
        root: tk.Tk,
        config: dict,
        length: int | None = None,
        on_done: Callable[[str, list[Coord]], None] | None = None,
        done_text: str = "Save",
        title: str = "hpfold — initial conformation",
    ) -> None:
        self.root = root
        self.output_path = Path(config["protein"]["initial_conformation"])
        self.on_done = on_done
        self.done_text = done_text
        self.title = title
        length = config["protein"]["length"] if length is None else length
        side = _core.free_lattice_side(length)
        self.chain = Chain(side, side, length)

        gui = config["gui"]
        # Size of the grid when the window opens. Resizing the window changes
        # cell_size and origin (see _on_resize); the residues keep the same
        # proportion to the cells.
        self.initial_cell_size = gui["cell_size"]
        self.radius_ratio = gui["residue_radius"] / gui["cell_size"]
        self.cell_size, self.origin = fit_grid(
            (self.chain.width + 1) * self.initial_cell_size,
            (self.chain.height + 1) * self.initial_cell_size,
            self.chain.width,
            self.chain.height,
        )
        self.color_untyped = gui["color_untyped"]
        self.colors = {"H": gui["color_h"], "P": gui["color_p"]}

        # Current step: 1 = drawing the chain, 2 = assigning the H/P types.
        self.step = 1
        # Type assigned by a click during step 2.
        self.active_type = tk.StringVar(value=RESIDUE_TYPES[0])
        self.status = tk.StringVar()

        self._build_widgets()
        self._refresh()
        self.status.set("Step 1: click on a node to place the first residue.")

    # ------------------------------------------------------------------
    # Construction of the window

    def _build_widgets(self) -> None:
        self.root.title(self.title)

        self.canvas = grid_canvas(
            self.root,
            (self.chain.width + 1) * self.initial_cell_size,
            (self.chain.height + 1) * self.initial_cell_size,
        )
        # The canvas takes all the space gained when the window is enlarged;
        # the panel on the right keeps its natural size.
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Configure>", self._on_resize)

        panel = tk.Frame(self.root)
        panel.grid(row=0, column=1, sticky="ns", padx=8, pady=8)

        # One box per group of controls; the box of the current step has a
        # blue border (see _refresh).
        t = THEME
        box = card(panel, "Progress")
        box.pack(fill="x", pady=(0, 10))
        self.info = KeyValueList(box, background=t["surface"])
        self.info.pack(fill="x")

        # Step 1 controls.
        box = card(panel, "Step 1 · Shape of the chain")
        box.pack(fill="x", pady=(0, 10))
        hint(box, "Click on a node next to the last residue.").pack(anchor="w", pady=(0, 8))
        self.undo_button = FlatButton(box, text="Undo last residue", command=self._undo)
        self.undo_button.pack(fill="x", pady=(0, 4))
        self.clear_button = FlatButton(box, text="Clear all", command=self._clear)
        self.clear_button.pack(fill="x", pady=(0, 4))
        self.to_step2_button = FlatButton(
            box, text="Go to step 2 (H/P) →", command=lambda: self._set_step(2)
        )
        self.to_step2_button.pack(fill="x")
        self.step_boxes = {1: box}

        # Step 2 controls.
        box = card(panel, "Step 2 · Type of the residues")
        box.pack(fill="x", pady=(0, 10))
        hint(box, "Click on a residue to give it the selected type (H/P keys to switch).").pack(
            anchor="w", pady=(0, 8)
        )
        # One button per type, coloured like its residues (see
        # _update_type_buttons): the active one is filled with the colour.
        self.type_buttons: dict[str, tk.Button] = {}
        for residue_type in RESIDUE_TYPES:
            button = tk.Button(
                box,
                font=VALUE_FONT,
                relief="flat",
                borderwidth=0,
                highlightthickness=2,
                disabledforeground=t["text_disabled"],
                command=lambda rt=residue_type: self.active_type.set(rt),
            )
            button.pack(fill="x", pady=(0, 4))
            self.type_buttons[residue_type] = button
            for key in (residue_type.lower(), residue_type):
                self.root.bind(
                    f"<KeyPress-{key}>", lambda event, rt=residue_type: self._choose_type(rt)
                )
        self.active_type.trace_add("write", lambda *args: self._update_type_buttons())
        row = tk.Frame(box, background=t["surface"])
        row.pack(anchor="w", pady=(2, 0))
        swatch(row, self.color_untyped).pack(side="left")
        tk.Label(
            row, text="type not assigned", background=t["surface"], foreground=t["text_muted"]
        ).pack(side="left", padx=(4, 0))
        self.to_step1_button = FlatButton(
            box, text="← Back to step 1", command=lambda: self._set_step(1)
        )
        self.to_step1_button.pack(fill="x", pady=(8, 0))
        self.step_boxes[2] = box

        # Main action, out of the boxes.
        self.save_button = FlatButton(
            panel, primary=True, text=self.done_text, command=self._save
        )
        self.save_button.pack(fill="x", ipady=3)

        # Result of on_done in a second column, with the legend: under the
        # controls they would not fit in the height of a 1080p screen.
        self.result: KeyValueList | None = None
        if self.on_done is None:
            legend_box(panel, self.colors)
        else:
            column = tk.Frame(self.root)
            column.grid(row=0, column=2, sticky="ns", padx=(0, 8), pady=8)
            legend_box(column, self.colors, side="top")
            self.result = prediction_box(column, side="top")
            self.result_shown = False

        status_bar(self.root, self.status).grid(row=1, column=0, columnspan=3, sticky="ew")

    # ------------------------------------------------------------------
    # Event handlers

    def _on_click(self, event: tk.Event) -> None:
        cell = pixel_to_cell(event.x, event.y, self.cell_size, self.origin)
        if self.step == 1:
            self._place_residue(cell)
        else:
            self._assign_type(cell)
        self._refresh()

    def _on_resize(self, event: tk.Event) -> None:
        """Rescales the grid to the new size of the canvas."""
        self.cell_size, self.origin = fit_grid(
            event.width, event.height, self.chain.width, self.chain.height
        )
        self._draw()

    def _place_residue(self, cell: Coord) -> None:
        error = self.chain.add(cell)
        if error is not None:
            self.status.set(MOVE_ERROR_MESSAGES[error])
        elif self.chain.is_complete:
            self.status.set("Chain complete: go to step 2 to assign the H/P types.")
        else:
            self.status.set(f"Residue {len(self.chain.positions) - 1} placed at {cell}.")

    def _assign_type(self, cell: Coord) -> None:
        index = self.chain.index_at(cell)
        if index is None:
            self.status.set("No residue on this cell.")
            return
        self.chain.set_type(index, self.active_type.get())
        if self.chain.is_fully_typed:
            self.status.set(
                f"All residues have a type: press \"{self.done_text}\" to continue."
            )
        else:
            self.status.set(f"Residue {index}: type {self.active_type.get()}.")

    def _undo(self) -> None:
        if self.chain.undo():
            self.status.set("Last residue removed.")
        else:
            self.status.set("No residue to undo.")
        self._refresh()

    def _clear(self) -> None:
        self.chain.clear()
        self.status.set("Chain cleared.")
        self._refresh()

    def _choose_type(self, residue_type: str) -> None:
        """H/P keys: like the buttons, only during step 2."""
        if self.step == 2:
            self.active_type.set(residue_type)

    def _set_step(self, step: int) -> None:
        self.step = step
        if step == 1:
            self.status.set("Step 1: fix the shape of the chain.")
        else:
            self.status.set("Step 2: choose H or P and click on the residues.")
        self._refresh()

    def show_result(self, text: str) -> None:
        """Shows the result of on_done in the box on the right."""
        if self.result is not None:
            self.result.set_text(text)
            self.result_shown = True

    def _save(self) -> None:
        if self.on_done is not None:
            self.on_done(self.chain.sequence, list(self.chain.positions))
            return
        if self.output_path.exists() and not messagebox.askyesno(
            "Overwrite?", f"The file {self.output_path} already exists. Overwrite it?"
        ):
            return
        try:
            save_conformation(self.output_path, self.chain.sequence, self.chain.positions)
        except OSError as error:
            messagebox.showerror("Save error", str(error))
            return
        messagebox.showinfo(
            "Initial conformation", f"Conformation saved to {self.output_path}."
        )
        # The work of phase 0 is done: close the window, which ends mainloop().
        self.root.destroy()

    # ------------------------------------------------------------------
    # Drawing

    def _refresh(self) -> None:
        """Redraws the canvas and updates labels and button states."""
        self._draw()
        if self.result is not None:
            # Every action on the chain makes the last result old.
            if self.result_shown or not self.result.text:
                self.result.set_text(f'Press "{self.done_text}" to see\nwhat the model says.')
            self.result_shown = False

        chain = self.chain
        titles = {1: "1 of 2", 2: "2 of 2"}
        typed = sum(t is not None for t in chain.types)
        self.info.set_text(
            f"Step: {titles[self.step]}\n"
            f"Residues: {len(chain.positions)}/{chain.length}\n"
            f"With a type: {typed}/{len(chain.positions)}"
        )

        in_step1 = self.step == 1
        for step, box in self.step_boxes.items():
            color = THEME["button_primary"] if step == self.step else THEME["border"]
            box.config(highlightbackground=color, highlightcolor=color)
        self.undo_button.set_enabled(in_step1 and bool(chain.positions))
        self.clear_button.set_enabled(in_step1 and bool(chain.positions))
        self.to_step2_button.set_enabled(in_step1 and chain.is_complete)
        self._update_type_buttons()
        self.to_step1_button.set_enabled(not in_step1)
        self.save_button.set_enabled(chain.is_fully_typed)

    def _update_type_buttons(self) -> None:
        """The active type is filled with its colour and has a check mark;
        the other one has its colour on white. Grey during step 1."""
        t = THEME
        for residue_type, button in self.type_buttons.items():
            color = self.colors[residue_type]
            if self.step == 1:
                button.config(
                    state="disabled", text=RESIDUE_NAMES[residue_type], cursor="",
                    background=t["button_disabled"], highlightbackground=t["button_disabled"],
                    highlightcolor=t["button_disabled"],
                )
            elif residue_type == self.active_type.get():
                button.config(
                    state="normal", text=f"✓  {RESIDUE_NAMES[residue_type]}", cursor="hand2",
                    background=color, foreground=t["residue_label"],
                    activebackground=color, activeforeground=t["residue_label"],
                    highlightbackground=color, highlightcolor=color,
                )
            else:
                button.config(
                    state="normal", text=RESIDUE_NAMES[residue_type], cursor="hand2",
                    background=t["surface"], foreground=color,
                    activebackground=t["button_disabled"], activeforeground=color,
                    highlightbackground=t["border"], highlightcolor=t["border"],
                )

    def _pixel(self, cell: Coord) -> tuple[float, float]:
        return cell_to_pixel(cell, self.cell_size, self.origin)

    def _draw(self) -> None:
        chain = self.chain
        self.canvas.delete("all")
        draw_lattice(self.canvas, range(chain.width), range(chain.height), self._pixel)
        colors = [
            self.color_untyped if residue_type is None else self.colors[residue_type]
            for residue_type in chain.types
        ]
        draw_chain(
            self.canvas,
            [self._pixel(cell) for cell in chain.positions],
            colors,
            self.radius_ratio * self.cell_size,
        )


def draw_lattice(
    canvas: tk.Canvas,
    xs: range,
    ys: range,
    to_pixel: Callable[[Coord], tuple[float, float]],
) -> None:
    """Lattice of the nodes xs x ys: one line per row and per column, with a
    dot on every node."""
    left, top = to_pixel((xs[0], ys[0]))
    right, bottom = to_pixel((xs[-1], ys[-1]))
    for x in xs:
        px, _ = to_pixel((x, ys[0]))
        canvas.create_line(px, top, px, bottom, fill=THEME["lattice"])
    for y in ys:
        _, py = to_pixel((xs[0], y))
        canvas.create_line(left, py, right, py, fill=THEME["lattice"])
    for x in xs:
        for y in ys:
            px, py = to_pixel((x, y))
            canvas.create_oval(
                px - 2, py - 2, px + 2, py + 2, fill=THEME["lattice"], outline=""
            )


def draw_chain(
    canvas: tk.Canvas, points: list[tuple[float, float]], colors: list[str], r: float
) -> None:
    """Residues (circles of radius r with their index) at the canvas points,
    joined by the bonds of the chain."""
    # A negative font size is in pixels instead of points.
    font = ("TkDefaultFont", -max(1, round(r)), "bold")

    # Bonds between consecutive residues, drawn before the residues so that
    # the circles cover the ends of the segments.
    if len(points) > 1:
        canvas.create_line(
            *points, width=max(1, r / 4), fill=THEME["bond"], capstyle="round", joinstyle="round"
        )

    for index, ((px, py), color) in enumerate(zip(points, colors)):
        canvas.create_oval(
            px - r, py - r, px + r, py + r, fill=color, outline=THEME["residue_outline"]
        )
        canvas.create_text(px, py, text=str(index), fill=THEME["residue_label"], font=font)


def hh_contacts(sequence: str, coordinates: list[Coord]) -> list[tuple[int, int]]:
    """Pairs of H residues adjacent on the lattice but not consecutive in the
    chain: each one lowers the energy by 1."""
    index = {cell: i for i, cell in enumerate(coordinates)}
    pairs = []
    for i, (x, y) in enumerate(coordinates):
        for cell in ((x + 1, y), (x, y + 1)):
            j = index.get(cell)
            if j is not None and abs(i - j) > 1 and sequence[i] == sequence[j] == "H":
                pairs.append((min(i, j), max(i, j)))
    return sorted(pairs)


def interpolate(a: list[Coord], b: list[Coord], t: float) -> list[tuple[float, float]]:
    """Positions of the residues a fraction t (0 ... 1) of the way from a to b."""
    return [(xa + (xb - xa) * t, ya + (yb - ya) * t) for (xa, ya), (xb, yb) in zip(a, b)]


def lattice_bounds(frames: list[list[Coord]]) -> tuple[range, range]:
    """Lattice shown: the box of all the frames, with one free node around."""
    cells = [cell for frame in frames for cell in frame]
    xs = range(min(x for x, _ in cells) - 1, max(x for x, _ in cells) + 2)
    ys = range(min(y for _, y in cells) - 1, max(y for _, y in cells) + 2)
    return xs, ys


# Animated GIF of the path (save_folding_gif). Side in pixels of the grid,
# with a strip over it for the legend and one under it for the caption of
# the frame.
GIF_SIDE = 600
GIF_LEGEND_HEIGHT = 36
GIF_CAPTION_HEIGHT = 40
GIF_CAPTION_FONT_SIZE = 15  # pixels; smaller for a caption that does not fit
# The frames are drawn this many times larger and then scaled down: Pillow
# does not smooth the edges of what it draws.
GIF_SCALE = 2
# Intermediate positions drawn while the residues slide. Fewer than in the
# window: every image of the GIF is kept in memory until the file is written.
GIF_TWEEN_FRAMES = 12
# The native state is held this many times animation_pause_ms before the GIF
# starts again.
GIF_END_PAUSES = 3


@functools.lru_cache(maxsize=32)
def gif_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Font of `size` pixels for the GIF: DejaVu Sans if installed, otherwise
    the default font of Pillow."""
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size)


def dashed_line(
    draw: ImageDraw.ImageDraw,
    a: tuple[float, float],
    b: tuple[float, float],
    dash: tuple[float, float],
    width: int,
    fill: str,
) -> None:
    """Segment from a to b drawn as dashes of dash[0] pixels, dash[1] apart
    (Pillow has no dashed lines)."""
    (xa, ya), (xb, yb) = a, b
    length = math.hypot(xb - xa, yb - ya)
    on, off = dash
    d = 0.0
    while d < length:
        e = min(d + on, length)
        draw.line(
            [(xa + (xb - xa) * d / length, ya + (yb - ya) * d / length),
             (xa + (xb - xa) * e / length, ya + (yb - ya) * e / length)],
            fill=fill, width=width,
        )
        d += on + off


def save_folding_gif(
    path: Path | str,
    sequence: str,
    frames: list[list[Coord]],
    captions: list[str],
    colors: dict[str, str],
    radius_ratio: float,
    move_ms: int,
    pause_ms: int,
) -> None:
    """Writes the path to the native state to `path` as an animated GIF.

    Same drawing and timing as FoldingAnimation: every frame is held for
    `pause_ms` with its H-H contacts and its caption under the grid, then
    the residues slide to the next one in `move_ms`. Over the grid, the
    legend of the colours as in the box of the window. The GIF loops forever.
    """
    s = GIF_SCALE
    width, side = GIF_SIDE * s, GIF_SIDE * s
    legend_bottom = GIF_LEGEND_HEIGHT * s
    caption_top = legend_bottom + side
    height = caption_top + GIF_CAPTION_HEIGHT * s
    xs, ys = lattice_bounds(frames)
    cell_size, (ox, oy) = fit_grid(side, side, len(xs), len(ys))
    origin = (ox, oy + legend_bottom)
    r = radius_ratio * cell_size
    label_font = gif_font(max(1, round(r)), bold=True)
    residue_colors = [colors[c] for c in sequence]

    def to_pixel(cell: tuple[float, float]) -> tuple[float, float]:
        x, y = cell
        return cell_to_pixel((x - xs[0], y - ys[0]), cell_size, origin)

    def caption_font(text: str, draw: ImageDraw.ImageDraw) -> ImageFont.FreeTypeFont:
        size = GIF_CAPTION_FONT_SIZE * s
        while size > 6 * s and draw.textlength(text, font=gif_font(size)) > width - 24 * s:
            size -= s
        return gif_font(size)

    def draw_legend(draw: ImageDraw.ImageDraw) -> None:
        """Strip at the top: a circle and the name of each type, as
        legend_box."""
        draw.rectangle([0, 0, width, legend_bottom], fill=THEME["surface"])
        draw.line([(0, legend_bottom), (width, legend_bottom)], fill=THEME["border"], width=s)
        y = legend_bottom / 2
        x = 12 * s
        font = gif_font(GIF_CAPTION_FONT_SIZE * s)
        for residue_type in RESIDUE_TYPES:
            draw.ellipse([x, y - 6 * s, x + 12 * s, y + 6 * s], fill=colors[residue_type])
            x += 18 * s
            name = RESIDUE_NAMES[residue_type]
            draw.text((x, y), name, fill=THEME["value_label"], font=font, anchor="lm")
            x += draw.textlength(name, font=font) + 20 * s

    def render(k: int, target: int | None = None, t: float = 0.0) -> Image.Image:
        image = Image.new("RGB", (width, height), THEME["grid"])
        draw = ImageDraw.Draw(image)
        draw_legend(draw)

        # Lattice, as draw_lattice.
        left, top = to_pixel((xs[0], ys[0]))
        right, bottom = to_pixel((xs[-1], ys[-1]))
        for x in xs:
            px, _ = to_pixel((x, ys[0]))
            draw.line([(px, top), (px, bottom)], fill=THEME["lattice"], width=s)
        for y in ys:
            _, py = to_pixel((xs[0], y))
            draw.line([(left, py), (right, py)], fill=THEME["lattice"], width=s)
        for x in xs:
            for y in ys:
                px, py = to_pixel((x, y))
                draw.ellipse([px - 2 * s, py - 2 * s, px + 2 * s, py + 2 * s],
                             fill=THEME["lattice"])

        frame = frames[k]
        if target is not None:
            positions = interpolate(frame, frames[target], t)
        else:
            positions = frame
            # The contacts are drawn only when the chain is on the lattice.
            for i, j in hh_contacts(sequence, frame):
                dashed_line(draw, to_pixel(frame[i]), to_pixel(frame[j]), (4 * s, 4 * s),
                            max(1, round(r / 6)), THEME["bond"])

        # Chain, as draw_chain.
        points = [to_pixel(cell) for cell in positions]
        if len(points) > 1:
            draw.line(points, fill=THEME["bond"], width=max(1, round(r / 4)), joint="curve")
        for index, ((px, py), color) in enumerate(zip(points, residue_colors)):
            draw.ellipse([px - r, py - r, px + r, py + r], fill=color,
                         outline=THEME["residue_outline"], width=s)
            draw.text((px, py), str(index), fill=THEME["residue_label"], font=label_font,
                      anchor="mm")

        # Caption of the frame (the one of the frame left while sliding, as
        # the status bar of the window).
        draw.rectangle([0, caption_top, width, height], fill=THEME["surface"])
        draw.line([(0, caption_top), (width, caption_top)], fill=THEME["border"], width=s)
        caption = captions[k]
        draw.text((12 * s, (caption_top + height) / 2), caption, fill=THEME["text"],
                  font=caption_font(caption, draw), anchor="lm")

        image = image.resize((width // s, height // s), Image.Resampling.LANCZOS)
        # Palette image right away: much less memory than RGB.
        return image.convert("P", palette=Image.Palette.ADAPTIVE)

    images, durations = [], []
    for k in range(len(frames)):
        last = k == len(frames) - 1
        images.append(render(k))
        durations.append(pause_ms * (GIF_END_PAUSES if last else 1))
        if not last:
            for i in range(1, GIF_TWEEN_FRAMES):
                images.append(render(k, k + 1, i / GIF_TWEEN_FRAMES))
                durations.append(max(10, move_ms // GIF_TWEEN_FRAMES))
    images[0].save(
        path, format="GIF", save_all=True, append_images=images[1:], duration=durations, loop=0
    )


class FoldingAnimation:
    """Window that shows, move after move, the path from a conformation to the
    native state (phase ``discriminator_score``).

    The window opens on the starting conformation; ``play(path)`` gives it the
    path (a hpfold.folding_path.FoldingPath) and starts the animation. Every
    move slides the residues from one frame to the next in
    ``gui.animation_move_ms`` and then waits ``gui.animation_pause_ms``. The
    dashed lines are the H-H contacts.

    `prediction`, if given, is shown at the bottom right (the outputs of the
    decoy discriminator).

    The Pause/Play button stops the animation and resumes it from where it
    stopped. While it is stopped, the previous and next move buttons slide the
    chain one move backwards or forwards. Once the path is known, Save GIF
    writes the whole animation to a file (by default in
    ``output.animation_dir``).
    """

    # Intermediate positions drawn while the residues slide.
    TWEEN_FRAMES = 20
    # The canvas of the grid is a square at least this large and at least as
    # tall as the panel, but leaving this margin on the screen (title bar,
    # status bar, taskbar): a larger window could push the buttons out of it.
    MIN_CANVAS_SIDE = 640
    SCREEN_MARGIN = 160

    def __init__(
        self,
        master: tk.Misc,
        config: dict,
        sequence: str,
        coordinates: list[Coord],
        on_close: Callable[[], None] | None = None,
        prediction: str | None = None,
        title: str = "hpfold — folding to the native state",
    ) -> None:
        gui = config["gui"]
        self.prediction = prediction
        self.radius_ratio = gui["residue_radius"] / gui["cell_size"]
        self.colors = {"H": gui["color_h"], "P": gui["color_p"]}
        self.move_ms = gui["animation_move_ms"]
        self.pause_ms = gui["animation_pause_ms"]
        self.animation_dir = Path(config["output"]["animation_dir"])
        self.on_close = on_close

        self.sequence = sequence
        self.frames = [list(coordinates)]
        self.captions = ["Start"]
        # The path given to play(), None while it is computed.
        self.path = None
        # Frame shown, or left by the slide in progress; frame the slide goes
        # to (None when the chain is still) and fraction of the way done.
        self.frame = 0
        self.target: int | None = None
        self.tween = 0.0
        # True while the animation goes on by itself, False when it is
        # stopped (also while a single move asked by the user slides).
        self.playing = False
        self._job: str | None = None

        self.window = tk.Toplevel(master)
        self.window.title(title)
        self.window.protocol("WM_DELETE_WINDOW", self._close)
        self.status = tk.StringVar()
        self._set_bounds()
        self._build_widgets()
        self._draw()

    def _build_widgets(self) -> None:
        # Size fixed at the end, once the size of the panel is known.
        self.canvas = grid_canvas(self.window, self.MIN_CANVAS_SIDE, self.MIN_CANVAS_SIDE)
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(0, weight=1)
        self.canvas.bind("<Configure>", lambda event: self._fit(event.width, event.height))

        panel = tk.Frame(self.window)
        panel.grid(row=0, column=1, sticky="ns", padx=8, pady=8)
        if self.prediction is not None:
            prediction_box(panel).set_text(self.prediction)
        legend_box(panel, self.colors)

        # Same boxes as the panel of ConformationEditor; the box of the
        # controls that can be used now has a blue border (see _update_buttons).
        box = card(panel, "Path to the native state")
        box.pack(fill="x", pady=(0, 10))
        self.info = KeyValueList(box, background=THEME["surface"])
        self.info.pack(fill="x")
        self.info.set_text(f"Sequence: {self.sequence}")

        box = card(panel, "Playback")
        box.pack(fill="x", pady=(0, 10))
        hint(box, "Space key: pause/play.").pack(anchor="w", pady=(0, 8))
        self.play_button = FlatButton(box, primary=True, text="⏸ Pause", command=self._toggle)
        self.play_button.pack(fill="x", pady=(0, 4), ipady=3)
        self.restart_button = FlatButton(box, text="Restart", command=self._restart)
        self.restart_button.pack(fill="x")
        self.control_boxes = {"playback": box}

        box = card(panel, "Step by step")
        box.pack(fill="x", pady=(0, 10))
        hint(box, "When paused, one move at a time (←/→ keys).").pack(anchor="w", pady=(0, 8))
        self.back_button = FlatButton(
            box, text="← Previous move", command=lambda: self._go(-1)
        )
        self.back_button.pack(fill="x", pady=(0, 4))
        self.next_button = FlatButton(box, text="Next move →", command=lambda: self._go(1))
        self.next_button.pack(fill="x")
        self.control_boxes["step"] = box

        box = card(panel, "Export")
        box.pack(fill="x", pady=(0, 10))
        hint(box, "The whole path as an animated GIF.").pack(anchor="w", pady=(0, 8))
        self.gif_button = FlatButton(box, text="Save GIF…", command=self._save_gif)
        self.gif_button.pack(fill="x")
        self.window.bind("<space>", lambda event: self._toggle())
        self.window.bind("<Left>", lambda event: self._go(-1))
        self.window.bind("<Right>", lambda event: self._go(1))

        status_bar(self.window, self.status).grid(row=1, column=0, columnspan=2, sticky="ew")
        self._update_buttons()

        # The grid takes more room than the panel, as long as the screen has
        # room for it.
        self.window.update_idletasks()
        side = min(
            max(self.MIN_CANVAS_SIDE, panel.winfo_reqheight()),
            self.window.winfo_screenheight() - self.SCREEN_MARGIN,
            self.window.winfo_screenwidth() - panel.winfo_reqwidth() - self.SCREEN_MARGIN,
        )
        self.canvas.config(width=side, height=side)
        self._fit(side, side)

    # ------------------------------------------------------------------
    # Path and playback

    def set_status(self, text: str) -> None:
        self.status.set(text)

    def show_message(self, text: str) -> None:
        """Box with the text in the middle of the window, over the chain."""
        self.hide_message()
        self._message = tk.Frame(
            self.window, padx=24, pady=16, background=THEME["surface"],
            highlightthickness=1, highlightbackground=THEME["border"],
            highlightcolor=THEME["border"],
        )
        tk.Label(
            self._message, text=text, font=TITLE_FONT,
            background=THEME["surface"],
        ).pack()
        self._message.place(in_=self.canvas, relx=0.5, rely=0.5, anchor="center")

    def hide_message(self) -> None:
        if getattr(self, "_message", None) is not None:
            self._message.destroy()
            self._message = None

    def play(self, path) -> None:
        """Takes the frames of the path and starts the animation."""
        self.path = path
        self.frames = path.frames
        self.captions = [f"Start: energy {path.start_energy}, {len(path.steps)} moves to go"]
        for i, step in enumerate(path.steps, start=1):
            self.captions.append(
                f"Move {i}/{len(path.steps)}: {step.move_type} at residue {step.residue}"
                f" — energy {step.energy}, {step.distance} moves left"
            )
        if path.steps:
            self.captions[-1] += " — native state reached"
        else:
            self.captions[0] = f"The conformation is already native (energy {path.start_energy})"
        self.info.set_text(
            f"Sequence: {path.sequence}\n"
            f"Lowest energy (exact): {path.target_energy}\n"
            f"Native states: {path.n_native}\n"
            f"Moves to the native state: {len(path.steps)}"
        )
        self._set_bounds()
        self._fit(self.canvas.winfo_width(), self.canvas.winfo_height())
        self._restart()

    def _restart(self) -> None:
        self._cancel()
        self.frame, self.target, self.tween = 0, None, 0.0
        self.playing = len(self.frames) > 1
        self._show_frame()
        if self.playing:
            self._job = self.window.after(self.pause_ms, self._next_move)

    def _toggle(self) -> None:
        """Pause/Play: stops the animation or resumes it from where it is."""
        if self.playing:
            # A move halfway is not cut: it ends, and then the chain stays.
            self.playing = False
            if self.target is None:
                self._cancel()
            self._update_buttons()
        elif self.target is None and self.frame == len(self.frames) - 1:
            # At the native state Play starts again from the beginning.
            self._restart()
        elif len(self.frames) > 1:
            self.playing = True
            self._update_buttons()
            if self.target is None:
                self._next_move()

    def _go(self, delta: int) -> None:
        """Slides the chain to the previous (-1) or the next (+1) frame.

        Only while the animation is stopped and the chain is still.
        """
        if self.playing or self.target is not None:
            return
        if 0 <= self.frame + delta < len(self.frames):
            self._start_slide(self.frame + delta)

    def _next_move(self) -> None:
        self._job = None
        if self.frame < len(self.frames) - 1:
            self._start_slide(self.frame + 1)

    def _start_slide(self, target: int) -> None:
        self.target, self.tween = target, 0.0
        self._update_buttons()
        self._slide()

    def _slide(self) -> None:
        """One intermediate position of the move from the frame to the target."""
        self.tween += 1 / self.TWEEN_FRAMES
        if self.tween < 1 - 1e-9:
            self._draw()
            self._job = self.window.after(
                max(1, self.move_ms // self.TWEEN_FRAMES), self._slide
            )
            return
        self.frame, self.target, self.tween = self.target, None, 0.0
        self._job = None
        if self.frame == len(self.frames) - 1:
            self.playing = False
        self._show_frame()
        if self.playing:
            self._job = self.window.after(self.pause_ms, self._next_move)

    def _save_gif(self) -> None:
        """Asks for a file and saves the whole path to it as an animated GIF."""
        if self.path is None:
            return
        self.animation_dir.mkdir(parents=True, exist_ok=True)
        file = filedialog.asksaveasfilename(
            parent=self.window,
            title="Save the animation as GIF",
            initialdir=self.animation_dir,
            initialfile=f"folding_{self.sequence}.gif",
            defaultextension=".gif",
            filetypes=[("GIF", "*.gif")],
        )
        if not file:
            return
        # The GIF takes a few seconds and blocks the window.
        self.window.config(cursor="watch")
        self.status.set("Saving the GIF...")
        self.window.update()
        try:
            save_folding_gif(
                file, self.sequence, self.frames, self.captions, self.colors,
                self.radius_ratio, self.move_ms, self.pause_ms,
            )
        except OSError as error:
            messagebox.showerror("Save error", str(error), parent=self.window)
        else:
            messagebox.showinfo("GIF", f"Animation saved to {file}.", parent=self.window)
        finally:
            self.window.config(cursor="")
            self.status.set(self.captions[self.frame])

    def _cancel(self) -> None:
        if self._job is not None:
            self.window.after_cancel(self._job)
            self._job = None

    def _close(self) -> None:
        self._cancel()
        self.window.destroy()
        if self.on_close is not None:
            self.on_close()

    def _show_frame(self) -> None:
        self.status.set(self.captions[self.frame])
        self._update_buttons()
        self._draw()

    def _update_buttons(self) -> None:
        has_moves = len(self.frames) > 1
        # One move at a time: only while stopped and with the chain still.
        still = not self.playing and self.target is None
        self.play_button.config(text="⏸ Pause" if self.playing else "▶ Play")
        self.play_button.set_enabled(has_moves)
        self.back_button.set_enabled(still and self.frame > 0)
        self.next_button.set_enabled(still and self.frame < len(self.frames) - 1)
        self.restart_button.set_enabled(has_moves)
        self.gif_button.set_enabled(self.path is not None)
        # Blue border: playback while it plays, step by step while paused.
        active = None if not has_moves else "playback" if self.playing else "step"
        for name, box in self.control_boxes.items():
            color = THEME["button_primary"] if name == active else THEME["border"]
            box.config(highlightbackground=color, highlightcolor=color)

    # ------------------------------------------------------------------
    # Drawing

    def _set_bounds(self) -> None:
        self.xs, self.ys = lattice_bounds(self.frames)

    def _fit(self, width: int, height: int) -> None:
        self.cell_size, self.origin = fit_grid(width, height, len(self.xs), len(self.ys))
        if hasattr(self, "play_button"):
            self._draw()

    def _pixel(self, cell: tuple[float, float]) -> tuple[float, float]:
        x, y = cell
        return cell_to_pixel((x - self.xs[0], y - self.ys[0]), self.cell_size, self.origin)

    def _draw(self) -> None:
        canvas = self.canvas
        canvas.delete("all")
        draw_lattice(canvas, self.xs, self.ys, self._pixel)
        frame = self.frames[self.frame]
        r = self.radius_ratio * self.cell_size
        if self.target is not None:
            positions = interpolate(frame, self.frames[self.target], self.tween)
        else:
            positions = frame
            # The contacts are drawn only when the chain is on the lattice.
            for i, j in hh_contacts(self.sequence, frame):
                canvas.create_line(
                    *self._pixel(frame[i]), *self._pixel(frame[j]),
                    width=max(1, r / 6), dash=(4, 4), fill=THEME["bond"],
                )
        draw_chain(
            canvas,
            [self._pixel(cell) for cell in positions],
            [self.colors[c] for c in self.sequence],
            r,
        )


class StartDialog:
    """First window: asks whether the user wants to draw a new protein.

    It is built by hand instead of with tkinter.messagebox, whose layout is
    fixed: here the text and the buttons stay centred when the window is
    resized.
    """

    def __init__(
        self,
        root: tk.Tk,
        details: str,
        title: str = "Initial conformation",
        question: str = "Do you want to draw a new protein?",
    ) -> None:
        # True = draw a new protein, False = use the default one,
        # None = dialog closed without choosing.
        self.answer: bool | None = None

        self.window = tk.Toplevel(root)
        self.window.title(title)
        # Closing the window from the title bar is the same as "Cancel".
        self.window.protocol("WM_DELETE_WINDOW", self.window.destroy)

        # A widget packed with expand and without fill is kept in the middle
        # of the space it is given: the whole content moves with the window.
        content = tk.Frame(self.window, padx=24, pady=16)
        content.pack(expand=True)

        tk.Label(content, text=question, font=TITLE_FONT).pack(
            pady=(0, 12)
        )
        tk.Label(content, justify="center", text=details).pack(pady=(0, 16))

        buttons = tk.Frame(content)
        buttons.pack()
        for text, answer in (("Yes", True), ("No", False), ("Cancel", None)):
            FlatButton(
                buttons, primary=answer is True, text=text, width=8,
                command=lambda a=answer: self._choose(a),
            ).pack(side="left", padx=4)

        # The window cannot be made smaller than its content.
        self.window.update_idletasks()
        self.window.minsize(self.window.winfo_reqwidth(), self.window.winfo_reqheight())

    def _choose(self, answer: bool | None) -> None:
        self.answer = answer
        self.window.destroy()

    def ask(self) -> bool | None:
        """Shows the dialog and waits for the answer."""
        self.window.wait_window()
        return self.answer


def write_default_conformation(config: dict) -> Path:
    """Saves a default conformation for the sequence in the configuration.

    Used when the user chooses not to draw the protein: the sequence is
    ``protein.sequence`` and the shape is the one of default_coordinates().
    Returns the path of the written file.
    """
    protein = config["protein"]
    sequence = protein["sequence"]
    if len(sequence) != protein["length"]:
        raise ValueError(
            f"protein.length is {protein['length']} but protein.sequence "
            f"has {len(sequence)} residues"
        )
    side = grid_side(config)
    coordinates = default_coordinates(side, side, len(sequence))
    validate_conformation(sequence, coordinates, side, side)
    path = Path(protein["initial_conformation"])
    save_conformation(path, sequence, coordinates)
    return path


def run(config: dict) -> None:
    """Opens the grid and saves the configuration drawn by the user."""
    root = hidden_root()
    draw_new = StartDialog(
        root,
        "Yes: you draw the chain and assign the H/P types.\n"
        "No: the default protein of the configuration file is used\n"
        f"(sequence {config['protein']['sequence']}).",
    ).ask()
    if draw_new is None:
        # Dialog cancelled: nothing is written.
        root.destroy()
        return

    if not draw_new:
        path = Path(config["protein"]["initial_conformation"])
        if not path.exists() or messagebox.askyesno(
            "Overwrite?", f"The file {path} already exists. Overwrite it?"
        ):
            try:
                write_default_conformation(config)
            except (ValueError, OSError) as error:
                messagebox.showerror("Error", str(error))
            else:
                messagebox.showinfo(
                    "Initial conformation", f"Default protein saved to {path}."
                )
        root.destroy()
        return

    show_editor(root, ConformationEditor(root, config))


def hidden_root() -> tk.Tk:
    """Main window of a phase, hidden until the first question is answered."""
    root = tk.Tk()
    root.withdraw()
    apply_theme(root)
    # Tk translates the buttons of its dialogs (Yes/No/Cancel) to the system
    # language: force English, like the rest of the interface.
    root.tk.call("::msgcat::mclocale", "en")
    return root


def show_editor(root: tk.Tk, editor: ConformationEditor) -> None:
    """Shows the hidden main window with the editor and waits until it is closed."""
    root.deiconify()
    # The window opens at its natural size and can be shrunk down to half of
    # it, but never so much that the controls on the right get cut off.
    root.update_idletasks()
    root.minsize(
        root.winfo_reqwidth() // 2,
        max(root.winfo_reqheight() // 2, root.winfo_reqheight() - editor.canvas.winfo_reqheight()),
    )
    root.mainloop()
