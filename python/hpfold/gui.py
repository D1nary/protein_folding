"""Phase 0 — GUI (Tkinter) to draw the initial configuration by hand.

The drawn configuration is saved to the file given by
``protein.initial_conformation``.

This module contains only the window: the state of the chain, the validation
of the moves and the JSON file are handled by ``hpfold.conformation``.
"""

import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import messagebox

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
    ``discriminator_score``); `length` then replaces ``protein.length``.
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

        self.canvas = tk.Canvas(
            self.root,
            width=(self.chain.width + 1) * self.initial_cell_size,
            height=(self.chain.height + 1) * self.initial_cell_size,
            background="white",
            highlightthickness=0,
        )
        # The canvas takes all the space gained when the window is enlarged;
        # the panel on the right keeps its natural size.
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Configure>", self._on_resize)

        panel = tk.Frame(self.root)
        panel.grid(row=0, column=1, sticky="n", padx=8, pady=8)

        self.step_label = tk.Label(panel, font=("TkDefaultFont", 11, "bold"))
        self.step_label.pack(anchor="w")
        self.counter_label = tk.Label(panel)
        self.counter_label.pack(anchor="w", pady=(0, 8))

        # Step 1 controls.
        self.undo_button = tk.Button(panel, text="Undo last residue", command=self._undo)
        self.undo_button.pack(fill="x")
        self.clear_button = tk.Button(panel, text="Clear all", command=self._clear)
        self.clear_button.pack(fill="x")
        self.to_step2_button = tk.Button(
            panel, text="Go to step 2 (H/P) →", command=lambda: self._set_step(2)
        )
        self.to_step2_button.pack(fill="x", pady=(0, 12))

        # Step 2 controls; the selector doubles as the colour legend.
        tk.Label(panel, text="Active type / legend").pack(anchor="w")
        legend_names = {"H": "H — hydrophobic", "P": "P — polar"}
        self.type_buttons = []
        for residue_type in RESIDUE_TYPES:
            row = tk.Frame(panel)
            row.pack(anchor="w")
            swatch = tk.Canvas(row, width=16, height=16, highlightthickness=0)
            swatch.create_oval(2, 2, 14, 14, fill=self.colors[residue_type], outline="black")
            swatch.pack(side="left")
            button = tk.Radiobutton(
                row,
                text=legend_names[residue_type],
                variable=self.active_type,
                value=residue_type,
            )
            button.pack(side="left")
            self.type_buttons.append(button)
        row = tk.Frame(panel)
        row.pack(anchor="w")
        swatch = tk.Canvas(row, width=16, height=16, highlightthickness=0)
        swatch.create_oval(2, 2, 14, 14, fill=self.color_untyped, outline="black")
        swatch.pack(side="left")
        tk.Label(row, text="type not assigned").pack(side="left", padx=(4, 0))

        self.to_step1_button = tk.Button(
            panel, text="← Back to step 1", command=lambda: self._set_step(1)
        )
        self.to_step1_button.pack(fill="x", pady=(8, 12))

        self.save_button = tk.Button(panel, text=self.done_text, command=self._save)
        self.save_button.pack(fill="x")

        tk.Label(self.root, textvariable=self.status, anchor="w", relief="sunken").grid(
            row=1, column=0, columnspan=2, sticky="ew"
        )

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

    def _set_step(self, step: int) -> None:
        self.step = step
        if step == 1:
            self.status.set("Step 1: fix the shape of the chain.")
        else:
            self.status.set("Step 2: choose H or P and click on the residues.")
        self._refresh()

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

        chain = self.chain
        titles = {1: "Step 1: drawing the chain", 2: "Step 2: assigning H/P"}
        self.step_label.config(text=titles[self.step])
        self.counter_label.config(text=f"Residues: {len(chain.positions)}/{chain.length}")

        def state(enabled: bool) -> str:
            return "normal" if enabled else "disabled"

        in_step1 = self.step == 1
        self.undo_button.config(state=state(in_step1 and bool(chain.positions)))
        self.clear_button.config(state=state(in_step1 and bool(chain.positions)))
        self.to_step2_button.config(state=state(in_step1 and chain.is_complete))
        for button in self.type_buttons:
            button.config(state=state(not in_step1))
        self.to_step1_button.config(state=state(not in_step1))
        self.save_button.config(state=state(chain.is_fully_typed))

    def _pixel(self, cell: Coord) -> tuple[float, float]:
        return cell_to_pixel(cell, self.cell_size, self.origin)

    def _draw(self) -> None:
        canvas = self.canvas
        chain = self.chain
        canvas.delete("all")

        # Lattice: one line per row and per column, with a dot on every node.
        left, top = self._pixel((0, 0))
        right, bottom = self._pixel((chain.width - 1, chain.height - 1))
        for x in range(chain.width):
            px, _ = self._pixel((x, 0))
            canvas.create_line(px, top, px, bottom, fill="#d0d0d0")
        for y in range(chain.height):
            _, py = self._pixel((0, y))
            canvas.create_line(left, py, right, py, fill="#d0d0d0")
        for x in range(chain.width):
            for y in range(chain.height):
                px, py = self._pixel((x, y))
                canvas.create_oval(px - 2, py - 2, px + 2, py + 2, fill="#a0a0a0", outline="")

        # Residues, bonds and indices scale with the grid.
        r = self.radius_ratio * self.cell_size
        # A negative font size is in pixels instead of points.
        font = ("TkDefaultFont", -max(1, round(r)))

        # Bonds between consecutive residues, drawn before the residues so
        # that the circles cover the ends of the segments.
        points = [self._pixel(cell) for cell in chain.positions]
        if len(points) > 1:
            canvas.create_line(*points, width=max(1, r / 4), fill="black")

        for index, ((px, py), residue_type) in enumerate(zip(points, chain.types)):
            color = self.color_untyped if residue_type is None else self.colors[residue_type]
            canvas.create_oval(px - r, py - r, px + r, py + r, fill=color, outline="black")
            canvas.create_text(px, py, text=str(index), fill="white", font=font)


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

        tk.Label(content, text=question, font=("TkDefaultFont", 11, "bold")).pack(
            pady=(0, 12)
        )
        tk.Label(content, justify="center", text=details).pack(pady=(0, 16))

        buttons = tk.Frame(content)
        buttons.pack()
        for text, answer in (("Yes", True), ("No", False), ("Cancel", None)):
            tk.Button(
                buttons, text=text, width=8, command=lambda a=answer: self._choose(a)
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
