"""Phase 3b, use — the trained decoy discriminator as a judge of one
conformation, from the command line.

A first window asks whether to draw the conformation in the GUI or to use the
one of the file ``discriminator_model.input_conformation`` (a separate TOML
file in config/). When the conformation is drawn, the editor stays open: more
conformations can be judged one after the other.

After the judgement a window animates, move after move, the shortest path
from the conformation to the nearest native state. The path does not come
from the model but from the exact oracle of the policy dataset
(hpfold.folding_path); it is printed on the terminal too.

The outputs of the model are printed on the terminal and shown at the bottom
right of the window of the animation and, when the conformation is drawn, of
the editor.

The model is the one saved by phase ``discriminator_train`` in
``output.discriminator_model_dir``.
"""

import tomllib
from pathlib import Path
from tkinter import messagebox

import numpy as np
import torch

from hpfold import discriminator_model as dm
from hpfold import folding_path as fp
from hpfold import gui
from hpfold.conformation import Coord, validate_conformation
from hpfold.policy_dataset import contact_map


def load_input(path, length: int) -> tuple[str, list[Coord]]:
    """Sequence and coordinates of the TOML file of the input conformation.

    Raises ValueError if the chain does not have `length` residues or is not a
    self-avoiding walk on the lattice.
    """
    with open(path, "rb") as f:
        data = tomllib.load(f)
    sequence = data["sequence"]
    coordinates = [(int(x), int(y)) for x, y in data["coordinates"]]
    check_input(sequence, coordinates, length)
    return sequence, coordinates


def check_input(sequence: str, coordinates: list[Coord], length: int) -> None:
    """Raises ValueError if the conformation cannot be given to the model."""
    if len(sequence) != length:
        raise ValueError(
            f"the model was trained on chains of {length} residues, "
            f"the sequence has {len(sequence)}"
        )
    if not coordinates:
        raise ValueError("the conformation is empty")
    # The coordinates can be anywhere (negative ones too): the walk is checked
    # on the smallest grid that contains it.
    x0 = min(x for x, _ in coordinates)
    y0 = min(y for _, y in coordinates)
    shifted = [(x - x0, y - y0) for x, y in coordinates]
    width = max(x for x, _ in shifted) + 1
    height = max(y for _, y in shifted) + 1
    validate_conformation(sequence, shifted, width, height)


def evaluate(model: dm.Discriminator, sequence: str, coordinates: list[Coord]) -> dict:
    """Outputs of the model for one conformation, plus its exact energy.

    Keys: ``prob`` (probability of native), ``logit``, ``energy``, ``e_min``
    and ``gap`` (auxiliary heads) and ``exact_energy`` (minus the H-H
    contacts, computed without the model).
    """
    hp = torch.tensor([[c == "H" for c in sequence]], dtype=torch.float32)
    contacts = torch.tensor(contact_map(np.asarray([coordinates])), dtype=torch.float32)
    out = dm.predict(model, hp, contacts)
    result = {name: float(values[0]) for name, values in out.items()}
    result["exact_energy"] = int(dm.hh_energy(hp, contacts)[0])
    return result


def format_result(sequence: str, result: dict) -> str:
    """Text of the outputs, shown in the window and on the terminal."""
    verdict = "native" if result["prob"] >= 0.5 else "decoy"
    return "\n".join([
        f"Sequence: {sequence}",
        f"Verdict: {verdict}",
        f"Probability of native: {result['prob']:.4f}",
        f"Logit: {result['logit']:.3f}",
        f"Energy (predicted): {result['energy']:.2f}   exact: {result['exact_energy']}",
        f"Lowest energy of the sequence (predicted): {result['e_min']:.2f}",
        f"Gap from the lowest energy (predicted): {result['gap']:.2f}",
    ])


def animate_folding(
    root,
    config: dict,
    sequence: str,
    coordinates: list[Coord],
    on_close=None,
    prediction: str | None = None,
) -> gui.FoldingAnimation:
    """Opens the window of the path to the native state and starts it.

    `prediction` is the text of the outputs of the model, shown in the window.

    The window shows the conformation, with a pop-up over it, while the path
    is computed: the first time the graph of all the conformations is built,
    which takes a few seconds.
    """
    animation = gui.FoldingAnimation(
        root, config, sequence, coordinates, on_close=on_close, prediction=prediction
    )
    message = "Computing the path to the native state..."
    animation.set_status(message)
    animation.show_message(message)
    animation.window.config(cursor="watch")

    def compute() -> None:
        if not animation.window.winfo_exists():
            return  # window closed before the computation started
        try:
            path = fp.fold_to_native(
                sequence, coordinates, config["policy_dataset"]["exact_max_length"]
            )
        except ValueError as error:
            animation.set_status(f"No path to the native state: {error}")
            print(f"No path to the native state: {error}\n")
        else:
            print(fp.format_path(path) + "\n")
            animation.play(path)
        finally:
            animation.hide_message()
            animation.window.config(cursor="")

    # The computation blocks the window until it ends: it starts only after
    # the window and the pop-up have been drawn.
    animation.window.update()
    animation.window.after(100, compute)
    return animation


def run(config: dict) -> None:
    """Asks for a conformation (GUI or file) and shows what the model says."""
    model_dir = Path(config["output"]["discriminator_model_dir"])
    if not (model_dir / "discriminator.pt").exists():
        raise SystemExit(
            f"No trained model in {model_dir}: run the phase discriminator_train first."
        )
    model = dm.load_model(model_dir)
    length = model.hyperparameters["length"]
    input_path = Path(config["discriminator_model"]["input_conformation"])

    editor = None

    def show(sequence: str, coordinates: list[Coord], on_close=None) -> None:
        # The outputs go to the terminal, to the editor (if the conformation
        # was drawn) and to the window of the animation.
        text = format_result(sequence, evaluate(model, sequence, coordinates))
        print(text + "\n")
        if editor is not None:
            editor.show_result(text)
        animate_folding(root, config, sequence, coordinates, on_close=on_close,
                        prediction=text)

    root = gui.hidden_root()
    draw_new = gui.StartDialog(
        root,
        f"Yes: you draw a chain of {length} residues and assign the H/P types.\n"
        "No: the conformation of the input file is used\n"
        f"({input_path}).",
        title="Decoy discriminator",
        question="Do you want to draw the conformation to judge?",
    ).ask()
    if draw_new is None:
        root.destroy()
        return

    if not draw_new:
        try:
            sequence, coordinates = load_input(input_path, length)
        except (OSError, KeyError, ValueError, tomllib.TOMLDecodeError) as error:
            messagebox.showerror("Error", f"{input_path}: {error}")
            root.destroy()
            return
        # The phase ends when the window of the animation is closed.
        show(sequence, coordinates, on_close=root.destroy)
        root.mainloop()
        return

    editor = gui.ConformationEditor(
        root,
        config,
        length=length,
        on_done=show,
        done_text="Evaluate",
        title="hpfold — conformation for the decoy discriminator",
    )
    gui.show_editor(root, editor)
