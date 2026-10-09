"""Entry point of the pipeline: ``python -m hpfold --config <file> <phase>``."""

import argparse

from hpfold import dataset, gui, ml, simulation
from hpfold.config import load_config

PHASES = {
    "gui": gui.run,
    "simulate": simulation.run,
    "dataset": dataset.run,
    "train": ml.run,
}


def main() -> None:
    parser = argparse.ArgumentParser(prog="hpfold", description=__doc__)
    parser.add_argument("--config", required=True, help="file di configurazione TOML")
    parser.add_argument("phase", choices=PHASES, help="fase da eseguire")
    args = parser.parse_args()

    PHASES[args.phase](load_config(args.config))


if __name__ == "__main__":
    main()
