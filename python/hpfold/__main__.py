"""Entry point of the pipeline: ``python -m hpfold --config <file> <phase>``."""

import argparse

from hpfold import discriminator_dataset, gui, ml, policy_dataset, simulation, verification
from hpfold.config import load_config


def discriminator_train(config: dict) -> None:
    # Imported here so that torch (optional group `ml`) is needed only by
    # this phase, and the other phases start without loading it.
    from hpfold import discriminator_model

    discriminator_model.run(config)


# Maps each phase name accepted on the command line to the function that runs
# it. Every function takes the loaded configuration as its only argument.
# The functions are stored without being called (no parentheses): the chosen
# one is invoked in main().
PHASES = {
    "gui": gui.run,
    "simulate": simulation.run,
    "verify": verification.run,
    # Dataset for the folding policy.
    "policy_dataset": policy_dataset.run,
    # Dataset of natives and decoys for the decoy discriminator.
    "discriminator_dataset": discriminator_dataset.run,
    # Training and evaluation of the decoy discriminator (needs torch).
    "discriminator_train": discriminator_train,
    "train": ml.run,
}


def main() -> None:

    # ------------------------------------------------------------------
    # Parser creation
    # argparse is the standard module that parses arguments passed from the terminal
    parser = argparse.ArgumentParser(prog="hpfold", description=__doc__)

    # Named option, made mandatory by `required=True`: path of the TOML file
    # holding the configuration for the run.
    parser.add_argument("--config", required=True, help="configuration file TOML")

    # Positional argument: must be one of the keys of PHASES
    parser.add_argument("phase", choices=PHASES, help="phase to execute")

    # Read and validate the command line: the values end up in
    args = parser.parse_args()

    # Load the configuration from TOML file and  passes it to the function of the phase
    # Example: simulation.run(config)
    PHASES[args.phase](load_config(args.config))

    # Example of running command:
    # python3 -m hpfold --config ../config/default.toml simulate
    #
    # IN QUESTO MOMENTO POSSO ESEGUIRE LE DIVERSE FASI DEL PROGETTO
    # SOLO SEPARATAMENTE
    #
    # SARÀ NECESSARIO UNIRE ALMENO LE PRIME 3 FASI IN UNA UNICA CHIAMATA
    # ------------------------------------------------------------------

# True only when this file is the one being executed (`python -m hpfold`),
# not when it is imported: importing the module must not start the pipeline.
if __name__ == "__main__":
    main()
