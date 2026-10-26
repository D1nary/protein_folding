"""Phase 3b — decoy discriminator: a model that tells native conformations from
decoys, trained on the dataset of phase ``discriminator_dataset``.

The label of an example is ``energy == e_min``: the task is made of two parts,
the energy of the conformation (local: minus the number of H-H contacts) and
the lowest energy of the sequence (global: it depends on the sequence only).
The model (``Discriminator``) has one branch for each:

- conformation branch: a graph neural network on the residues, with the bonds
  and the lattice contacts as edges, pooled into ``z_conf``; the head
  ``energy`` predicts the energy of the conformation from it;
- sequence branch: a 1D convolutional network on the HP string only, pooled
  into ``z_seq``; the head ``e_min`` predicts the lowest energy of the sequence;
- classifier: an MLP on ``z_conf``, ``z_seq`` and their product, which predicts
  the label (as a logit) and the gap.

The loss is the BCE of the label plus the MSE of the three auxiliary heads,
with the weights of ``discriminator_model.loss_weights``.

Every model of this file, the baselines included, receives the same input: the
HP sequence (``hp``, 1 for H) and the contact map of the conformation. Both
come from ``discriminator_dataset.model_inputs``, never from the other columns
of the dataset (tier, gap, e_min, provenance), which give the answer away.

The baselines measure where the difficulty is (notes/discriminator_model.md):
``B0a`` logistic regression on the energy, ``B0b`` the same plus features of
the sequence that bound the lowest energy, ``B1b`` an MLP on the flattened
H-H contact map.
"""

import csv
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from hpfold import discriminator_dataset as dd
from hpfold.policy_dataset import contact_map

# Targets of the heads of the Discriminator; "label" is the main one.
TARGETS = ("label", "energy", "e_min", "gap")
# Name of the main model in the files and in the report.
MAIN_MODEL = "GNN"
BASELINES = ("B0a", "B0b", "B1b")
# Features of a residue, computed from the sequence: H, parity of the index,
# position along the chain, chain end.
N_NODE_FEATURES = 4


# ----------------------------------------------------------------------
# Inputs


def load_examples(directory, split: str, train_sequences: int = 0, seed: int = 0) -> dict:
    """Examples of one split as tensors, ready for the models.

    ``hp`` (S, N) and ``contacts`` (S, N, N) are the input; ``label``,
    ``energy``, ``e_min``, ``gap`` the targets; ``info`` holds numpy arrays
    for the analysis of the errors (seq_id, key, tier, ...), never an input.
    With ``train_sequences`` > 0 only that many sequences, chosen at random,
    are kept (for learning curves).
    """
    data = dd.load_split(directory, split)
    if train_sequences > 0:
        ids = np.unique(data["seq_id"])
        if train_sequences < len(ids):
            chosen = np.random.default_rng(seed).choice(ids, train_sequences, replace=False)
            keep = np.isin(data["seq_id"], chosen)
            data = {name: values[keep] for name, values in data.items()}
    inputs = dd.model_inputs(data)
    examples = {
        "hp": torch.tensor(inputs["hp"], dtype=torch.float32),
        "contacts": torch.tensor(contact_map(inputs["coordinates"]), dtype=torch.float32),
    }
    for name in TARGETS:
        examples[name] = torch.tensor(data[name], dtype=torch.float32)
    examples["info"] = {name: data[name] for name in
                        ("seq_id", "key", "sequence", "tier", "kinetic_trap", "native_overlap")}
    return examples


def node_features(hp: torch.Tensor) -> torch.Tensor:
    """Features of the residues (B, N, 4): H, parity, position, chain end.

    On the square lattice two residues can touch only if their indices have
    different parity, so the parity is what bounds the number of contacts.
    """
    b, n = hp.shape
    i = torch.arange(n, device=hp.device)
    fixed = torch.stack([(i % 2).float(), i.float() / (n - 1), ((i == 0) | (i == n - 1)).float()],
                        dim=-1)
    return torch.cat([hp.unsqueeze(-1), fixed.expand(b, n, 3)], dim=-1)


def reverse_chain(hp: torch.Tensor, contacts: torch.Tensor):
    """The same chain read from the other end: residue i becomes N-1-i.

    It is the same physical state, so energy, lowest energy and label do not
    change: a symmetry used for augmentation.
    """
    return hp.flip(1), contacts.flip(1, 2)


def hh_energy(hp: torch.Tensor, contacts: torch.Tensor) -> torch.Tensor:
    """HP energy (B,) from the inputs: minus the number of H-H contacts."""
    return -(contacts * hp.unsqueeze(2) * hp.unsqueeze(1)).sum(dim=(1, 2)) / 2


def sequence_features(hp: torch.Tensor) -> torch.Tensor:
    """Features of the sequence alone (B, 5) that bound |E_min|: H, H at even
    and at odd positions, the smaller of the two, H at the chain ends."""
    even, odd = hp[:, 0::2].sum(1), hp[:, 1::2].sum(1)
    return torch.stack([hp.sum(1), even, odd, torch.minimum(even, odd), hp[:, 0] + hp[:, -1]],
                       dim=1)


# ----------------------------------------------------------------------
# Discriminator


def mlp(n_in: int, hidden: int, n_out: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(dropout),
                         nn.Linear(hidden, n_out))


def pool(h: torch.Tensor) -> torch.Tensor:
    """One vector per example from the vectors of the residues: sum and max.

    The chains have a fixed length, so the mean would only be the sum divided
    by N: the max adds what the sum cannot see (the strongest residue).
    """
    return torch.cat([h.sum(1), h.max(1).values], dim=-1)


class GraphLayer(nn.Module):
    """One message-passing layer.

    Each residue receives a message from each neighbour, a linear function of
    the neighbour's vector with different weights for bonds and contacts; the
    messages are summed and an MLP combines the sum with the residue's own
    vector (residual connection and normalization).
    """

    def __init__(self, hidden: int):
        super().__init__()
        self.bond = nn.Linear(hidden, hidden)
        self.contact = nn.Linear(hidden, hidden)
        self.update = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                                    nn.Linear(hidden, hidden))
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, bonds, contacts):
        # (N, N) or (B, N, N) @ (B, N, hidden): sum of the messages of the neighbours.
        messages = bonds @ self.bond(h) + contacts @ self.contact(h)
        return self.norm(h + self.update(torch.cat([h, messages], dim=-1)))


class ConformationGNN(nn.Module):
    """Conformation branch: graph of the residues -> z_conf (B, 2 * hidden)."""

    def __init__(self, hidden: int, layers: int):
        super().__init__()
        self.embed = nn.Linear(N_NODE_FEATURES, hidden)
        self.layers = nn.ModuleList(GraphLayer(hidden) for _ in range(layers))

    def forward(self, x, bonds, contacts):
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h, bonds, contacts)
        return pool(h)


class SequenceCNN(nn.Module):
    """Sequence branch: HP string only -> z_seq (B, 2 * hidden).

    With `layers` convolutions of width `kernel` a residue sees
    layers * (kernel - 1) + 1 residues: with the defaults (4, 5) the whole
    chain of 16.
    """

    def __init__(self, hidden: int, layers: int, kernel: int):
        super().__init__()
        sizes = [N_NODE_FEATURES] + [hidden] * layers
        self.convs = nn.ModuleList(nn.Conv1d(a, b, kernel, padding=kernel // 2)
                                   for a, b in zip(sizes, sizes[1:]))

    def forward(self, x):
        h = x.transpose(1, 2)  # Conv1d wants (B, channels, N)
        for conv in self.convs:
            h = F.relu(conv(h))
        return pool(h.transpose(1, 2))


class Discriminator(nn.Module):
    """The main model: two branches, three heads (see the module docstring)."""

    def __init__(self, length: int, hidden: int = 64, gnn_layers: int = 4, cnn_layers: int = 4,
                 cnn_kernel: int = 5, dropout: float = 0.1):
        super().__init__()
        self.hyperparameters = dict(length=length, hidden=hidden, gnn_layers=gnn_layers,
                                    cnn_layers=cnn_layers, cnn_kernel=cnn_kernel, dropout=dropout)
        i = torch.arange(length)
        self.register_buffer("bonds", ((i[:, None] - i[None, :]).abs() == 1).float())
        self.conformation = ConformationGNN(hidden, gnn_layers)
        self.sequence = SequenceCNN(hidden, cnn_layers, cnn_kernel)
        z = 2 * hidden
        self.energy_head = mlp(z, hidden, 1, dropout)
        self.e_min_head = mlp(z, hidden, 1, dropout)
        self.classifier = mlp(3 * z, hidden, 2, dropout)  # logit, gap

    def forward(self, hp, contacts) -> dict:
        x = node_features(hp)
        z_conf = self.conformation(x, self.bonds, contacts)
        z_seq = self.sequence(x)
        out = self.classifier(torch.cat([z_conf, z_seq, z_conf * z_seq], dim=-1))
        return {
            "logit": out[:, 0],
            "gap": out[:, 1],
            "energy": self.energy_head(z_conf)[:, 0],
            "e_min": self.e_min_head(z_seq)[:, 0],
        }


# ----------------------------------------------------------------------
# Baselines


class Logistic(nn.Module):
    """B0a / B0b: logistic regression on the energy, alone (B0a) or with the
    features of the sequence (B0b). The features are standardized with the
    mean and spread of the training set (`fit_scaler`)."""

    def __init__(self, with_sequence: bool):
        super().__init__()
        self.with_sequence = with_sequence
        n = 1 + (5 if with_sequence else 0)
        self.linear = nn.Linear(n, 1)
        self.register_buffer("mean", torch.zeros(n))
        self.register_buffer("std", torch.ones(n))

    def features(self, hp, contacts):
        e = hh_energy(hp, contacts).unsqueeze(1)
        return torch.cat([e, sequence_features(hp)], dim=1) if self.with_sequence else e

    def fit_scaler(self, hp, contacts):
        f = self.features(hp, contacts)
        self.mean.copy_(f.mean(0))
        self.std.copy_(f.std(0).clamp_min(1e-6))

    def forward(self, hp, contacts):
        return {"logit": self.linear((self.features(hp, contacts) - self.mean) / self.std)[:, 0]}


class PairMLP(nn.Module):
    """B1b: MLP on the HP vector and on the H-H contacts of the pairs that can
    touch (j - i odd and at least 3): the same information as the GNN, with no
    graph structure and no weights shared between pairs."""

    def __init__(self, length: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        i, j = np.triu_indices(length, k=3)
        odd = (j - i) % 2 == 1
        self.register_buffer("i", torch.tensor(i[odd]))
        self.register_buffer("j", torch.tensor(j[odd]))
        self.net = nn.Sequential(nn.Linear(length + len(self.i), hidden), nn.ReLU(),
                                 nn.Dropout(dropout), mlp(hidden, hidden, 1, dropout))

    def forward(self, hp, contacts):
        hh = contacts[:, self.i, self.j] * hp[:, self.i] * hp[:, self.j]
        return {"logit": self.net(torch.cat([hp, hh], dim=1))[:, 0]}


def make_model(name: str, length: int, cfg: dict) -> nn.Module:
    if name == MAIN_MODEL:
        return Discriminator(length, cfg["hidden"], cfg["gnn_layers"], cfg["cnn_layers"],
                             cfg["cnn_kernel"], cfg["dropout"])
    if name in ("B0a", "B0b"):
        return Logistic(with_sequence=name == "B0b")
    if name == "B1b":
        return PairMLP(length, cfg["hidden"], cfg["dropout"])
    raise ValueError(f"unknown model {name!r}: expected {MAIN_MODEL} or one of {BASELINES}")


# ----------------------------------------------------------------------
# Training


def loss_terms(out: dict, batch: dict, weights: dict) -> dict:
    """BCE of the label, plus the weighted MSE of every auxiliary head the
    model has (only the Discriminator has them)."""
    terms = {"label": F.binary_cross_entropy_with_logits(out["logit"], batch["label"])}
    for name, weight in weights.items():
        if name in out and weight > 0:
            terms[name] = weight * F.mse_loss(out[name], batch[name])
    return terms


@torch.no_grad()
def predict(model: nn.Module, hp, contacts, batch_size: int = 4096) -> dict:
    """Outputs of a model as numpy arrays; `prob` is the probability of native."""
    model.eval()
    parts = [model(hp[s:s + batch_size], contacts[s:s + batch_size])
             for s in range(0, len(hp), batch_size)]
    out = {name: torch.cat([p[name] for p in parts]) for name in parts[0]}
    out["prob"] = torch.sigmoid(out["logit"])
    return {name: values.numpy() for name, values in out.items()}


def fit(model: nn.Module, train: dict, val: dict, cfg: dict, name: str, verbose: bool = True):
    """Trains `model` with Adam, keeping the weights of the epoch with the best
    validation AUROC; stops after `patience` epochs without improvement.
    Returns the history, one dict per epoch."""
    generator = torch.Generator().manual_seed(cfg["seed"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"],
                                 weight_decay=cfg["weight_decay"])
    weights = cfg["loss_weights"]
    n = len(train["label"])
    best, best_state, waited, history = -1.0, None, 0, []
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        start, totals = time.perf_counter(), {}
        for idx in torch.randperm(n, generator=generator).split(cfg["batch_size"]):
            hp, contacts = train["hp"][idx], train["contacts"][idx]
            if cfg["augment_reverse"]:
                # Half of the batch is read from the other end of the chain.
                flip = torch.rand(len(idx), generator=generator) < 0.5
                hp_r, contacts_r = reverse_chain(hp[flip], contacts[flip])
                hp, contacts = hp.clone(), contacts.clone()
                hp[flip], contacts[flip] = hp_r, contacts_r
            batch = {t: train[t][idx] for t in TARGETS}
            terms = loss_terms(model(hp, contacts), batch, weights)
            loss = sum(terms.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for t, v in terms.items():
                totals[t] = totals.get(t, 0.0) + v.item() * len(idx) / n
        scores = evaluate(predict(model, val["hp"], val["contacts"]), val)
        row = {"model": name, "epoch": epoch, "seconds": round(time.perf_counter() - start, 2),
               **{f"train_loss_{t}": round(v, 5) for t, v in totals.items()},
               "val_auroc": round(scores["auroc"], 5), "val_accuracy": round(scores["accuracy"], 5)}
        history.append(row)
        if verbose:
            print(f"  {name} epoch {epoch:3d}  loss {sum(totals.values()):.4f}  "
                  f"val AUROC {scores['auroc']:.4f}  acc {scores['accuracy']:.4f}  "
                  f"({row['seconds']:.1f} s)")
        if scores["auroc"] > best:
            best, waited = scores["auroc"], 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    return history


def fit_logistic(model: Logistic, train: dict, val: dict, cfg: dict, name: str,
                 verbose: bool = True):
    """Trains a logistic regression: the problem is convex, so full-batch
    L-BFGS reaches the optimum, with no epochs and no early stopping. Same
    signature and history as `fit`."""
    torch.manual_seed(cfg["seed"])
    optimizer = torch.optim.LBFGS(model.parameters(), max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(
            model(train["hp"], train["contacts"])["logit"], train["label"])
        loss.backward()
        return loss

    start = time.perf_counter()
    model.train()
    loss = optimizer.step(closure)
    scores = evaluate(predict(model, val["hp"], val["contacts"]), val)
    row = {"model": name, "epoch": 1, "seconds": round(time.perf_counter() - start, 2),
           "train_loss_label": round(loss.item(), 5), "val_auroc": round(scores["auroc"], 5),
           "val_accuracy": round(scores["accuracy"], 5)}
    if verbose:
        print(f"  {name} L-BFGS  loss {loss.item():.4f}  val AUROC {scores['auroc']:.4f}  "
              f"acc {scores['accuracy']:.4f}")
    return [row]


# ----------------------------------------------------------------------
# Evaluation


def auroc(score, label) -> float:
    """Area under the ROC curve: the probability that a random native scores
    higher than a random decoy (ties count one half). NaN with one class."""
    score, label = np.asarray(score, dtype=float), np.asarray(label).astype(bool)
    n1 = label.sum()
    n0 = len(label) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    _, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
    # Rank (from 1) of each distinct score, the average of its ties.
    ranks = (np.cumsum(counts) - (counts - 1) / 2)[inverse]
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def evaluate(out: dict, examples: dict) -> dict:
    """Metrics of the outputs of a model on a set of examples.

    - accuracy at threshold 0.5 and AUROC;
    - sequence_auroc: AUROC within each sequence, averaged (are the natives
      ranked above the decoys of the same sequence?);
    - accuracy by tier (native, near, mid, far), on kinetic traps and by
      lowest energy of the sequence;
    - mean absolute error of the auxiliary heads, when the model has them.
    """
    label = examples["label"].numpy().astype(int)
    prob = out["prob"]
    correct = (prob >= 0.5).astype(int) == label
    info = examples["info"]
    metrics = {"examples": int(len(label)), "accuracy": float(correct.mean()),
               "auroc": auroc(prob, label)}
    seq_ids = info["seq_id"]
    per_seq = [auroc(prob[seq_ids == s], label[seq_ids == s]) for s in np.unique(seq_ids)]
    metrics["sequence_auroc"] = float(np.nanmean(per_seq))
    metrics["accuracy_by_tier"] = {t: float(correct[info["tier"] == t].mean())
                                   for t in ("native",) + dd.TIERS if (info["tier"] == t).any()}
    traps = info["kinetic_trap"] == 1
    if traps.any():
        metrics["accuracy_kinetic_traps"] = float(correct[traps].mean())
    e_min = examples["e_min"].numpy().astype(int)
    metrics["accuracy_by_e_min"] = {str(e): float(correct[e_min == e].mean())
                                    for e in sorted(np.unique(e_min))}
    for name in ("energy", "e_min", "gap"):
        if name in out:
            metrics[f"mae_{name}"] = float(np.abs(out[name] - examples[name].numpy()).mean())
    return metrics


# ----------------------------------------------------------------------
# Files


def save_model(model: Discriminator, path: Path) -> None:
    torch.save({"hyperparameters": model.hyperparameters, "state_dict": model.state_dict()}, path)


def load_model(directory) -> Discriminator:
    """The trained Discriminator saved by `run`, ready for `predict`."""
    saved = torch.load(Path(directory) / "discriminator.pt", weights_only=True)
    model = Discriminator(**saved["hyperparameters"])
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model


def score(model: Discriminator, sequences, coordinates) -> np.ndarray:
    """Probability of native of conformations given as HP strings and lattice
    coordinates (S, N, 2): the use of the discriminator as a judge."""
    hp = torch.tensor([[c == "H" for c in s] for s in sequences], dtype=torch.float32)
    contacts = torch.tensor(contact_map(np.asarray(coordinates)), dtype=torch.float32)
    return predict(model, hp, contacts)["prob"]


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def report(metrics: dict, split: str = "test") -> str:
    """Table comparing the models on one split."""
    columns = ("accuracy", "auroc", "sequence_auroc")
    tiers = ("native",) + dd.TIERS
    lines = [f"Results on {split} ({next(iter(metrics.values()))[split]['examples']} examples)",
             f"{'model':6s} " + " ".join(f"{c:>14s}" for c in columns)
             + " " + " ".join(f"{t:>7s}" for t in tiers) + "    traps"]
    for name, m in metrics.items():
        s = m[split]
        by_tier = s["accuracy_by_tier"]
        lines.append(f"{name:6s} " + " ".join(f"{s[c]:14.4f}" for c in columns) + " "
                     + " ".join(f"{by_tier.get(t, float('nan')):7.3f}" for t in tiers)
                     + f"  {s.get('accuracy_kinetic_traps', float('nan')):7.3f}")
    lines.append("(native ... far, traps: accuracy within the group)")
    main = metrics.get(MAIN_MODEL, {}).get(split)
    if main:
        lines.append(f"{MAIN_MODEL} auxiliary heads, mean absolute error: "
                     + ", ".join(f"{h} {main[f'mae_{h}']:.3f}" for h in ("energy", "e_min", "gap")))
        lines.append(f"{MAIN_MODEL} accuracy by lowest energy of the sequence: "
                     + ", ".join(f"{e}: {a:.3f}" for e, a in main["accuracy_by_e_min"].items()))
    return "\n".join(lines)


def run(config: dict, verbose: bool = True) -> Path:
    """Trains the Discriminator and the baselines, evaluates them on val and
    test and writes the files in `output.discriminator_model_dir`."""
    cfg = config["discriminator_model"]
    # After some epochs the GNN produces denormal floats (smaller than about
    # 1e-38), which the CPU handles up to 10 times more slowly: they are
    # treated as 0, which makes no difference to the results.
    torch.set_flush_denormal(True)
    if cfg.get("threads", 0) > 0:
        torch.set_num_threads(cfg["threads"])
    directory = config["output"]["discriminator_dataset_dir"]
    length = dd.load_meta(directory)["length"]
    train = load_examples(directory, "train", cfg["train_sequences"], cfg["seed"])
    val = load_examples(directory, "val")
    test = load_examples(directory, "test")
    if verbose:
        print(f"Examples: train {len(train['label'])} ({len(np.unique(train['info']['seq_id']))} "
              f"sequences), val {len(val['label'])}, test {len(test['label'])}")

    out = Path(config["output"]["discriminator_model_dir"])
    out.mkdir(parents=True, exist_ok=True)
    metrics, history, predictions = {}, [], {}
    for name in [MAIN_MODEL] + list(cfg["baselines"]):
        torch.manual_seed(cfg["seed"])
        model = make_model(name, length, cfg)
        if isinstance(model, Logistic):
            model.fit_scaler(train["hp"], train["contacts"])
        if verbose:
            n_params = sum(p.numel() for p in model.parameters())
            print(f"Training {name} ({n_params} parameters)")
        train_model = fit_logistic if isinstance(model, Logistic) else fit
        history += train_model(model, train, val, cfg, name, verbose)
        metrics[name] = {}
        for split, examples in (("val", val), ("test", test)):
            outputs = predict(model, examples["hp"], examples["contacts"])
            metrics[name][split] = evaluate(outputs, examples)
            if split == "test":
                predictions[name] = outputs
        if name == MAIN_MODEL:
            save_model(model, out / "discriminator.pt")

    with open(out / "history.csv", "w", newline="") as f:
        fields = sorted({k for row in history for k in row},
                        key=lambda k: (k not in ("model", "epoch"), k))
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)

    with open(out / "test_predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        info = test["info"]
        heads = [h for h in ("energy", "e_min", "gap") if h in predictions[MAIN_MODEL]]
        writer.writerow(["seq_id", "key", "sequence", "tier", "kinetic_trap", "label", "energy",
                         "e_min", "gap"] + [f"prob_{m}" for m in predictions]
                        + [f"{MAIN_MODEL}_{h}" for h in heads])
        for k in range(len(test["label"])):
            writer.writerow([int(info["seq_id"][k]), int(info["key"][k]), info["sequence"][k],
                             info["tier"][k], int(info["kinetic_trap"][k])]
                            + [int(test[t][k]) for t in TARGETS]
                            + [f"{p['prob'][k]:.5f}" for p in predictions.values()]
                            + [f"{predictions[MAIN_MODEL][h][k]:.4f}" for h in heads])

    text = report(metrics, "test") + "\n\n" + report(metrics, "val")
    (out / "report.txt").write_text(text + "\n")
    with open(out / "metrics.json", "w") as f:
        json.dump({"metrics": metrics, "config": cfg, "dataset": str(directory),
                   "git_commit": _git_commit()}, f, indent=2)
    if verbose:
        print()
        print(text)
        print(f"\nSaved to: {out}")
    return out
