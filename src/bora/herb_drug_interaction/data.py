from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Bundle:
    ids: np.ndarray
    semantic: np.ndarray
    graph: np.ndarray
    structure: np.ndarray
    index: dict[str, int]


@dataclass
class Split:
    protocol: str
    seed: int
    example_id: np.ndarray
    left: np.ndarray
    right: np.ndarray
    label: np.ndarray
    part: np.ndarray


def _unit_rows(array: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(array, axis=1, keepdims=True)
    return np.divide(array, norm, out=np.zeros_like(array), where=norm > 0)


def load_bundle(data_root: str | Path, normalise: bool = True) -> Bundle:
    data = np.load(Path(data_root) / "inputs/entities.npz", allow_pickle=False)
    ids = data["ids"].astype(str)
    semantic = data["semantic"].astype(np.float32)
    graph = data["graph"].astype(np.float32)
    structure = data["structure"].astype(np.float32)
    if normalise:
        semantic, graph, structure = _unit_rows(semantic), _unit_rows(graph), _unit_rows(structure)
    return Bundle(ids, semantic, graph, structure, {value: i for i, value in enumerate(ids)})


def load_mechanism(data_root: str | Path) -> np.ndarray:
    return np.load(Path(data_root) / "inputs/pair_mechanism.npy", mmap_mode="r")


def load_constituents(data_root: str | Path):
    return np.load(Path(data_root) / "inputs/herb_constituents.npz", allow_pickle=False)


def load_split(data_root: str | Path, protocol: str, seed: int, bundle: Bundle) -> Split:
    path = Path(data_root) / f"splits/{protocol}/seed_{seed}.tsv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return Split(
        protocol=protocol,
        seed=seed,
        example_id=np.asarray([row["example_id"] for row in rows]),
        left=np.fromiter((bundle.index[row["herb_id"]] for row in rows), dtype=np.int64, count=len(rows)),
        right=np.fromiter((bundle.index[row["drug_id"]] for row in rows), dtype=np.int64, count=len(rows)),
        label=np.fromiter((int(row["label"]) for row in rows), dtype=np.int64, count=len(rows)),
        part=np.asarray([row["split"] for row in rows], dtype="U10"),
    )
