from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class FeatureBundle:
    ids: np.ndarray
    semantic: np.ndarray
    graph: np.ndarray
    structure: np.ndarray
    id_to_index: dict[str, int]


@dataclass
class TaskData:
    task: str
    protocol: str
    seed: int
    example_id: np.ndarray
    left: np.ndarray
    right: np.ndarray
    label: np.ndarray
    split: np.ndarray
    entity: FeatureBundle
    endpoint: None
    pair_mechanism: np.ndarray


def _feature_bundle(path: Path) -> FeatureBundle:
    data = np.load(path, allow_pickle=False)
    ids = data["ids"].astype(str)
    return FeatureBundle(ids, data["semantic"].astype(np.float32),
                         data["graph"].astype(np.float32),
                         data["structure"].astype(np.float32),
                         {value: i for i, value in enumerate(ids)})


def _indices(mapping: dict[str, int], values: np.ndarray) -> np.ndarray:
    return np.fromiter((mapping[str(value)] for value in values), dtype=np.int64, count=len(values))


def load_task(data_root: str | Path, protocol: str, seed: int) -> TaskData:
    root = Path(data_root)
    entity = _feature_bundle(root / "inputs/entities.npz")
    table = pd.read_csv(root / f"splits/{protocol}/seed_{seed}.tsv", sep="\t")
    mechanism = np.load(root / "inputs/pair_mechanism.npy", mmap_mode="r")
    return TaskData(
        task="ddi_binary",
        protocol=protocol,
        seed=seed,
        example_id=table["example_id"].to_numpy(),
        left=_indices(entity.id_to_index, table["drug_a_id"].astype(str).to_numpy()),
        right=_indices(entity.id_to_index, table["drug_b_id"].astype(str).to_numpy()),
        label=table["label"].to_numpy(np.int64),
        split=table["split"].to_numpy(dtype="U10"),
        entity=entity,
        endpoint=None,
        pair_mechanism=mechanism,
    )
