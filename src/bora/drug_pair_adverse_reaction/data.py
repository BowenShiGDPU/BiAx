from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd


SPLIT_NAMES = ("training", "validation", "test")


class PairADRDataset:
    def __init__(self, data_root: str | Path):
        self.root = Path(data_root)
        self.feature_dir = self.root / "inputs"
        self.split_dir = self.root / "splits"
        self.pairs = pd.read_csv(self.root / "entities/pairs.tsv", sep="\t", dtype=str)
        self.endpoints = pd.read_csv(self.root / "entities/endpoints.tsv", sep="\t", dtype=str)
        self.positive = self._load_mask(self.root / "labels/positive_mask.packbits.npy")

    def _load_mask(self, path: Path) -> np.ndarray:
        packed = np.load(path, allow_pickle=False)
        return np.unpackbits(packed, axis=1, bitorder="little")[:, :len(self.endpoints)].astype(bool)

    def split_codes(self, protocol: str, seed: int):
        run_dir = self.root / f"splits/{protocol}/seed_{seed}"
        lookup = {name: index for index, name in enumerate(SPLIT_NAMES)}
        if protocol in {"random_split", "unseen_drug"}:
            table = pd.read_csv(run_dir / "pair_assignment.tsv", sep="\t", dtype=str)
            return table["split"].map(lookup).to_numpy(np.int8), None
        if protocol == "unseen_endpoint":
            table = pd.read_csv(run_dir / "endpoint_assignment.tsv", sep="\t", dtype=str)
            return None, table["split"].map(lookup).to_numpy(np.int8)
        raise ValueError(protocol)

    def iter_examples(self, protocol: str, seed: int, split: str,
                      pair_chunk_size: int = 2048) -> Iterator[pd.DataFrame]:
        split_code = SPLIT_NAMES.index(split)
        negative = self._load_mask(self.root / f"splits/{protocol}/seed_{seed}/negative_mask.packbits.npy")
        pair_codes, endpoint_codes = self.split_codes(protocol, seed)
        allowed_endpoints = np.ones(len(self.endpoints), dtype=bool) if endpoint_codes is None else endpoint_codes == split_code
        endpoint_pool = np.flatnonzero(allowed_endpoints)
        for start in range(0, len(self.pairs), pair_chunk_size):
            stop = min(start + pair_chunk_size, len(self.pairs))
            pair_indices = np.arange(start, stop, dtype=np.int32)
            if pair_codes is not None:
                pair_indices = pair_indices[pair_codes[start:stop] == split_code]
            if not len(pair_indices):
                continue
            for label, matrix in ((1, self.positive[np.ix_(pair_indices, endpoint_pool)]),
                                  (0, negative[np.ix_(pair_indices, endpoint_pool)])):
                local_pair, local_endpoint = np.nonzero(matrix)
                if len(local_pair):
                    yield pd.DataFrame({
                        "pair_index": pair_indices[local_pair],
                        "endpoint_index": endpoint_pool[local_endpoint],
                        "label": np.full(len(local_pair), label, dtype=np.int8),
                    })


def deterministic_subset(dataset: PairADRDataset, protocol: str, seed: int,
                         split: str, limit: int) -> pd.DataFrame:
    kept = None
    salt = np.uint64(seed) ^ np.uint64(SPLIT_NAMES.index(split) * 0x9E3779B1)
    for chunk in dataset.iter_examples(protocol, seed, split):
        pair = chunk["pair_index"].to_numpy(np.uint64)
        endpoint = chunk["endpoint_index"].to_numpy(np.uint64)
        label = chunk["label"].to_numpy(np.uint64)
        value = pair * np.uint64(0x9E3779B185EBCA87) + endpoint * np.uint64(0xC2B2AE3D27D4EB4F) + label + salt
        value ^= value >> np.uint64(30)
        value *= np.uint64(0xBF58476D1CE4E5B9)
        value ^= value >> np.uint64(27)
        value *= np.uint64(0x94D049BB133111EB)
        value ^= value >> np.uint64(31)
        chunk = chunk.assign(_priority=value)
        kept = chunk if kept is None else pd.concat([kept, chunk], ignore_index=True)
        if len(kept) > 2 * limit:
            index = np.argpartition(kept["_priority"].to_numpy(), limit - 1)[:limit]
            kept = kept.iloc[index].copy()
    if kept is None or len(kept) < limit:
        raise ValueError("insufficient rows for deterministic subset")
    index = np.argpartition(kept["_priority"].to_numpy(), limit - 1)[:limit]
    return kept.iloc[index].sort_values("_priority").drop(columns="_priority").reset_index(drop=True)


def load_features(data_root: str | Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    root = Path(data_root)
    drug_file = np.load(root / "inputs/drug_entities.npz", allow_pickle=False)
    endpoint_file = np.load(root / "inputs/endpoint_entities.npz", allow_pickle=False)
    drug = {key: drug_file[key] for key in drug_file.files}
    endpoint = {key: endpoint_file[key] for key in endpoint_file.files}
    mechanism = np.load(root / "inputs/pair_mechanism.npy", mmap_mode="r")
    return drug, endpoint, mechanism
