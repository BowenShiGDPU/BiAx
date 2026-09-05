from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
import pandas as pd


STANDARD_SEEDS = [104729, 130363, 155921, 181081, 205019, 230047, 255019, 280031, 305021, 330019]
FORMULA_SPLIT_SEEDS = [158558373, 223138648, 669534447, 671725008, 899932873,
                       1175469643, 1339379025, 1474285194, 1731586375, 1750583325]
PROTOCOL_SEEDS = {
    "random_split": STANDARD_SEEDS,
    "unseen_endpoint": STANDARD_SEEDS,
    "unseen_formula": FORMULA_SPLIT_SEEDS,
    "external_validation": STANDARD_SEEDS,
    "source_crossfit": STANDARD_SEEDS,
}


@dataclass(frozen=True)
class Cohort:
    formula_ids: np.ndarray
    material_index: np.ndarray
    material_dose: np.ndarray
    material_mask: np.ndarray
    material_feature: np.ndarray
    presence: np.ndarray
    dose: np.ndarray
    functional_group: np.ndarray
    target_measured: np.ndarray
    target_association: np.ndarray
    mechanism_qwen: np.ndarray
    pair_mech: np.ndarray

    @property
    def n_formula(self) -> int:
        return len(self.formula_ids)


@dataclass(frozen=True)
class Endpoints:
    adr_ids: np.ndarray
    feature: np.ndarray
    target: np.ndarray
    hierarchy: np.ndarray


@dataclass(frozen=True)
class Bench:
    main: Cohort
    external: Cohort
    endpoints: Endpoints
    functional_group_feature: np.ndarray
    target_feature: np.ndarray
    main_pairs: pd.DataFrame
    external_pairs: pd.DataFrame
    main_y: np.ndarray
    external_y: np.ndarray
    main_pair_formula: np.ndarray
    main_pair_adr: np.ndarray
    external_pair_formula: np.ndarray
    external_pair_adr: np.ndarray


@dataclass(frozen=True)
class Context:
    protocol: str
    seed: int
    outer_fold: int
    train: np.ndarray
    inner: list[tuple[np.ndarray, np.ndarray]]
    test: np.ndarray
    test_is_external: bool = False


def _pad_to(array: np.ndarray, slots: int, fill: float | int | bool) -> np.ndarray:
    if array.shape[1] == slots:
        return array
    pad = np.full((array.shape[0], slots - array.shape[1]), fill, dtype=array.dtype)
    return np.concatenate([array, pad], axis=1)


def _root(data_root: str | Path | None) -> Path:
    value = data_root if data_root is not None else os.environ.get("BORA_FORMULA_DATA")
    if value is None:
        raise ValueError("provide data_root or set BORA_FORMULA_DATA")
    return Path(value)


def load_bench(data_root: str | Path | None = None) -> Bench:
    root = _root(data_root)
    main_z = np.load(root / "inputs/internal_features.npz", allow_pickle=False)
    ext_z = np.load(root / "inputs/external_features.npz", allow_pickle=False)
    slots = max(main_z["formula_material_index"].shape[1], ext_z["formula_material_index"].shape[1])
    main_material = np.concatenate(
        [main_z["material_qwen"], main_z["material_relation_aligned"]], axis=1
    ).astype(np.float32)
    external_material = np.concatenate(
        [ext_z["material_qwen"], ext_z["material_relation_aligned"]], axis=1
    ).astype(np.float32)
    main_registry = pd.read_csv(root / "entities/internal_formulas.csv")
    external_registry = pd.read_csv(root / "entities/external_formulas.csv")
    endpoint_registry = pd.read_csv(root / "entities/endpoints.csv")
    mechanism_keys = (
        "measured_target_dot", "measured_target_cosine",
        "association_target_dot", "association_target_cosine",
        "combined_target_dot", "combined_target_cosine",
    )

    def cohort(bundle, material, formula_ids) -> Cohort:
        pair_mechanism = np.stack([bundle[key].astype(np.float32) for key in mechanism_keys], axis=-1)
        return Cohort(
            formula_ids=np.asarray(formula_ids),
            material_index=_pad_to(bundle["formula_material_index"].astype(np.int64), slots, 0),
            material_dose=_pad_to(bundle["formula_material_dose"].astype(np.float32), slots, 0.0),
            material_mask=_pad_to(bundle["formula_material_mask"].astype(bool), slots, False),
            material_feature=material,
            presence=bundle["formula_presence"].astype(np.float32),
            dose=bundle["formula_dose"].astype(np.float32),
            functional_group=bundle["formula_functional_group"].astype(np.float32),
            target_measured=bundle["formula_target_measured"].astype(np.float32),
            target_association=bundle["formula_target_association"].astype(np.float32),
            mechanism_qwen=bundle["formula_mechanism_qwen"].astype(np.float32),
            pair_mech=pair_mechanism,
        )

    main = cohort(main_z, main_material, main_registry["formula_id"].to_numpy())
    external = cohort(ext_z, external_material, external_registry["formula_id"].to_numpy())
    endpoint_ids = endpoint_registry["endpoint_id"].to_numpy()
    endpoints = Endpoints(
        adr_ids=endpoint_ids,
        feature=np.concatenate(
            [main_z["adr_qwen"], main_z["adr_relation_aligned"], main_z["adr_mechanism_qwen"]],
            axis=1,
        ).astype(np.float32),
        target=main_z["adr_target"].astype(np.float32),
        hierarchy=np.load(root / "inputs/endpoint_hierarchy.npy", allow_pickle=False),
    )
    main_pairs = pd.read_csv(root / "labels/internal_pairs.csv")
    external_pairs = pd.read_csv(root / "labels/external_pairs.csv")
    formula_row = {value: i for i, value in enumerate(main.formula_ids)}
    endpoint_row = {value: i for i, value in enumerate(endpoint_ids)}
    external_formula_row = {value: i for i, value in enumerate(external.formula_ids)}
    return Bench(
        main=main,
        external=external,
        endpoints=endpoints,
        functional_group_feature=main_z["functional_group_qwen"].astype(np.float32),
        target_feature=main_z["target_qwen"].astype(np.float32),
        main_pairs=main_pairs,
        external_pairs=external_pairs,
        main_y=main_pairs["label"].to_numpy(np.int8),
        external_y=external_pairs["label"].to_numpy(np.int8),
        main_pair_formula=main_pairs["formula_id"].map(formula_row).to_numpy(np.int64),
        main_pair_adr=main_pairs["endpoint_id"].map(endpoint_row).to_numpy(np.int64),
        external_pair_formula=external_pairs["formula_id"].map(external_formula_row).to_numpy(np.int64),
        external_pair_adr=external_pairs["endpoint_id"].map(endpoint_row).to_numpy(np.int64),
    )


def build_contexts(bench: Bench, protocol: str, seed: int,
                   data_root: str | Path | None = None) -> list[Context]:
    root = _root(data_root)
    pair_index = {pair_id: i for i, pair_id in enumerate(bench.main_pairs["pair_id"])}
    if protocol == "random_split":
        table = pd.read_csv(root / "splits/random_split/assignments.csv")
        table = table[table["seed"] == seed]
        rows = table["pair_id"].map(pair_index).to_numpy()
        split = table["split"].str.lower().to_numpy()
        train, validation, test = (rows[split == name] for name in ("train", "validation", "test"))
        return [Context(protocol, seed, 0, train, [(train, validation)], test)]
    if protocol == "unseen_endpoint":
        roles = pd.read_csv(root / "splits/unseen_endpoint/pair_roles.csv")
        inner = pd.read_csv(root / "splits/unseen_endpoint/inner_endpoint_assignments.csv")
        roles, inner = roles[roles["seed"] == seed], inner[inner["seed"] == seed]
        endpoint_row = {value: i for i, value in enumerate(bench.endpoints.adr_ids)}
        contexts = []
        for fold in sorted(roles["outer_fold"].unique()):
            subset = roles[roles["outer_fold"] == fold]
            rows = subset["pair_id"].map(pair_index).to_numpy()
            role = subset["role"].str.lower().to_numpy()
            train = rows[role == "outer_train_pool"]
            test = rows[role == "outer_test"]
            inner_map = inner[inner["outer_fold"] == fold]
            inner_splits = []
            for inner_fold in sorted(inner_map["inner_fold"].unique()):
                held = {endpoint_row[value] for value in inner_map[inner_map["inner_fold"] == inner_fold]["endpoint_id"]}
                mask = np.isin(bench.main_pair_adr[train], list(held))
                inner_splits.append((train[~mask], train[mask]))
            contexts.append(Context(protocol, seed, int(fold), train, inner_splits, test))
        return contexts
    if protocol == "unseen_formula":
        outer = pd.read_csv(root / "splits/unseen_formula/outer_formula_assignments.csv")
        inner = pd.read_csv(root / "splits/unseen_formula/inner_formula_assignments.csv")
        outer, inner = outer[outer["seed"] == seed], inner[inner["seed"] == seed]
        formula_row = {value: i for i, value in enumerate(bench.main.formula_ids)}
        all_rows = np.arange(len(bench.main_pairs))
        contexts = []
        for fold in sorted(outer["outer_fold"].unique()):
            held = {formula_row[value] for value in outer[outer["outer_fold"] == fold]["formula_id"]}
            test_mask = np.isin(bench.main_pair_formula, list(held))
            train, test = all_rows[~test_mask], all_rows[test_mask]
            inner_map = inner[inner["outer_fold"] == fold]
            inner_splits = []
            for inner_fold in sorted(inner_map["inner_fold"].unique()):
                inner_held = {formula_row[value] for value in inner_map[inner_map["inner_fold"] == inner_fold]["formula_id"]}
                mask = np.isin(bench.main_pair_formula[train], list(inner_held))
                if mask.any() and (~mask).any():
                    inner_splits.append((train[~mask], train[mask]))
            contexts.append(Context(protocol, seed, int(fold), train, inner_splits, test))
        return contexts
    if protocol == "source_crossfit":
        roles = pd.read_csv(root / "splits/external_validation/source_crossfit_pair_roles.csv")
        roles = roles[roles["seed"] == seed]
        contexts = []
        for fold in sorted(roles["outer_fold"].unique()):
            subset = roles[roles["outer_fold"] == fold]
            rows = subset["pair_id"].map(pair_index).to_numpy()
            role = subset["role"].str.lower().to_numpy()
            train, validation, test = rows[role == "train"], rows[role == "validation"], rows[role == "oof_test"]
            contexts.append(Context(protocol, seed, int(fold), train, [(train, validation)], test))
        return contexts
    if protocol == "external_validation":
        all_rows = np.arange(len(bench.main_pairs))
        rng = np.random.default_rng(seed)
        permutation = rng.permutation(all_rows)
        cut = int(round(0.9 * len(permutation)))
        return [Context(protocol, seed, 0, all_rows, [(permutation[:cut], permutation[cut:])],
                        np.arange(len(bench.external_pairs)), test_is_external=True)]
    raise KeyError(protocol)
