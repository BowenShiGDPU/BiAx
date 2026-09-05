from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (average_precision_score, matthews_corrcoef,
                             precision_recall_fscore_support, roc_auc_score)

from .data import (Bundle, Split, load_bundle, load_constituents,
                   load_mechanism, load_split)
from .model import BORAHerbDrug, Config
from .unseen_herb_expert import UnseenHerbExpert


PROTOCOL_CONFIGS = {
    "random_split": Config(fusion="C", support_dropout=0.0,
                           graph_expert=True, cold_expert=False),
    "unseen_herb": Config(fusion="A", support_dropout=0.3,
                          graph_expert=False, cold_expert=False),
    "unseen_drug": Config(fusion="A", support_dropout=0.3,
                          support_dropout_drug=0.3, graph_expert=False,
                          cold_expert=True, hard_cold_route=True),
}


def best_threshold(label: np.ndarray, probability: np.ndarray) -> float:
    grid = np.unique(np.quantile(probability, np.linspace(0.50, 0.999, 200)))
    best, key = float(grid[0]), (-2.0, -1.0, -1.0)
    n_positive = float((label == 1).sum())
    n_negative = float((label == 0).sum())
    for threshold in grid:
        prediction = probability >= threshold
        true_positive = float((prediction & (label == 1)).sum())
        false_positive = float((prediction & (label == 0)).sum())
        false_negative = n_positive - true_positive
        true_negative = n_negative - false_positive
        precision = true_positive / max(1.0, true_positive + false_positive)
        recall = true_positive / max(1.0, true_positive + false_negative)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        denominator = np.sqrt(max(
            1e-12,
            (true_positive + false_positive) *
            (true_positive + false_negative) *
            (true_negative + false_positive) *
            (true_negative + false_negative),
        ))
        mcc = ((true_positive * true_negative -
                false_positive * false_negative) / denominator)
        if (mcc, f1, recall) > key:
            best, key = float(threshold), (mcc, f1, recall)
    return best


def metrics(label: np.ndarray, probability: np.ndarray,
            threshold: float) -> dict[str, float]:
    prediction = (probability >= threshold).astype(np.int8)
    precision, recall, f1, _ = precision_recall_fscore_support(
        label, prediction, average="binary", zero_division=0)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": float(roc_auc_score(label, probability)),
        "aupr": float(average_precision_score(label, probability)),
        "mcc": float(matthews_corrcoef(label, prediction)),
    }


def _axis_indices(bundle: Bundle) -> tuple[np.ndarray, np.ndarray,
                                           np.ndarray, np.ndarray]:
    herb_rows = np.flatnonzero(np.char.startswith(bundle.ids, "HERB:"))
    drug_rows = np.flatnonzero(np.char.startswith(bundle.ids, "DRUG:"))
    herb_of = np.full(len(bundle.ids), -1, dtype=np.int64)
    drug_of = np.full(len(bundle.ids), -1, dtype=np.int64)
    herb_of[herb_rows] = np.arange(len(herb_rows))
    drug_of[drug_rows] = np.arange(len(drug_rows))
    return herb_rows, drug_rows, herb_of, drug_of


def _training_labels(split: Split, herb_index: np.ndarray,
                     drug_index: np.ndarray, n_herb: int,
                     n_drug: int) -> tuple[np.ndarray, np.ndarray]:
    training = split.part == "train"
    labels = np.zeros((n_herb, n_drug), dtype=np.float32)
    observed = np.zeros((n_herb, n_drug), dtype=np.float32)
    labels[herb_index[training], drug_index[training]] = \
        split.label[training].astype(np.float32)
    observed[herb_index[training], drug_index[training]] = 1.0
    return labels, observed


@torch.no_grad()
def _score(model: BORAHerbDrug, expert: UnseenHerbExpert, bundle: Bundle,
           split: Split, constituents: np.lib.npyio.NpzFile,
           mechanism: np.ndarray, alpha: float, device: torch.device,
           rows: np.ndarray, batch_size: int = 100_000) -> np.ndarray:
    herb_rows, drug_rows, herb_of, drug_of = _axis_indices(bundle)
    herb_index = herb_of[split.left]
    drug_index = drug_of[split.right]
    labels, observed = _training_labels(
        split, herb_index, drug_index, len(herb_rows), len(drug_rows))

    semantic_herb = torch.from_numpy(bundle.semantic[herb_rows]).to(device)
    graph_herb = torch.from_numpy(bundle.graph[herb_rows]).to(device)
    structure_herb = torch.from_numpy(bundle.structure[herb_rows]).to(device)
    semantic_drug = torch.from_numpy(bundle.semantic[drug_rows]).to(device)
    graph_drug = torch.from_numpy(bundle.graph[drug_rows]).to(device)
    structure_drug = torch.from_numpy(bundle.structure[drug_rows]).to(device)
    constituent_tensor = torch.from_numpy(
        constituents["constituents"]).to(device)
    constituent_mask = torch.from_numpy(constituents["mask"]).to(device)

    model.eval()
    expert.eval()
    herb_tokens, _, herb_state = model.encode_herb(
        semantic_herb, graph_herb, structure_herb,
        constituent_tensor, constituent_mask)
    _, drug_state = model.encode_drug(
        semantic_drug, graph_drug, structure_drug)
    memory = model.retrieve_relations(
        herb_state, drug_state, torch.from_numpy(labels).to(device),
        torch.from_numpy(observed).to(device))

    output = np.empty(len(rows), dtype=np.float64)
    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows))
        selected = rows[start:stop]
        herb = torch.from_numpy(herb_index[selected]).to(device)
        drug = torch.from_numpy(drug_index[selected]).to(device)
        pair_mechanism = torch.from_numpy(np.asarray(
            mechanism[split.left[selected], split.right[selected]])).to(device)
        core_logit = model.score(
            herb_tokens, constituent_mask, herb_state, drug_state,
            herb, drug, pair_mechanism, memory)
        probability = model.route_unseen_herb(
            core_logit, expert, herb, drug, alpha)
        output[start:stop] = probability.detach().cpu().numpy()
    return output


def reproduce_run(data_root: str | Path, parameter_root: str | Path,
                  protocol: str, seed: int, device: str = "cpu") -> dict:
    if protocol not in PROTOCOL_CONFIGS:
        raise ValueError(f"unknown protocol: {protocol}")
    data_root, parameter_root = Path(data_root), Path(parameter_root)
    index = json.loads(
        (parameter_root / "router_parameters.json").read_text(encoding="utf-8"))
    expected = next(
        row for row in index["protocols"][protocol]
        if int(row["seed"]) == seed)

    bundle = load_bundle(data_root)
    split = load_split(data_root, protocol, seed, bundle)
    constituents = load_constituents(data_root)
    herb_rows, drug_rows, _, _ = _axis_indices(bundle)
    if list(constituents["herb_ids"].astype(str)) != \
            list(bundle.ids[herb_rows]):
        raise ValueError("constituent tensor is not aligned with herb entities")

    torch_device = torch.device(device)
    model = BORAHerbDrug(len(herb_rows), len(drug_rows),
                    PROTOCOL_CONFIGS[protocol]).to(torch_device)
    expert = UnseenHerbExpert(
        PROTOCOL_CONFIGS[protocol].semantic_dim +
        PROTOCOL_CONFIGS[protocol].graph_dim +
        PROTOCOL_CONFIGS[protocol].structure_dim).to(torch_device)
    model.load_state_dict(torch.load(
        parameter_root / expected["core_checkpoint"],
        map_location=torch_device, weights_only=True), strict=True)
    expert.load_state_dict(torch.load(
        parameter_root / expected["expert_checkpoint"],
        map_location=torch_device, weights_only=True), strict=True)

    validation_rows = np.flatnonzero(split.part == "validation")
    test_rows = np.flatnonzero(split.part == "test")
    mechanism = load_mechanism(data_root)
    validation_probability = _score(
        model, expert, bundle, split, constituents, mechanism,
        float(expected["alpha"]), torch_device, validation_rows)
    threshold = best_threshold(
        split.label[validation_rows], validation_probability)
    test_probability = _score(
        model, expert, bundle, split, constituents, mechanism,
        float(expected["alpha"]), torch_device, test_rows)
    observed = metrics(split.label[test_rows], test_probability, threshold)
    metric_differences = {
        name: observed[name] - float(expected["test_metrics"][name])
        for name in observed
    }
    return {
        "protocol": protocol,
        "seed": seed,
        "alpha": float(expected["alpha"]),
        "threshold": threshold,
        "expected_threshold": float(expected["validation_threshold"]),
        "metrics": observed,
        "expected_metrics": expected["test_metrics"],
        "metric_differences": metric_differences,
        "maximum_absolute_metric_difference": max(
            abs(value) for value in metric_differences.values()),
    }


def reproduce_all(data_root: str | Path, parameter_root: str | Path,
                  device: str = "cpu", reported_decimals: int = 3,
                  protocols: tuple[str, ...] | None = None,
                  seeds: tuple[int, ...] | None = None) -> dict:
    parameter_root = Path(parameter_root)
    index = json.loads(
        (parameter_root / "router_parameters.json").read_text(encoding="utf-8"))
    selected_protocols = protocols or tuple(PROTOCOL_CONFIGS)
    results = []
    for protocol in selected_protocols:
        available = [int(row["seed"])
                     for row in index["protocols"][protocol]]
        selected_seeds = seeds or tuple(available)
        for seed in selected_seeds:
            if seed not in available:
                raise ValueError(f"seed {seed} is unavailable for {protocol}")
            results.append(reproduce_run(
                data_root, parameter_root, protocol, seed, device))
    maximum = max(row["maximum_absolute_metric_difference"]
                  for row in results)
    summaries = {}
    mismatches = []
    metric_names = tuple(results[0]["metrics"])
    for protocol in selected_protocols:
        protocol_runs = [row for row in results if row["protocol"] == protocol]
        protocol_summary = {}
        for metric_name in metric_names:
            observed_values = np.asarray(
                [row["metrics"][metric_name] for row in protocol_runs])
            expected_values = np.asarray(
                [row["expected_metrics"][metric_name]
                 for row in protocol_runs])
            statistics = {
                "mean": (float(observed_values.mean()),
                         float(expected_values.mean())),
                "standard_deviation": (
                    float(observed_values.std(ddof=1)),
                    float(expected_values.std(ddof=1))),
            }
            protocol_summary[metric_name] = {}
            for statistic, (observed_value, expected_value) in statistics.items():
                observed_text = f"{observed_value:.{reported_decimals}f}"
                expected_text = f"{expected_value:.{reported_decimals}f}"
                matches = observed_text == expected_text
                protocol_summary[metric_name][statistic] = {
                    "observed": observed_value,
                    "expected": expected_value,
                    "observed_display": observed_text,
                    "expected_display": expected_text,
                    "matches": matches,
                }
                if not matches:
                    mismatches.append({
                        "protocol": protocol,
                        "metric": metric_name,
                        "statistic": statistic,
                        "observed": observed_text,
                        "expected": expected_text,
                    })
        summaries[protocol] = protocol_summary
    report = {
        "runs": results,
        "run_count": len(results),
        "reported_decimals": reported_decimals,
        "protocol_summaries": summaries,
        "reported_value_mismatches": mismatches,
        "maximum_absolute_metric_difference": maximum,
        "all_runs_match": not mismatches,
    }
    if not report["all_runs_match"]:
        raise RuntimeError(f"reported result mismatch: {mismatches}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce the Herb - drug interaction results")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--parameter-root", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reported-decimals", type=int, default=3)
    parser.add_argument("--output")
    arguments = parser.parse_args()
    report = reproduce_all(
        arguments.data_root, arguments.parameter_root,
        arguments.device, arguments.reported_decimals)
    rendered = json.dumps(report, indent=2) + "\n"
    if arguments.output:
        Path(arguments.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
