# BiAx

BiAx is a shared bi-axis architecture for Formula - ADR, Drug - drug interaction,
Drug pair - ADR, and Herb - drug interaction prediction. The repository contains
the four task instances and their task-specific conditional experts. Each run
trains one model. No model ensembling is used.

## Installation

Python 3.12 and PyTorch 2.9 are required. The package also declares its numeric
and table dependencies. Install the package and its test dependency in a clean
environment:

```bash
python -m pip install -e '.[test]'
python -m pytest
```

## Data

The repository includes every protocol split used by the model. Place the
companion BiAx data deposit under `data/` so that each task directory also has
its `inputs`, `entities`, and `labels` directories. The expected layout is:

```text
data/
  formula_adverse_reaction/
  drug_drug_interaction/
  drug_pair_adverse_reaction/
  herb_drug_interaction/
```

The public loaders accept the corresponding task directory as `data_root`.

## Herb - drug interaction reproduction

The per-seed core and router checkpoints, route strengths, and expected metrics
are included under `parameters/herb_drug_interaction/`. After placing the
companion data deposit under `data/`, reproduce all three protocols and ten
seeds with:

```bash
biax-reproduce-herb-drug \
  --data-root data/herb_drug_interaction \
  --parameter-root parameters/herb_drug_interaction \
  --device cuda:0 \
  --output herb_drug_reproduction.json
```

The command evaluates the released checkpoints on the released partitions,
selects each decision threshold from validation data, and checks the displayed
mean and standard deviation for every protocol and metric against the included
result record. Per-seed differences are retained in the reproduction report.

## Minimal checks

`pytest` runs a forward pass for every task instance, verifies symmetry of the
drug-pair encoder, and checks both invariants of the herb-support router. Set
`BIAX_HERB_DRUG_DATA` to the herb-drug data directory to include the complete
30-run reproduction test. Model outputs are logits; decision thresholds are
selected using the validation partition only.

## Repository boundary

This repository contains model code and protocol splits. It contains no figure
generation, manuscript, docking, molecular-dynamics, or case-analysis code.
