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

## Minimal checks

`pytest` runs a forward pass for every task instance and verifies symmetry of
the drug-pair encoder. Model outputs are logits; decision thresholds are
selected using the validation partition only.

## Repository boundary

This repository contains model code and protocol splits. It contains no figure
generation, manuscript, docking, molecular-dynamics, or case-analysis code.
