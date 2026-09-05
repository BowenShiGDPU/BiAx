# BORA

Code for **Predicting Medication Interactions and Adverse Reactions from
Biomedical Context and Observed Relations**.

BORA combines biomedical context with observed relations to predict
Formula–ADR associations, drug–drug interactions, drug pair–ADR associations,
and herb–drug interactions. Each task is trained separately. The repository
includes the four model implementations, task-specific prediction components,
data loaders, evaluation splits, and herb–drug model parameters.

## Installation

Use Python 3.12 or later and PyTorch 2.9. Install the package and tests:

```bash
python -m pip install -e '.[test]'
python -m pytest
```

## Data

Extract `BORA_data_and_supporting_tables.tar.gz` and copy the four directories
under `bora_data_and_tables/datasets/` into this repository's `data/` directory.
The directories contain the numerical inputs, entity registries, labels and
evaluation splits:

```text
data/
  formula_adverse_reaction/
  drug_drug_interaction/
  drug_pair_adverse_reaction/
  herb_drug_interaction/
```

Each loader accepts its task directory as `data_root`. The companion archive
also contains machine-readable results and figure source data. See its
`DATA_DICTIONARY.md` and `SOURCES_AND_RIGHTS.md` for schemas and source licences.

## Reproduce herb–drug interaction results

The parameters for three protocols and ten seeds are in
`parameters/herb_drug_interaction/`. Evaluate them with:

```bash
bora-reproduce-herb-drug \
  --data-root data/herb_drug_interaction \
  --parameter-root parameters/herb_drug_interaction \
  --device cuda:0 \
  --output herb_drug_reproduction.json
```

The command loads the supplied parameters, selects decision thresholds on
validation data, and compares test-set means and standard deviations with the
included results. It does not retrain the models.

## Tests

The tests cover forward passes for all four tasks, drug-pair symmetry and
support-dependent herb–drug routing. Set `BORA_HERB_DRUG_DATA` to the herb–drug
data directory to include the complete 30-run reproduction test. Set
`BORA_DEVICE` to select a device; the default is `cpu`.

Model outputs are logits. Decision thresholds are selected on validation
partitions, not test partitions.

## Licence

The code is available under the MIT licence. Third-party data remain subject
to the terms described in the companion archive.
