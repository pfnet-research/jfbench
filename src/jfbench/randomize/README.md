# Randomize Utilities

This package provides helpers for shuffling constraint instructions in SFT/DPO datasets and
verifying the resulting artifacts.

## Randomizing a dataset

Run the randomizer with the original dataset directory as input and a destination directory for
the randomized dataset:

```bash
python -m jfbench.randomize.run --input-dir data/generated_dataset --output-dir data/generated_dataset_randomized --seed 42
```

The command keeps the original prompt source, constraint types, and metadata while rebuilding the
constraint instructions and prompts. The destination directory must not exist ahead of time because
`Dataset.save_to_disk` creates it during export.

## Validating the randomized dataset

Use the validator to compare the dataset before and after randomization:

```bash
python -m jfbench.randomize.validate --before data/generated_dataset --after data/generated_dataset_randomized
```

You can provide multiple directories via repeated `--before`/`--after` arguments, for example to
compare merged datasets. The validator compares recorded evaluations directly, so it does not make
additional judge calls during verification.

The validator checks three conditions for every `data_id`:

1. The prompt text (`prompt_document` or `prompt`) is identical.
2. The list of constraint types matches exactly.
3. Constraint evaluations, produced by the default judge client during data generation, stay the same
   (for SFT responses as well as DPO chosen/rejected pairs).

The command exits with an error if any mismatch is found.
