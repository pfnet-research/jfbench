# JFBench: Japanese instruction Following Benchmark

[日本語版はこちら](README.ja.md)

JFBench is a benchmark suite for evaluating Japanese LLM instruction-following performance. It provides scripts for generation, evaluation, summary, and visualization, as well as scripts for building SFT / DPO / GRPO training datasets from the same constraint library.

## Setup

Dependencies are managed with `uv`.

```bash
uv sync
```

Some constraints use an LLM as a judge for evaluation. By default, `gpt-oss-120b` is used via OpenRouter. Set the OpenRouter API key in `OPENROUTER_API_KEY`.

```bash
export OPENROUTER_API_KEY="your_openrouter_api_key"
```

To use a local vLLM server instead, pass `"provider": "vllm"` in the model spec. When a model name is omitted for the vllm provider, it is read from `JFBENCH_LOCAL_MODEL`.

```bash
export JFBENCH_LOCAL_MODEL="/path/to/model"
```

## Scripts and Arguments

The scripts below live under `src/jfbench`. Unless noted otherwise, invoke them as `uv run python -m <module> ...`.

### Benchmark Run: `src/jfbench/benchmark/eval.py`

Example (evaluate a model on OpenRouter):

```bash
uv run python -m jfbench.benchmark.eval \
  --benchmark "ifbench" \
  --output-dir data/benchmark_results \
  --n-constraints "1" \
  --constraint-set "test" \
  --n-benchmark-data 1 \
  --model-specs-json  '[{"provider": "openrouter", "model": "qwen/qwen3-30b-a3b-thinking-2507", "model_short": "Qwen3 30B A3B Thinking 2507"}]' \
  --judge-model-spec-json '{"provider": "openrouter", "model": "openai/gpt-oss-120b", "model_short": "gpt-oss-120b", "extra_body": {"reasoning_effort": "medium"}}'
```

Example (evaluate a local vLLM server): options are passed via `extra_body`. See `src/jfbench/llm.py` for details.

```bash
uv run python -m jfbench.benchmark.eval \
  --benchmark "ifbench" \
  --output-dir data/benchmark_results \
  --n-constraints "1" \
  --constraint-set "test" \
  --n-benchmark-data 1 \
  --model-specs-json  '[{"provider": "vllm", "model": "/path/to/model_to_evaluate", "model_short": "Model to evaluate", "extra_body": {"base_url": "http://localhost:8001/v1"}}]'\
  --judge-model-spec-json '{"provider": "vllm", "model": "/path/to/judge_model", "model_short": "Local vLLM Judge", "extra_body": {"base_url": "http://localhost:8000/v1"}}'
```

- `--benchmark`: Benchmark name. Only `ifbench` is supported. Default `ifbench`.
- `--ifbench-dataset-path`: Path to an external IFBench JSONL file. Default `None`, which uses the bundled dataset under `data/`.
- `--dataset-path`: Path to a prebuilt benchmark dataset directory or `.jsonl.zst` file. Can be specified multiple times to merge datasets. When provided, this is used instead of building from `--ifbench-dataset-path`.
- `--output-dir`: Directory for result JSONL files. Default `data/benchmark_results`.
- `--with-generate/--no-with-generate`: Enable or disable generation. Default enabled.
- `--with-eval/--no-with-eval`: Enable or disable evaluation. Default enabled.
- `--override`: Re-run even if matching entries already exist. Default disabled.
- `--n-constraints`: Number of constraints. Comma-separated values supported. Default `1`.
- `--constraint-set`: Constraint set (`train`/`test`). Default `test`.
- `--n-benchmark-data`: Number of entries to use. If omitted, use all entries when `n_constraints` is 1. Required when `n_constraints` is 2 or higher.
- `--seed`: Random seed. Default `42`.
- `--model-specs-json` (required): JSON string that lists the evaluated models.
- `--judge-model-spec-json`: JSON string for the judge model. Pass a JSON object. By default, OpenRouter `gpt-oss-120b` is used with reasoning effort `medium`.
- `--judge-max-concurrency`: Concurrent judge requests. Default `32`.
- `--judge-max-retries`: Maximum retries per judge request. Default `5`.
- `--judge-retry-base-seconds`: Base seconds for exponential backoff between judge retries. Default `1.0`.
- `--n-concurrent-generations`: Concurrent generation requests. Use `-1` to send all at once. Default `-1`.

### Benchmark Summary: `src/jfbench/benchmark/analyze.py`

Example:

```bash
uv run python -m jfbench.benchmark.analyze \
  --results-path data/benchmark_results
```

- `--results-path`: JSONL file or directory to analyze. Default `data/benchmark_results.jsonl`.
- `--constraint`: Filter to records that include the named constraint.
- `--show-generated`: Show generated responses after the summary table.

### Visualization: `src/jfbench/visualization/visualize.py`

Example:

```bash
uv run python -m jfbench.visualization.visualize \
  --input-dir data/benchmark_results \
  --output-dir visualization_output \
  --n-constraints 1 \
  --prompt-source ifbench
```

- `--input-dir`: Directory with result JSONL files. Default `data/benchmark_results`.
- `--output-dir`: Output directory for charts. Default `visualization_output`.
- `--drop-incomplete`: Drop rows without completed evaluations. Default disabled.
- `--n-constraints` (required): Constraint counts to include. Can be repeated or comma-separated.
- `--prompt-source`: Prompt sources to include (`ifbench`, `ja_stackoverflow`). Can be repeated or comma-separated.
- `--models`: Filter to specific model names. Can be repeated.
- `--constraint-set`: Constraint set filters (`train`/`test`). Can be repeated or comma-separated. By default both are included.
- `--model-label-map`: JSON string mapping model labels.

### SFT Dataset Generation: `src/jfbench/sft_dataset/generate.py`

Samples constrained prompts and stores the successful generations as a Hugging Face dataset.

```bash
uv run python -m jfbench.sft_dataset.generate \
  --n-data 10000 \
  --prompt-source ja_stackoverflow \
  --constraint-counts 1,2,4 \
  --mode-spec-json '{"provider":"vllm","model":"/path/to/model","model_short":"Local-vLLM","extra_body":{"base_url":"http://localhost:8001/v1"}}' \
  --output-dir data/generated_dataset
```

- `--n-data` (required): Number of successful samples to collect.
- `--prompt-source`: Prompt source under `jfbench.prompts` (`ifbench`, `ja_stackoverflow`). Default `ja_stackoverflow`.
- `--output-dir`: Output directory for the dataset. Default `data/generated_dataset`.
- `--seed`: Random seed. Default `20250101`.
- `--constraint-counts`: Comma-separated constraint counts. Default `1,2,4,8`.
- `--max-attempts`: Retries allowed per sample. Default `3`.
- `--max-workers`: Number of concurrent workers. Default `100`.
- `--save-every`: Force a save after this many successful samples. Default `1000`.
- `--mode-spec-json` (required): JSON string for the generation model. `provider`/`model`/`model_short` are required.
- `--reward-base-url`, `--reward-api-key`, `--reward-model`, `--reward-token-limit`, `--reward-concurrency`: Reward model endpoint settings used for scoring candidates.
- `--max-failures`: Abort after this many failures. Default `1000`.
- `--interactive-mode`: Confirm each sample interactively.
- `--log-level`: Log level. Default `WARNING`.

Companion scripts: `screening.py` filters generated samples, `to_jsonl.py` converts a Hugging Face dataset into JSONL, and `analyze.py` reports constraint counts and group frequencies.

### DPO Dataset Generation: `src/jfbench/dpo_dataset/generate.py`

Samples constrained prompts and stores `chosen`/`rejected` pairs as a Hugging Face dataset.

```bash
uv run python -m jfbench.dpo_dataset.generate \
  --n-data 5000 \
  --prompt-source ja_stackoverflow \
  --output-dir data/dpo_dataset \
  --constraint-counts 1,2,4 \
  --model-spec-json '{"provider":"vllm","model":"/path/to/model","model_short":"Local-vLLM","extra_body":{"base_url":"http://localhost:8001/v1"}}'
```

- `--n-data` (required): Number of preference pairs. Existing entries are skipped and only the remainder is generated.
- `--prompt-source`: Prompt source under `jfbench.prompts` (`ifbench`, `ja_stackoverflow`). Default `ja_stackoverflow`.
- `--output-dir`: Output directory. Default `data/dpo_dataset`.
- `--seed`: Sampling seed. Default `20250101`.
- `--constraint-counts`: Constraint counts. Default `1,2,4,8`.
- `--model-spec-json` (required): JSON config for the generation model.
- `--max-workers`, `--save-every`, `--max-failures`: Concurrency and checkpointing controls.
- `--n-chains`: Number of independent initial generations. Default `5`.
- `--n-iterations`: Maximum iterations after the initial generation. Defaults to `n_constraints + 1`.
- `--reward-base-url`, `--reward-api-key`, `--reward-model`, `--reward-token-limit`, `--reward-concurrency`: Reward model endpoint settings.
- `--log-level`: Log level. Default `INFO`.

Companion scripts: `postprocess.py`, `screening.py`, `to_jsonl.py`, and `analyze.py`.

### GRPO Dataset Generation: `src/jfbench/grpo_dataset/generate.py`

Builds rule-verifiable prompts for GRPO training. Only `ja_stackoverflow` is supported as a prompt source.

```bash
uv run python -m jfbench.grpo_dataset.generate \
  --n-data 10000 \
  --constraint-counts 1,2,4 \
  --mode-spec-json '{"provider":"vllm","model":"/path/to/model","model_short":"Local-vLLM","extra_body":{"base_url":"http://localhost:8001/v1"}}' \
  --output-dir data/grpo_dataset
```

Arguments mirror `sft_dataset/generate.py`. `postprocess.py` reconstructs constraints on a generated dataset, and `to_jsonl.py` converts it to JSONL.

### Constraint Randomization: `src/jfbench/randomize/run.py`

Rebuilds constraint instructions in a generated dataset while keeping the prompt source, constraint types, and metadata intact.

```bash
uv run python -m jfbench.randomize.run \
  --input-dir data/generated_dataset \
  --output-dir data/generated_dataset_randomized \
  --seed 42
```

Use `jfbench.randomize.validate` with `--before`/`--after` to compare a dataset before and after randomization. The validator compares recorded evaluations directly, so it makes no additional judge calls.

## Development

```bash
uv run pre-commit run --all-files
uv run pytest tests/
```

## License

The source code in this repository is distributed under the [MIT License](LICENSE).

The data files bundled under `data/` are covered by their own licenses. See [`data/README.md`](data/README.md) for the per-file source and license, in particular:

- `data/ifbench_ja_translated.jsonl` — derived from IFBench, licensed under [ODC-By v1.0](data/LICENSE-ODC-BY-1.0).
- `data/ja_stackoverflow_train.jsonl.zst` — content from Japanese Stack Overflow, licensed under [CC BY-SA 4.0](data/LICENSE-CC-BY-SA-4.0). Redistribution and adaptations must retain attribution and remain under CC BY-SA 4.0.
