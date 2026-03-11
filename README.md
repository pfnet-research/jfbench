# JFBench: Japanese instruction Following Benchmark

JFBench is a benchmark suite for evaluating Japanese LLM instruction-following performance. It provides scripts for generation, evaluation, summary, and visualization.

## Setup

Dependencies are managed with `uv`.

```bash
uv sync
```

Some constraints use an LLM as a judge for evaluation. By default, `gpt-oss-120b` is used via OpenRouter. Set the OpenRouter API key in `OPENROUTER_API_KEY`.

```bash
export OPENROUTER_API_KEY="your_openrouter_api_key"
```

## Scripts and Arguments

The scripts below live under `src/jfbench`.

### Benchmark Run: `src/jfbench/benchmark/eval.py`

Example (evaluate a model on OpenRouter):

```bash
uv run python src/jfbench/benchmark/eval.py \
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
uv run python src/jfbench/benchmark/eval.py \
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
- `--n-concurrent-generations`: Concurrent generation requests. Use `-1` to send all at once. Default `-1`.

### Benchmark Summary: `src/jfbench/benchmark/analyze.py`

Example:

```bash
uv run python src/jfbench/benchmark/analyze.py \
  --results-path data/benchmark_results
```

- `--results-path`: JSONL file or directory to analyze. Default `data/benchmark_results.jsonl`.
- `--constraint`: Filter to records that include the named constraint.
- `--show-generated`: Show generated responses after the summary table.

### Visualization: `src/jfbench/visualization/visualize.py`

Example:

```bash
uv run python src/jfbench/visualization/visualize.py \
  --input-dir data/benchmark_results \
  --output-dir visualization_output \
  --n-constraints 1 \
  --prompt-source ifbench
```

- `--input-dir`: Directory with result JSONL files. Default `data/benchmark_results`.
- `--output-dir`: Output directory for charts. Default `visualization_output`.
- `--drop-incomplete`: Drop rows without completed evaluations. Default disabled.
- `--n-constraints` (required): Constraint counts to include. Can be repeated or comma-separated.
- `--prompt-source`: Prompt sources to include. Only `ifbench` is supported. Can be repeated or comma-separated.
- `--models`: Filter to specific model names. Can be repeated.
- `--constraint-set`: Constraint set filters (`train`/`test`). Can be repeated or comma-separated. By default both are included.
- `--model-label-map`: JSON string mapping model labels.
