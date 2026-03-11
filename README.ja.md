# JFBench: Japanese instruction Following Benchmark

JFBenchは、日本語におけるLLMの指示追従性能を評価するためのベンチマークスイートです。生成・評価・集計・可視化のスクリプトを提供します。

## セットアップ

依存関係は`uv`で管理しています。

```bash
uv sync
```

また、いくつかの制約は評価のためにLLM as a Judgeを用いています。デフォルトではOpenRouter経由で`gpt-oss-120b`が使用されます。OpenRouterのAPIキーを環境変数`OPENROUTER_API_KEY`に設定してください。

```bash
export OPENROUTER_API_KEY="your_openrouter_api_key"
```

## スクリプトと引数

以下のスクリプトは`src/jfbench`配下にあります。

### ベンチマーク実行: `src/jfbench/benchmark/eval.py`

例（OpenRouter上のモデルを評価する場合）:

```bash
uv run python src/jfbench/benchmark/eval.py \
  --benchmark "ifbench" \
  --output-dir data/benchmark_results \
  --n-constraints "1" \
  --constraint-set "test" \
  --n-benchmark-data 1 \
  --model-specs-json  '[{"provider": "openrouter", "model": "qwen/qwen3-30b-a3b-thinking-2507", "model_short": "Qwen3 30B A3B Thinking 2507"}]'
```

例（vLLMサーバーで立てたモデルを評価する場合）: オプションは`extra_body`経由で指定します。詳細は`src/jfbench/llm.py`の実装を確認してください。

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

- `--benchmark`: ベンチマーク名。現時点では`ifbench`のみ対応。デフォルトは`ifbench`。
- `--ifbench-dataset-path`: 外部のIFBench JSONLを指定するパス。デフォルトは`None`で、その時は`data/`以下に同梱されたデータを使用。
- `--dataset-path`: 事前生成済みベンチマークデータセットのディレクトリまたは`.jsonl.zst`ファイルへのパス。複数回指定してマージ可能。指定した場合は`--ifbench-dataset-path`ではなくこちらを優先して読み込む。
- `--output-dir`: 結果JSONLの出力先。デフォルトは`data/benchmark_results`。
- `--with-generate/--no-with-generate`: 生成処理の有効/無効。デフォルト有効。
- `--with-eval/--no-with-eval`: 評価処理の有効/無効。デフォルト有効。
- `--override`: 既存の結果があっても再実行するフラグ。デフォルトは無効。
- `--n-constraints`: 制約数。カンマ区切り指定可。デフォルト`1`。
- `--constraint-set`: 制約セット（`train`/`test`）。デフォルトは`test`。
- `--n-benchmark-data`: 使用する評価データ数。未指定の場合は制約数が1の時には全件を利用。制約数2以上の時は指定必須。
- `--seed`: 乱数シード。デフォルト`42`。
- `--model-specs-json`（必須）: 評価対象モデルのリストを指定するJSON文字列。指定必須。
- `--judge-model-spec-json`: 判定用モデルを指定するJSON文字列。JSONオブジェクトを指定する。デフォルトではOpenrouterの`gpt-oss-120b`がreasoning effort `medium`で使用される。
- `--n-concurrent-generations`: 生成リクエストの同時送信数。`-1`で全件同時に処理する。デフォルトは`-1`。

### ベンチマーク集計: `src/jfbench/benchmark/analyze.py`

例:

```bash
uv run python src/jfbench/benchmark/analyze.py \
  --results-path data/benchmark_results
```

- `--results-path`: 集計対象のJSONLファイルまたはディレクトリ。デフォルト`data/benchmark_results.jsonl`。
- `--constraint`: 指定した制約名を含むレコードのみを対象にするフィルタ。
- `--show-generated`: 生成結果を表形式の後に表示。

### 可視化生成: `src/jfbench/visualization/visualize.py`

例:

```bash
uv run python src/jfbench/visualization/visualize.py \
  --input-dir data/benchmark_results \
  --output-dir visualization_output \
  --n-constraints 1 \
  --prompt-source ifbench
```

- `--input-dir`: JSONLファイルを置いたディレクトリ。デフォルト`data/benchmark_results`。
- `--output-dir`: 図表の出力先。デフォルト`visualization_output`。
- `--drop-incomplete`: 評価結果が未完了の行を除外。デフォルトは無効。
- `--n-constraints`（必須）: 可視化対象の制約数。複数回指定またはカンマ区切り指定。
- `--prompt-source`: 対象のプロンプトソース。`ifbench`のみ対応。複数回指定またはカンマ区切り指定。
- `--models`: 特定モデルに絞る場合に使用。複数指定可。
- `--constraint-set`: `train`/`test`を指定。複数回指定またはカンマ区切り指定。デフォルトでは両方を対象。
- `--model-label-map`: グラフ上の表示名を差し替えるためのJSON文字列。
