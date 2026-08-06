# JFBench: Japanese instruction Following Benchmark

[English version is here](README.md)

JFBenchは、日本語におけるLLMの指示追従性能を評価するためのベンチマークスイートです。生成・評価・集計・可視化のスクリプトに加えて、同じ制約ライブラリを使ってSFT / DPO / GRPO用の学習データセットを構築するスクリプトを提供します。

## セットアップ

依存関係は`uv`で管理しています。

```bash
uv sync
```

いくつかの制約は評価のためにLLM as a Judgeを用いています。デフォルトではOpenRouter経由で`gpt-oss-120b`が使用されます。OpenRouterのAPIキーを環境変数`OPENROUTER_API_KEY`に設定してください。

```bash
export OPENROUTER_API_KEY="your_openrouter_api_key"
```

ローカルのvLLMサーバーを使う場合は、モデル指定に`"provider": "vllm"`を渡してください。vllm providerでモデル名を省略した場合は、環境変数`JFBENCH_LOCAL_MODEL`から読み込まれます。

```bash
export JFBENCH_LOCAL_MODEL="/path/to/model"
```

## スクリプトと引数

以下のスクリプトは`src/jfbench`配下にあります。特に記載がない限り、`uv run python -m <module> ...`の形式で呼び出してください。

### ベンチマーク実行: `src/jfbench/benchmark/eval.py`

例（OpenRouter上のモデルを評価する場合）:

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

例（vLLMサーバーで立てたモデルを評価する場合）: オプションは`extra_body`経由で指定します。詳細は`src/jfbench/llm.py`の実装を確認してください。

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

- `--benchmark`: 使用するベンチマーク。`ifbench`のみサポートしています。デフォルト`ifbench`。
- `--ifbench-dataset-path`: IFBenchのJSONLを外部から指定したい場合のパス。未指定なら`data/`配下の同梱データを使用。
- `--dataset-path`: 構築済みベンチマークデータのディレクトリまたは`.jsonl.zst`ファイルのパス。複数回指定するとマージされます。指定した場合は`--ifbench-dataset-path`からの構築を行わず、こちらが使われます。
- `--output-dir`: 結果を書き出すディレクトリ。デフォルト`data/benchmark_results`。
- `--with-generate/--no-with-generate`: 生成処理の有効/無効。デフォルト有効。
- `--with-eval/--no-with-eval`: 評価処理の有効/無効。デフォルト有効。
- `--override`: 既存の結果があっても再実行するフラグ。デフォルト無効。
- `--n-constraints`: 制約数。カンマ区切りで複数指定可。デフォルト`1`。
- `--constraint-set`: 制約セット（`train`/`test`）。デフォルト`test`。
- `--n-benchmark-data`: 使用する評価データ数。未指定の場合は制約1件時に全件を利用。制約2件以上では必須。
- `--seed`: 乱数シード。デフォルト`42`。
- `--model-specs-json`(必須): 評価対象モデルをJSON文字列で指定します。
- `--judge-model-spec-json`: judgeモデルのJSON文字列（JSONオブジェクト）。デフォルトではOpenRouterの`gpt-oss-120b`をreasoning effort `medium`で使用します。
- `--judge-max-concurrency`: judgeモデルへの同時リクエスト数。デフォルト`32`。
- `--judge-max-retries`: judgeモデルへの各リクエストの最大retry回数。デフォルト`5`。
- `--judge-retry-base-seconds`: judge retry時の指数backoffの基準秒数。デフォルト`1.0`。
- `--n-concurrent-generations`: 生成リクエストの同時送信数。`-1`で全件同時。デフォルト`-1`。

### ベンチマーク集計: `src/jfbench/benchmark/analyze.py`

```bash
uv run python -m jfbench.benchmark.analyze \
  --results-path data/benchmark_results
```

- `--results-path`: 集計対象のJSONLファイルまたはディレクトリ。デフォルト`data/benchmark_results.jsonl`。
- `--constraint`: 指定した制約名を含むレコードのみを対象にするフィルタ。
- `--show-generated`: 集計表の後に生成結果を表示。

### 可視化: `src/jfbench/visualization/visualize.py`

```bash
uv run python -m jfbench.visualization.visualize \
  --input-dir data/benchmark_results \
  --output-dir visualization_output \
  --n-constraints 1 \
  --prompt-source ifbench
```

- `--input-dir`: JSONLファイルを置いたディレクトリ。デフォルト`data/benchmark_results`。
- `--output-dir`: 図表の出力先。デフォルト`visualization_output`。
- `--drop-incomplete`: 評価結果が未完了の行を除外。デフォルト無効。
- `--n-constraints`(必須): 可視化対象の制約数。複数回指定またはカンマ区切りで指定可。
- `--prompt-source`: 対象のプロンプトソース（`ifbench`,`ja_stackoverflow`）。複数回指定またはカンマ区切りで指定可。
- `--models`: 特定モデルに絞る場合に使用。複数指定可。
- `--constraint-set`: `train`/`test`を指定。複数回指定またはカンマ区切りで指定可。デフォルトは両方を含みます。
- `--model-label-map`: 表示名を差し替えるためのJSON文字列。

### SFTデータ生成: `src/jfbench/sft_dataset/generate.py`

制約付きプロンプトをサンプリングし、成功した生成結果をHugging Face Dataset形式で保存します。

```bash
uv run python -m jfbench.sft_dataset.generate \
  --n-data 10000 \
  --prompt-source ja_stackoverflow \
  --constraint-counts 1,2,4 \
  --mode-spec-json '{"provider":"vllm","model":"/path/to/model","model_short":"Local-vLLM","extra_body":{"base_url":"http://localhost:8001/v1"}}' \
  --output-dir data/generated_dataset
```

- `--n-data`(必須): 収集する成功サンプル数。
- `--prompt-source`: `jfbench.prompts`配下のプロンプトソース名（`ifbench`,`ja_stackoverflow`）。デフォルト`ja_stackoverflow`。
- `--output-dir`: 生成したデータセットの書き出し先。デフォルト`data/generated_dataset`。
- `--seed`: 乱数シード。デフォルト`20250101`。
- `--constraint-counts`: 制約数の一覧（カンマ区切り）。デフォルト`1,2,4,8`。
- `--max-attempts`: 各サンプルでモデルが再試行できる回数。デフォルト`3`。
- `--max-workers`: 並列ワーカー数。デフォルト`100`。
- `--save-every`: 指定件数の成功サンプルごとに強制的に保存。デフォルト`1000`。
- `--mode-spec-json`(必須): 生成モデルを指定するJSON文字列。`provider`/`model`/`model_short`が必須。
- `--reward-base-url`,`--reward-api-key`,`--reward-model`,`--reward-token-limit`,`--reward-concurrency`: 候補のスコアリングに使う報酬モデルのエンドポイント設定。
- `--max-failures`: 失敗がこの回数を超えたら中断。デフォルト`1000`。
- `--interactive-mode`: 対話的モードで逐次確認を行う。
- `--log-level`: ログレベル。デフォルト`WARNING`。

関連スクリプト: `screening.py`（生成サンプルのフィルタ）、`to_jsonl.py`（Hugging Face DatasetのJSONL変換）、`analyze.py`（制約数やグループごとの出現頻度の集計）。

### DPOデータ生成: `src/jfbench/dpo_dataset/generate.py`

制約付きプロンプトをサンプリングし、`chosen`/`rejected`のペアをHugging Face Dataset形式で保存します。

```bash
uv run python -m jfbench.dpo_dataset.generate \
  --n-data 5000 \
  --prompt-source ja_stackoverflow \
  --output-dir data/dpo_dataset \
  --constraint-counts 1,2,4 \
  --model-spec-json '{"provider":"vllm","model":"/path/to/model","model_short":"Local-vLLM","extra_body":{"base_url":"http://localhost:8001/v1"}}'
```

- `--n-data`(必須): 収集したいペア数。既存ファイル内の重複をスキップして残りを生成します。
- `--prompt-source`: `jfbench.prompts`配下のプロンプトソース名（`ifbench`,`ja_stackoverflow`）。デフォルト`ja_stackoverflow`。
- `--output-dir`: 書き出し先。デフォルト`data/dpo_dataset`。
- `--seed`: サンプリングシード。デフォルト`20250101`。
- `--constraint-counts`: 制約数の候補。デフォルト`1,2,4,8`。
- `--model-spec-json`(必須): 生成に使用するモデルのJSON設定。
- `--max-workers`,`--save-every`,`--max-failures`: 並列度とチェックポイントの制御。
- `--n-chains`: 初期生成の独立試行数。デフォルト`5`。
- `--n-iterations`: 初期生成後の最大反復数。未指定時は`n_constraints + 1`。
- `--reward-base-url`,`--reward-api-key`,`--reward-model`,`--reward-token-limit`,`--reward-concurrency`: 報酬モデルのエンドポイント設定。
- `--log-level`: ログレベル。デフォルト`INFO`。

関連スクリプト: `postprocess.py`、`screening.py`、`to_jsonl.py`、`analyze.py`。

### GRPOデータ生成: `src/jfbench/grpo_dataset/generate.py`

GRPO学習向けに、ルールで検証可能なプロンプトを構築します。プロンプトソースは`ja_stackoverflow`のみをサポートします。

```bash
uv run python -m jfbench.grpo_dataset.generate \
  --n-data 10000 \
  --constraint-counts 1,2,4 \
  --mode-spec-json '{"provider":"vllm","model":"/path/to/model","model_short":"Local-vLLM","extra_body":{"base_url":"http://localhost:8001/v1"}}' \
  --output-dir data/grpo_dataset
```

引数は`sft_dataset/generate.py`と同様です。`postprocess.py`で生成済みデータセットの制約を再構築し、`to_jsonl.py`でJSONLに変換できます。

### 制約のランダム化: `src/jfbench/randomize/run.py`

プロンプトソース・制約タイプ・メタデータを保ったまま、生成済みデータセットの制約指示文を作り直します。

```bash
uv run python -m jfbench.randomize.run \
  --input-dir data/generated_dataset \
  --output-dir data/generated_dataset_randomized \
  --seed 42
```

ランダム化の前後を比較するには`jfbench.randomize.validate`に`--before`/`--after`を指定してください。記録済みの評価結果を直接比較するため、検証時に追加のjudge呼び出しは発生しません。

## 開発フロー

```bash
uv run pre-commit run --all-files
uv run pytest tests/
```

## ライセンス

本リポジトリのソースコードは[MIT License](LICENSE)で配布しています。

`data/`配下に同梱しているデータファイルは、それぞれ別のライセンスが適用されます。ファイルごとの出典とライセンスは[`data/README.md`](data/README.md)を参照してください。特に以下の点に注意してください。

- `data/ifbench_ja_translated.jsonl` — IFBench由来。[ODC-By v1.0](data/LICENSE-ODC-BY-1.0)。
- `data/ja_stackoverflow_train.jsonl.zst` — 日本語版Stack Overflowのコンテンツ。[CC BY-SA 4.0](data/LICENSE-CC-BY-SA-4.0)。再配布および二次的著作物は、帰属表示を保持し、CC BY-SA 4.0で提供する必要があります。
