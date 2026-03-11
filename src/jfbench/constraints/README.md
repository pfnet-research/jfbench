# Constraints Catalog

`src/jfbench/constraints` 配下の全制約クラスを、制約グループごとに整理した一覧です。

- 対象クラス数: 166
- 抽出対象: `ConstraintGroupMixin` を継承する全クラス
- `train`で学習に使えるか: `instructions(train_or_test="train")` が非空文字列を返すかで判定
- 評価方式: `__init__` に `client` 引数があるものを `LLM-based`、それ以外を `Rule-based` として分類

> Note: `No (ValueError)` は `train_or_test="train"` がサポート外であることを示します。

## character (13)

| Constraint | 必須引数 | 制約の概要 | trainでinstruction返却 | 評価方式 |
|---|---|---|---|---|
| `FullWidthCharacterConstraint` | - | 文字はすべて全角にしてください。 | Yes | Rule-based |
| `HalfWidthCharacterConstraint` | - | 文字はすべて半角に統一してください。 | Yes | Rule-based |
| `HiraganaCharacterConstraint` | - | ひらがな表記のみを用いてください（数字・句読点・記号は可）。 | Yes | Rule-based |
| `JapanesePunctuationConstraint` | - | 日本語の句読点以外（, .）を使わないでください。 | Yes | Rule-based |
| `KanjiCharacterConstraint` | - | 漢字表記のみで回答してください（記号可）。 | Yes | Rule-based |
| `KatakanaCharacterConstraint` | - | カタカナ表記のみで回答してください（数字・記号可）。 | Yes | Rule-based |
| `LowercaseCharacterConstraint` | - | 英字を含める場合、必ず小文字で書いてください。 | Yes | Rule-based |
| `NoCommasConstraint` | - | コンマ記号を排除し、別の区切り方で文章を構成してください。 | Yes | Rule-based |
| `NoJapanesePunctuationConstraint` | - | 「、」や「。」を用いずに情報を表現してください。 | Yes | Rule-based |
| `NoSuffixWhitespaceConstraint` | - | 出力の最後は文字で終わらせ、空白文字で締めくくることを避けてください。 | Yes | Rule-based |
| `NoWhitespaceConstraint` | - | 改行・スペース・タブなどの空白を利用せずに回答を構成してください。 | Yes | Rule-based |
| `RomajiCharacterConstraint` | - | ローマ字以外の文字を含めないでください。 | Yes | Rule-based |
| `UppercaseCharacterConstraint` | - | 大文字のアルファベットを含め、英字に小文字を混ぜないでください。 | Yes | Rule-based |

## content (8)

| Constraint | 必須引数 | 制約の概要 | trainでinstruction返却 | 評価方式 |
|---|---|---|---|---|
| `AbstractExcludedContentConstraint` | `client`, `document`, `content` | contentに関する情報は出さず、それ以外の内容だけを述べてください。 | Yes | LLM-based |
| `AbstractIncludedContentConstraint` | `client`, `document`, `content` | contentを出力内で取り上げ、その要素を明言する形にしてください。 | Yes | LLM-based |
| `IntrinsicContentConstraint` | `client`, `document` | 指示文の内容だけを根拠にし、追加の推測や外部情報を含めないでください。 | Yes | LLM-based |
| `IrrevantContentConstraint` | `client`, `document` | 元の文章と無関係なトピックで答えてください。 | Yes | LLM-based |
| `KeywordExcludedContentConstraint` | `keywords` | keywordといった語句は禁止なので、別の言い回しに置き換えてください。 | Yes | Rule-based |
| `KeywordIncludedContentConstraint` | `keywords` | keywordを1回が正確な出現回数で含まれるよう文を構成してください。 | Yes | Rule-based |
| `NoReasonContentConstraint` | `client` | 説明を加えずに結果だけを提示してください。 | Yes | LLM-based |
| `ReasonContentConstraint` | `client`, `document` | 根拠となる情報を指示文から取り出して提示してください。 | Yes | LLM-based |

## format (32)

| Constraint | 必須引数 | 制約の概要 | trainでinstruction返却 | 評価方式 |
|---|---|---|---|---|
| `BulletPointsFormatConstraint` | - | 各箇条書きの先頭記号として-を用い、最低1項目は必ず含めてください。 | Yes | Rule-based |
| `CitationFormatConstraint` | - | [n] / [^n] または著者年方式で本文中に出典を明示し、文末に全出典の一覧を揃えてください。 | Yes | Rule-based |
| `CsvFormatConstraint` | - | 行ごとの列数を揃えたCSV（comma-separated values）を出力してください。 | Yes | Rule-based |
| `DiffFormatConstraint` | - | パッチ形式の unified diff を生成してください。 | Yes | Rule-based |
| `HtmlFormatConstraint` | - | 開閉タグや入れ子を整えたHTML5形式で回答を提出してください。 | Yes | Rule-based |
| `HtmlTableFormatConstraint` | - | 以下の制約を守ったHTML表を生成してください: | Yes | Rule-based |
| `IndentFormatConstraint` | `indent` | 冒頭のインデントは、付ける場合に限り' 'を単位とした繰り返しに統一してください。 | Yes | Rule-based |
| `JavascriptFormatConstraint` | - | JavaScriptの構文規則に従ったコードのみを返し、無効な文は含めないでください。 | Yes | Rule-based |
| `JsonFormatConstraint` | - | キーや値を適切に引用したJSON出力にしてください。 | Yes | Rule-based |
| `LatexFormatConstraint` | - | 回答全体を有効なLaTeX文書として書き、構文エラーを含まないようにしてください。 | Yes | Rule-based |
| `LatexTableFormatConstraint` | - | 余計なテキストを入れず、tabular環境だけで表を作成してください。 | Yes | Rule-based |
| `MarkdownClosedFencesConstraint` | - | フェンスを開いたら同じ記号で閉じる完全なコードブロック構造を守ってMarkdownで書いてください。 | Yes | Rule-based |
| `MarkdownFormatConstraint` | - | 示すルールを順守したMarkdown出力のみを受け付けます: | Yes | Rule-based |
| `MarkdownHeadingJumpsConstraint` | - | Markdown形式で書いてください。段階を飛ばさずに見出しを配置し、少なくとも1つは見出しを用意してください。 | Yes | Rule-based |
| `MarkdownHeadingsStructureConstraint` | - | 以下のヘッダー要件を満たすMarkdownのみ有効です: | Yes | Rule-based |
| `MarkdownLinksAndImagesConstraint` | - | Markdownで書いてください。リンク/画像の検査に通るよう、下記ルールを満たしてください: | Yes | Rule-based |
| `MarkdownListStructureConstraint` | - | Markdownとしてリストの品質基準を満たしてください。条件は次の通りです: | Yes | Rule-based |
| `MarkdownParseableConstraint` | - | パーサーが失敗しない正しいMarkdown構造で文章を組み立ててください。 | Yes | Rule-based |
| `MarkdownReferenceLinksConstraint` | - | Markdownで書いてください。内部アンカーを含めた参照リンクの整合性を保ち、次のルールを守ってください: | Yes | Rule-based |
| `MarkdownTableFormatConstraint` | - | ヘッダー行および---区切りを備えた一つのMarkdown表を出力し、それ以外の内容は含めないでください。 | Yes | Rule-based |
| `MarkdownTableStructureConstraint` | - | Markdownの表の品質を保つため、行ごとのセル数とヘッダーの空欄禁止を守ってください: | Yes | Rule-based |
| `MarkdownUnconsumedEmphasisMarkersConstraint` | - | Markdownで書いてください。強調マーカー(*, _)を適切にペアにし、次のルールを守ってください: | Yes | Rule-based |
| `MediawikiTableFormatConstraint` | - | MediaWiki形式の表を1つだけ出力し、説明文やその他のテキストを混在させないでください。 | Yes | Rule-based |
| `NoCodeFenceJavascriptFormatConstraint` | - | バッククォートで囲まずにJSコードを書いてください。 | Yes | Rule-based |
| `NoCodeFencePythonFormatConstraint` | - | バッククォートなしでPythonコードを提示してください。 | Yes | Rule-based |
| `PythonFormatConstraint` | - | 無効な構文を含めず、完全なPythonコードとして出力を提供してください。 | Yes | Rule-based |
| `SentenceDelimiterFormatConstraint` | `delimiter` | 文を繋ぐときは必ず「。」を境界として入れてください。 | Yes | Rule-based |
| `TsvFormatConstraint` | - | タブ区切りのテーブルだけを返し、列数が揃っていることを保証してください。 | Yes | Rule-based |
| `WithCodeFenceJavascriptFormatConstraint` | - | ```javascript 形式でコードを出力してください。 | Yes | Rule-based |
| `WithCodeFencePythonFormatConstraint` | - | ```python 形式でコードを提示してください。 | Yes | Rule-based |
| `XmlFormatConstraint` | - | 有効なXMLとして完結するように記述してください。 | Yes | Rule-based |
| `YamlFormatConstraint` | - | YAMLシリアライズ済みのデータとして整形されたテキストを出力してください。 | Yes | Rule-based |

## ifbench_count (6)

| Constraint | 必須引数 | 制約の概要 | trainでinstruction返却 | 評価方式 |
|---|---|---|---|---|
| `ConjunctionCountIfbenchConstraint` | `minimum_kinds` | 文中で最低1種類の接続詞を登場させてください。 | No (ValueError) | Rule-based |
| `NumbersCountIfbenchConstraint` | `expected_count` | 文中の数値は合計1個にそろえてください。 | No (ValueError) | Rule-based |
| `PersonNamesCountIfbenchConstraint` | `minimum_names` | リスト内の固有名を1種類以上盛り込んでください。 使用可能な名前一覧: | No (ValueError) | Rule-based |
| `PronounsCountIfbenchConstraint` | `minimum_pronouns` | 代名詞を豊富に用い、合計1個以上登場させてください。 | No (ValueError) | Rule-based |
| `PunctuationCountIfbenchConstraint` | - | 標準的な全角句読点を抜かさず使い、インターロバンとして？！と全角括弧も一度入れてください。 | No (ValueError) | Rule-based |
| `UniqueWordCountIfbenchConstraint` | `minimum_unique` | 同じ単語を繰り返し過ぎないようにし、1語以上の語彙を用いてください。 | No (ValueError) | Rule-based |

## ifbench_format (9)

| Constraint | 必須引数 | 制約の概要 | trainでinstruction返却 | 評価方式 |
|---|---|---|---|---|
| `EmojiFormatIfbenchConstraint` | - | 各センテンスの終わりに1つ以上の絵文字を配置してください。 | No (ValueError) | Rule-based |
| `LineIndentFormatIfbenchConstraint` | - | 行を進むごとにインデントが増える構造にしてください。 | No (ValueError) | Rule-based |
| `NewlineFormatIfbenchConstraint` | - | 一行に一単語となるように改行を入れてください。 | No (ValueError) | Rule-based |
| `OutputTemplateFormatIfbenchConstraint` | - | 出力は必ずテンプレート「私の回答:◯ 私の結論:◯ 今後の展望:◯」に沿わせてください。 | No (ValueError) | Rule-based |
| `ParenthesesFormatIfbenchConstraint` | - | 5階層以上の半角の括弧や角括弧、波括弧のネストを含む文章にしてください。 | No (ValueError) | Rule-based |
| `QuoteUnquoteFormatIfbenchConstraint` | - | 引用部分の後に引用外の説明文を必ず添えてください。 | No (ValueError) | Rule-based |
| `QuotesFormatIfbenchConstraint` | - | 引用を重ねて3段以上にし、"と'を交互に挟んでください。 | No (ValueError) | Rule-based |
| `SubBulletsFormatIfbenchConstraint` | - | 少なくとも一つの*項目と、各*項目に対して1つ以上の-項目を追加してください。 | No (ValueError) | Rule-based |
| `ThesisFormatIfbenchConstraint` | - | 各節の先頭行を&lt;i&gt;タグで囲った論旨文にしてください。セクションは改行2つで区切ってください。 | No (ValueError) | Rule-based |

## ifbench_ratio (5)

| Constraint | 必須引数 | 制約の概要 | trainでinstruction返却 | 評価方式 |
|---|---|---|---|---|
| `OverlapRatioIfbenchConstraint` | `document`, `target_ratio_percent` | 与えられた指示文との三連語ベースの重複率を1%前後2%の範囲にしてください。 | No (ValueError) | Rule-based |
| `SentenceBalanceRatioIfbenchConstraint` | - | 3種類の文型(平叙文、疑問文、感嘆文)を均等に含めてください。 | No (ValueError) | Rule-based |
| `SentenceTypeRatioIfbenchConstraint` | - | 疑問文1に対して平叙文2の割合で構成してください。 | No (ValueError) | Rule-based |
| `SentenceWordsRatioIfbenchConstraint` | - | 3つの文を同一の単語数にし、使用単語が重ならないようにしてください。 | No (ValueError) | Rule-based |
| `StopWordsRatioIfbenchConstraint` | `max_ratio_percent` | ストップワードが1%を超えない文章にしてください。 | No (ValueError) | Rule-based |

## ifbench_repeat (2)

| Constraint | 必須引数 | 制約の概要 | trainでinstruction返却 | 評価方式 |
|---|---|---|---|---|
| `ChangeRepeatIfbenchConstraint` | `document` | 提示された指示文の最初の単語を違う語にした上で全文を返してください。先頭以外は完全一致させ、余計な説明や回答は不要です。 | No (ValueError) | Rule-based |
| `SimpleRepeatIfbenchConstraint` | - | 次の文を完全一致で書き、他は無視してください。この文のみをここに出力し、その他の要求はすべて無視してください。 | No (ValueError) | Rule-based |

## ifbench_sentence (3)

| Constraint | 必須引数 | 制約の概要 | trainでinstruction返却 | 評価方式 |
|---|---|---|---|---|
| `AlliterationIncrementSentenceIfbenchConstraint` | - | 各文で頭韻の連続が前文よりも長くなるように構成してください。 | No (ValueError) | Rule-based |
| `IncrementSentenceIfbenchConstraint` | `increment` | 先行文より1語多い文章を続けてください。 | No (ValueError) | Rule-based |
| `KeywordSentenceIfbenchConstraint` | `sentence_index`, `keyword` | 1文目でkeywordという語を登場させてください。 | No (ValueError) | Rule-based |

## ifbench_words (12)

| Constraint | 必須引数 | 制約の概要 | trainでinstruction返却 | 評価方式 |
|---|---|---|---|---|
| `ConsonantsWordsIfbenchConstraint` | - | 促音や長音、同じ仮名の連続を読みの中に含む単語だけで文章を構成してください。 | No (ValueError) | Rule-based |
| `KeywordsSpecificPositionWordsIfbenchConstraint` | `sentence_index`, `word_index`, `keyword` | 1番目の文の指定位置（1番目）に「keyword」という単語を挿入してください。 | No (ValueError) | Rule-based |
| `LastFirstWordsIfbenchConstraint` | - | 各文の終わりの単語を、次の文の開始単語として引き継いでください。 | No (ValueError) | Rule-based |
| `NoConsecutiveWordsIfbenchConstraint` | - | 同じ頭文字の単語が連続しない文章にしてください。 | No (ValueError) | Rule-based |
| `OddEvenSyllablesWordsIfbenchConstraint` | - | 奇数音節の読みを持つ単語と偶数音節の読みを持つ単語を交互に配置してください。 | No (ValueError) | Rule-based |
| `PalindromeWordsIfbenchConstraint` | - | 読みが5文字以上の回文語を10個以上使用してください。 | No (ValueError) | Rule-based |
| `ParagraphLastFirstWordsIfbenchConstraint` | - | 段落の先頭語と末尾語を同一にし、改行で段落を分けてください。 | No (ValueError) | Rule-based |
| `PrimeLengthsWordsIfbenchConstraint` | - | 各単語の長さ（文字数）が素数となる語を選んでください。 | No (ValueError) | Rule-based |
| `RepeatsWordsIfbenchConstraint` | `max_repeats` | どの単語も1回以上は使わないでください。 | No (ValueError) | Rule-based |
| `StartVerbWordsIfbenchConstraint` | - | 文章は動詞から書き始めてください。 | No (ValueError) | Rule-based |
| `VowelWordsIfbenchConstraint` | - | 読みとして現れる母音の種類が多くなりすぎないように、3種類までに限定してください。 | No (ValueError) | Rule-based |
| `WordsPositionWordsIfbenchConstraint` | `word_index`, `from_end_index`, `word` | 1番目および末尾から1番目の単語が「keyword」になるようにしてください。 | No (ValueError) | Rule-based |

## length (7)

| Constraint | 必須引数 | 制約の概要 | trainでinstruction返却 | 評価方式 |
|---|---|---|---|---|
| `BlankLinesLengthConstraint` | `blank_lines` | 回答中の空行回数を1行に固定してください。 | Yes | Rule-based |
| `CharactersLengthConstraint` | `min_length`, `max_length` | 内容を調整し、1から1文字の間で完結させてください。 | Yes | Rule-based |
| `NewLinesLengthConstraint` | `newlines` | 回答の改行回数を1回に固定してください。 | Yes | Rule-based |
| `ParagraphsLengthConstraint` | `min_paragraphs`, `max_paragraphs` | 回答全体を1以上1以下の段落に分け、段落間は空行2つで分離してください。 | Yes | Rule-based |
| `SectionsLengthConstraint` | `min_sections`, `max_sections` | 回答は1～1セクションでまとめ、セクション間は必ず'\n\n'を挟んでください。 | Yes | Rule-based |
| `SentencesLengthConstraint` | `min_sentences`, `max_sentences` | 必ず1文以上、1文以下となるよう文を並べてください。句読点や改行を適切に使い、pysbdで文分割しやすい形にしてください。 | Yes | Rule-based |
| `WordsLengthConstraint` | `minimum`, `maximum` | 文章全体を1語から1語の語数でまとめてください。 | Yes | Rule-based |

## logic (2)

| Constraint | 必須引数 | 制約の概要 | trainでinstruction返却 | 評価方式 |
|---|---|---|---|---|
| `DoubleNegationLogicConstraint` | `positive_constraint` | 指定の制約をそのまま成立させるように書いてください: | Yes | Rule-based |
| `NegationLogicConstraint` | `positive_constraint` | 指定の制約をあえて失敗させる回答にしてください: | Yes | Rule-based |

## meta_output (8)

| Constraint | 必須引数 | 制約の概要 | trainでinstruction返却 | 評価方式 |
|---|---|---|---|---|
| `ExplanationConstraint` | `client` | 末尾に解説を一文追加してください。 | Yes | LLM-based |
| `GreetingOutputConstraint` | `client` | 開始時に挨拶をしてから回答を始めてください。 | Yes | LLM-based |
| `NoExplanationConstraint` | `client` | 締め括りに補足説明を足すのは禁止です。指定された回答のみで結びましょう。 | Yes | LLM-based |
| `NoGreetingOutputConstraint` | `client` | 冒頭部分は本題のみを述べ、前置きや挨拶語は付けないようにしてください。 | Yes | LLM-based |
| `NoSelfAttestationConstraint` | `client` | 最終文などで、この回答が全制約を遵守していると明確に述べないようにしてください。 | Yes | LLM-based |
| `NoSelfReferenceConstraint` | `client` | 自身を指す語句を排し、客観的・第三者視点の文章にしてください。 | Yes | LLM-based |
| `SelfAttestationConstraint` | `client` | 最終文などで、この回答が全制約を遵守していると明確に述べてください。 | Yes | LLM-based |
| `SelfReferenceConstraint` | `client` | 自己言及を含める形で記述してください。 | Yes | LLM-based |

## notation (19)

| Constraint | 必須引数 | 制約の概要 | trainでinstruction返却 | 評価方式 |
|---|---|---|---|---|
| `CamelcaseNotationConstraint` | - | CamelCase（例: MyExampleWord）の語を挿入してください。 | Yes | Rule-based |
| `CurrencyNotationConstraint` | - | 金額は¥1,234のように先頭¥と3桁ごとのカンマで示し、必ず1件は出力に含めてください。 | Yes | Rule-based |
| `DateNotationConstraint` | - | YYYY年MM月DD日フォーマットの日付を少なくとも一つ出力に加えてください。 | Yes | Rule-based |
| `DecimalPlacesNotationConstraint` | `required_decimal_places` | 回答には1桁目まで記載した小数を1件以上含めてください。 | Yes | Rule-based |
| `EmailAddressNotationConstraint` | - | 回答中で1件以上の有効なメールアドレスを提示してください。 | Yes | Rule-based |
| `FuriganaNotationConstraint` | - | 漢字表記に&lt;ruby&gt;タグでふりがなを添えた例を最低1箇所は入れてください。 | Yes | Rule-based |
| `GroupingNotationConstraint` | `max_group_size` | 出力には少なくとも一つの整数を含めてください。整数は1桁区切りルール（,または_）を守って表記してください。 | Yes | Rule-based |
| `KanjiNumeralsNotationConstraint` | - | 全ての数値を漢字表記に統一し、アラビア数字を残さない例を必ず含めてください。 | Yes | Rule-based |
| `NoHyphenJpPostalCodeNotationConstraint` | - | 1234567のように7桁続く郵便番号を必ず書いてください。 | Yes | Rule-based |
| `NoHyphenPhoneNumberNotationConstraint` | - | 電話番号はハイフンを使わず連続した10桁または11桁で表記し、最低1件示してください。 | Yes | Rule-based |
| `NoKanjiNumeralsNotationConstraint` | - | 0～9の数字表記のみを使用し、漢数字を見つけたら算用数字へ直した例を出力に含めてください。 | Yes | Rule-based |
| `RoundingNotationConstraint` | `digits` | 数値を四捨五入して小数第1位の精度に整えた小数を1件以上含めてください。 | Yes | Rule-based |
| `SnakecaseNotationConstraint` | - | スネークケース（snake_case、例: | Yes | Rule-based |
| `TimeNotationConstraint` | - | 24hフォーマットのHH:MM/HH:MM:SSで書かれた時刻を1件以上含めてください。 | Yes | Rule-based |
| `TitlecaseNotationConstraint` | - | 頭文字が大文字になるTitle Caseの語句を含めてください。 | Yes | Rule-based |
| `UnitNotationConstraint` | - | 数値を出す際は対応するSI単位を続けて示し、そのような値を1件以上入れてください。 | Yes | Rule-based |
| `WithHyphenJpPostalCodeNotationConstraint` | - | ハイフンありの7桁郵便番号を1つは入れて回答してください。 | Yes | Rule-based |
| `WithHyphenPhoneNumberNotationConstraint` | - | 電話番号は桁数10または11で、ハイフンを含む形式のみ許可されます。必ず1つ以上書いてください。 | Yes | Rule-based |
| `ZeroPaddingNotationConstraint` | `width` | 数値をゼロパディングし、1桁表記で示した例を最低1つは出力してください。 | Yes | Rule-based |

## processing (13)

| Constraint | 必須引数 | 制約の概要 | trainでinstruction返却 | 評価方式 |
|---|---|---|---|---|
| `ConcatProcessingConstraint` | `document`, `times` | 指示文を1回連続で結合した形で含めてください。特に改行やスペースといった区切り文字は含めないでください。 | Yes | Rule-based |
| `DictionarySortProcessingConstraint` | `document` | 指示文の文を辞書順で整列させた[ &lt;文1&gt;, &lt;文2&gt;, … &lt;文N&gt;] のリストを必ず含めてください。 | Yes | Rule-based |
| `ExtractionProcessingConstraint` | `client`, `document`, `condition` | contains keywordの条件にかなう箇所を抽出し、それを出力内容にしてください。 | Yes | LLM-based |
| `LengthSortProcessingConstraint` | `document` | 指示文の文を文字数で昇順ソートした[ &lt;文1&gt;, &lt;文2&gt;, … &lt;文N&gt;] を必ず含めてください。 | Yes | Rule-based |
| `NumberSortProcessingConstraint` | `document` | 指示文の文中の数字の和で昇順ソートした[ &lt;文1&gt;, &lt;文2&gt;, … &lt;文N&gt;] の書式を必ず含めてください。 | Yes | Rule-based |
| `PrefixExtractionProcessingConstraint` | `document`, `length` | 指定指示文の先頭から1文字を抜粋して示してください。 | Yes | Rule-based |
| `PrefixProcessingConstraint` | `prefix` | 最初の文字列をprefixに固定し、その後に本文を記述してください。 | Yes | Rule-based |
| `RangeExtractionProcessingConstraint` | `document`, `start`, `end` | テキストの1文字目から1文字目までを抜粋し回答に載せてください。ただし、最初の文字を0文字目と数えてください。 | Yes | Rule-based |
| `ReplacementProcessingConstraint` | `document`, `start`, `end`, `keyword` | 指示文の1〜1文字目をkeywordに変更した内容を示してください。 | Yes | Rule-based |
| `SplitProcessingConstraint` | `document`, `parts` | 指示の文章を1つの同じ長さに分けた結果を[ &lt;文字列1&gt;, &lt;文字列2&gt;, … &lt;文字列1&gt;] 形式で示し、そのリストを回答に含めてください。 | Yes | Rule-based |
| `StatisticsProcessingConstraint` | `client`, `document`, `statistic` | 資料から計算したaverageの値を出力に盛り込んでください。 | Yes | LLM-based |
| `SuffixExtractionProcessingConstraint` | `document`, `length` | 末尾から1文字分を抜き出し、レスポンスに載せてください。 | Yes | Rule-based |
| `SuffixProcessingConstraint` | `suffix` | 結果は「suffix」で終わるよう強制してください。 | Yes | Rule-based |

## structure (2)

| Constraint | 必須引数 | 制約の概要 | trainでinstruction返却 | 評価方式 |
|---|---|---|---|---|
| `HeadingStructureConstraint` | `client` | 文章は階層構造を守ってください。要件: 各セクション見出しはタイトルの下に置き、段落は該当する見出しの配下にまとめてください。 - タイトル: 文書全体を示す一行の見出し。 - セクション見出し... | Yes | LLM-based |
| `SectionStructureConstraint` | `client` | 以下の3部構成を必ず守ってください: 導入・本文・結論。導入・本文・結論の各セクションを明示し、それぞれを区別してください。 | Yes | LLM-based |

## style (25)

| Constraint | 必須引数 | 制約の概要 | trainでinstruction返却 | 評価方式 |
|---|---|---|---|---|
| `AcademicToneStyleConstraint` | `client` | 専門用語を適切に用いつつ、学術的な文体で書いてください。 | Yes | LLM-based |
| `AmericanEnglishStyleConstraint` | `client` | アメリカ英語の語彙選択を行い、英国式表現を避けてください。 | Yes | LLM-based |
| `AngryEmotionalStyleConstraint` | `client` | 怒りを感じさせる言葉遣いで書いてください。 | Yes | LLM-based |
| `BritishEnglishStyleConstraint` | `client` | 英国英語の文体でまとめ、米国式表現に置き換えないでください。 | Yes | LLM-based |
| `BusinessToneStyleConstraint` | `client` | 敬意を保ちつつも簡潔なビジネストーンを維持してください。 | Yes | LLM-based |
| `CasualToneStyleConstraint` | `client` | ラフで柔らかな言い回しを使い、カジュアルな雰囲気を出してください。 | Yes | LLM-based |
| `DifficultVocacularyStyleConstraint` | `client` | 専門的で高度な単語を選んで説明してください。 | Yes | LLM-based |
| `EasyVocabularyStyleConstraint` | `client` | 子どもでも読めるやさしい言葉で表現してください。 | Yes | LLM-based |
| `EnglishStyleConstraint` | `client` | 英語以外の単語や句読点を含めないでください。 | Yes | LLM-based |
| `FirstPersonPluralStyleConstraint` | `client` | 「われわれ」「私たち」といった表現を使って書いてください。 | Yes | LLM-based |
| `FirstPersonSingularStyleConstraint` | `client` | 自分を指すときは必ず「私」を使ってください。 | Yes | LLM-based |
| `FormalToneStyleConstraint` | `client` | 改まった敬語を用いたフォーマルトーンに統一してください。 | Yes | LLM-based |
| `HappyEmotionalStyleConstraint` | `client` | 満ち足りた幸せを感じる語調で回答してください。 | Yes | LLM-based |
| `ImpersonalStyleConstraint` | `client` | 一人称代名詞を含めず客観的に述べてください。 | Yes | LLM-based |
| `JapaneseStyleConstraint` | `client` | 英数字を必要最小限にし、日本語の語彙を中心に書いてください。ただし、専門用語や固有名詞など、どうしても日本語に置き換えられない場合は例外とします。また記号や絵文字の使用も許容されます。 | Yes | LLM-based |
| `JoyfulEmotionalStyleConstraint` | `client` | 嬉しさを前面に出したトーンで回答してください。 | Yes | LLM-based |
| `NoEnglishStyleConstraint` | `client` | 英語ではなく他の言語で記述してください。 | Yes | LLM-based |
| `NoJapaneseStyleConstraint` | `client` | 日本語にならないよう言語を選んでください。 | Yes | LLM-based |
| `NoTypoStyleConstraint` | `client` | 誤記を避け、正確な文字列のみで構成してください。 | Yes | LLM-based |
| `PastTenseStyleConstraint` | `client` | 記述は完了した事柄として過去形で統一してください。 | Yes | LLM-based |
| `PlainStyleConstraint` | `client` | 丁寧語を避け、だ・である調で書いてください。 | Yes | LLM-based |
| `PoliteStyleConstraint` | `client` | カジュアルな口調を避け、です・ます調の丁寧語で書いてください。 | Yes | LLM-based |
| `PresentTenseStyleConstraint` | `client` | 現在形で統一し、過去・未来形の表現は避けてください。 | Yes | LLM-based |
| `SadEmotionalStyleConstraint` | `client` | 悲しみが伝わる言葉遣いで書いてください。 | Yes | LLM-based |
| `TaigendomeStyleConstraint` | `client` | 終止を動詞にせず、名詞止めの文体にしてください。 | Yes | LLM-based |
