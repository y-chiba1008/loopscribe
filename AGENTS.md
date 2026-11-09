# AGENTS.md

コーディングエージェント向けの作業ルールです。人間向けの概要は `README.md`、
要件と設計の詳細は `docs/requirements-and-design.md` を参照してください。
**実装を始める前に、必ず設計資料を最後まで読んでください。**

## プロジェクト概要

`loopscribe`: Windowsのシステム音声(WASAPIループバック)を、ローカルで日本語に文字起こしするCLI。
faster-whisper(CPU / int8)+ Whisper `small`(`--model kotoba` で kotoba-whisper に切り替え可)。GPUなし・従量課金なし。
`uv tool install` で配布するPythonパッケージ。

## 守るべき決定事項(変更する場合は、ユーザーに確認すること)

1. **実行環境はWindowsのみ**。音声取得は WASAPIループバック(`pyaudiowpatch`)。開発はWSL2で行う。
2. **Python 3.13以降**。
3. **文字起こしの遅れは無制限に許容する。未処理の音声は破棄しない**(メモリに保持する)。
   バッファ上限による破棄や、欠落マーカーの出力は、初期版では実装しない。
4. **警告のみの方針**: 実行中に続行可能な問題は警告を出して継続する。
   ただし起動時の致命的な状況(デバイス未検出、モデル取得不可)は、エラーメッセージを出して終了コード1で終了する。
   実行中にデバイスの切断などで取得ストリームが失敗した場合は、警告を出して録音を停止し、
   バックログを処理してから終了コード1で終了する(確定済み)。
5. **進捗表示**: `処理済み時間 / 録音時間` をrichのプログレスバーでリアルタイム表示する。
6. **完全ローカル処理**。ネットワーク通信はモデルのダウンロード時のみ。
7. 音声の取得は**システム音声のみ**(マイク入力は将来対応)。ファイルの文字起こしは `transcribe` で、16bit PCM の wav だけを対象にする。

## コマンド

```bash
uv sync                               # 依存を取得(WSLでも可)
uv run pytest                         # 通常のテスト(slowを除く)
uv run pytest -m slow                 # 実モデルを使うテスト
uv run ruff check . && uv run ruff format .
uv run scripts/prepare_bench_audio.py  # 検証用wavと正解テキストを scripts/data/ に作る
uv run scripts/bench_models.py scripts/data/sample.wav --reference scripts/data/sample.txt
```

## ディレクトリ構成

```
src/loopscribe/
├── cli.py            # Typerアプリ。start / record / transcribe / download-model / devices / config
├── config.py         # 設定ファイル(TOML)の読み書き。項目は default-model だけ(設計資料 §5.2)
├── constants.py      # モデルに依存しない設定値(CLIオプション化しない)
├── errors.py         # 起動時の致命的エラーなどの例外
├── model.py          # モデルの取得(download_model、未取得判定。repo id を受け取る汎用の処理)
├── models/
│   ├── __init__.py   # モデルの登録(MODELS、DEFAULT_MODEL)
│   ├── base.py       # ModelSpec(モデルごとの取得・修正・読み込みの共通インターフェース)
│   ├── small.py      # small の推論パラメータと推論(WhisperModel.transcribe)
│   └── kotoba.py     # kotoba-whisper の推論パラメータと推論(BatchedInferencePipeline)、設定の修正
├── audio/
│   ├── source.py     # AudioSource プロトコル
│   ├── wasapi.py     # WasapiLoopbackSource(Windows専用。importは遅延させる)
│   ├── wavfile.py    # WavFileSource(transcribe とテストで使う。WSLで動く)
│   └── convert.py    # モノラル化 + 16kHzへのリサンプル(soxr.ResampleStream)
├── capture.py        # AudioCapture(取得と変換のスレッド。start / record / transcribe で共有)
├── buffer.py         # AudioBuffer(int16 / 16kHz / モノラル。スレッドセーフ)
├── chunker.py        # チャンク区間と担当境界の計算
├── transcriber.py    # Transcriber プロトコル、Segment / Word、faster-whisperの結果の変換
├── pipeline.py       # スレッド構成、チャンク処理、セグメントの担当割当て
├── recorder.py       # record の録音(WavWriter で wav に追記。Recorder)
├── output.py         # コンソール出力とtxt出力(Sink)、出力名の検査と連番付け
└── progress.py       # プログレスバー(rich)。record の録音時間・ファイルサイズの表示
scripts/bench_models.py          # モデル検証用(単体で動く。PEP 723)
scripts/prepare_bench_audio.py   # 検証用wavの作成(ReazonSpeech。生成物は scripts/data/ で gitignore)
tests/
```

## 設計上のルール

- **音声取得とコアを分離する**。`AudioSource` の後ろにWindows固有の処理を閉じ込め、それ以外はOS非依存にする。
  `pyaudiowpatch` のimportは `wasapi.py` の中だけで行い、モジュールのimportも遅延させる(WSLで `uv sync` / テストが通るように)。
- チャンク長などの設定値は `constants.py`(モデルに依存しない値)で定義する。
  設定ファイル(TOML)の優先順位は CLIオプション > 設定ファイル > 組み込みの既定。
- **`-n/--output-name`** は拡張子なしの名前として受け取り、拡張子はコマンドが付ける(末尾の `.txt` / `.wav` は取り除く)。
  同名のファイルがあれば `_2`、`_3` … を付けて警告する。パスの区切り文字を含む名前はエラー(終了コード1)。
  名前の検査と同名のファイルの確認(連番と警告)は、モデルの読み込みやデバイスを開く前に行う。
  出力ファイルは上書きしない(新規作成のモードで開く)。
- **推論パラメータはモデルごとに固定値で持つ**(`models/<モデル>.py`)。ユーザーが調整しきれないので、CLIオプションにしない。
  モデルを追加するときは `ModelSpec` を実装し、`models/__init__.py` の `MODELS` に登録する。
- **設定値は引数で受け取り、既定値として定数を使う**。関数やクラスが `constants` をグローバルに直接参照すると、
  テストでチャンク長を変えられない。
- 表示は **richの単一のConsole** に統一する(ログ・警告・本文・プログレスバーを同じConsole経由にして、表示を崩さない)。
  標準出力がTTYでない場合は、プログレスバーを無効にして、本文だけを出す。
- 起動時に `sys.stdout.reconfigure(encoding="utf-8")` を行う。ファイルもUTF-8で書く。
- txtは1セグメントごとにflushする(クラッシュしても失われないように)。
- **Ctrl+Cの扱い**: メインスレッドはイベント待ちのループ(`Event.wait(timeout)` を短い間隔で繰り返す)にし、
  文字起こしは別スレッドで実行する(ネイティブの推論呼び出し中はPythonのシグナル処理が遅れるため)。
  1回目: 録音を止めて、バックログを処理してから終了。2回目: sinkをflushして `os._exit(130)`。
- wavの文字起こしは `transcribe` サブコマンドで行う(WSLでも動く)。WSLでパイプライン全体を手動で試すときも `transcribe` を使う。
 ファイル入力だけ、先読みに上限を設ける(`read_ahead_seconds`)。ライブ入力では使わない。
- 型ヒントを付ける。`ruff` の指摘は残さない。コメントとdocstringは日本語でよい。

## テスト

- **サンプルwavでの文字起こしテスト**を軸にする。`WavFileSource` で本番と同じ経路を通す。
- 通常のテストは、偽の `Transcriber` を使って高速に回す(チャンク境界、担当割当て、出力形式、進捗の計算)。
- 実モデルを使うテストには `@pytest.mark.slow` を付ける(既定ではスキップ)。結果は完全一致ではなくCERで判定する。
- テスト用のwavは数秒の短いものにして `tests/data/` に置く。**著作権・プライバシーに問題のない音声**だけを使う
  (音声ファイルは、ユーザーが用意する想定)。

## やらないこと

- 上記の「守るべき決定事項」に反する変更(バックログの破棄など)を、確認なしに入れること
- 実在する個人の音声や、権利のある音声をリポジトリに含めること
- 設計資料に書かれていない大きな依存の追加(必要なら理由を添えてユーザーに確認する)

## 未確定事項

未確定事項・検証待ちの項目・既知の不具合は、GitHubのissueで管理します(`gh issue list`)。
決まった内容は、設計資料の決定ログに追記してください。

## プルリクエスト

- Test plan にWindowsでの操作(実機での確認)が含まれる場合は、そのブランチを指定してインストールするコマンドをPRに載せる。
 ブランチ名の `#` はURLのフラグメントと解釈されるので `%23` にエンコードする。

 ```powershell
 uv tool install --force "git+https://github.com/y-chiba1008/loopscribe.git@feature/%2334_add_output_name_option"
 ```

## その他
技術的な知見は /home/chiba/workspaces/knowledge-base を参照すること。
まず INDEX.md を読み、rg でキーワード検索する。詳細は同ディレクトリの AGENTS.md に従う。
