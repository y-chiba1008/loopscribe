# AGENTS.md

コーディングエージェント向けの作業ルールです。人間向けの概要は `README.md`、
要件と設計の詳細は `docs/requirements-and-design.md` を参照してください。
**実装を始める前に、必ず設計資料を最後まで読んでください。**

## プロジェクト概要

`loopscribe`: Windowsのシステム音声(WASAPIループバック)を、ローカルで日本語に文字起こしするCLI。
faster-whisper(CPU / int8)+ kotoba-whisper。GPUなし・従量課金なし。
`uv tool install` で配布するPythonパッケージ。現在は設計済み・未実装の状態。

## 守るべき決定事項(変更する場合は、ユーザーに確認すること)

1. **実行環境はWindowsのみ**。音声取得は WASAPIループバック(`pyaudiowpatch`)。開発はWSL2で行う。
2. **Python 3.13以降**。
3. **CLIオプションは最小限**: `start` の `--device` と `-o/--output-dir` だけ。
   モデル名・チャンク長などの設定値は、すべて `src/loopscribe/constants.py` の定数で定義する(オプションを増やさない)。
4. **サブコマンドは3つだけ**: `start` / `download-model` / `devices`。
5. **文字起こしの遅れは無制限に許容する。未処理の音声は破棄しない**(メモリに保持する)。
   バッファ上限による破棄や、欠落マーカーの出力は、初期版では実装しない。
6. **警告のみの方針**: 実行中に続行可能な問題は警告を出して継続する。
   ただし起動時の致命的な状況(デバイス未検出、モデル取得不可)は、エラーメッセージを出して終了コード1で終了する。
   実行中にデバイスの切断などで取得ストリームが失敗した場合は、警告を出して録音を停止し、
   バックログを処理してから終了コード1で終了する(確定済み)。
7. **進捗表示**: `処理済み時間 / 録音時間` をrichのプログレスバーでリアルタイム表示する。
8. **完全ローカル処理**。ネットワーク通信はモデルのダウンロード時のみ。
9. 初期版の対象は**システム音声のみ**(マイク入力は将来対応)。

## コマンド

```bash
uv sync                               # 依存を取得(WSLでも可)
uv run pytest                         # 通常のテスト(slowを除く)
uv run pytest -m slow                 # 実モデルを使うテスト
uv run ruff check . && uv run ruff format .
uv run scripts/bench_models.py sample.wav --reference sample.txt   # モデル検証
```

## ディレクトリ構成(目標)

```
src/loopscribe/
├── cli.py            # Typerアプリ。start / download-model / devices
├── constants.py      # すべての設定値(CLIオプション化しない)
├── errors.py         # 起動時の致命的エラーなどの例外
├── model.py          # モデルの取得(download_model、未取得判定)
├── audio/
│   ├── source.py     # AudioSource プロトコル
│   ├── wasapi.py     # WasapiLoopbackSource(Windows専用。importは遅延させる)
│   ├── wavfile.py    # WavFileSource(開発・テスト用。WSLで動く)
│   └── convert.py    # モノラル化 + 16kHzへのリサンプル(soxr.ResampleStream)
├── buffer.py         # AudioBuffer(int16 / 16kHz / モノラル。スレッドセーフ)
├── chunker.py        # チャンク区間と担当境界の計算
├── transcriber.py    # Transcriber プロトコル + faster-whisper実装
├── pipeline.py       # スレッド構成、チャンク処理、セグメントの担当割当て
├── output.py         # コンソール出力とtxt出力(Sink)
└── progress.py       # プログレスバー(rich)
scripts/bench_models.py   # モデル検証用(単体で動く。PEP 723)
tests/
```

## 設計上のルール

- **音声取得とコアを分離する**。`AudioSource` の後ろにWindows固有の処理を閉じ込め、それ以外はOS非依存にする。
  `pyaudiowpatch` のimportは `wasapi.py` の中だけで行い、モジュールのimportも遅延させる(WSLで `uv sync` / テストが通るように)。
- **設定値は引数で受け取り、既定値として定数を使う**。関数やクラスが `constants` をグローバルに直接参照すると、
  テストでチャンク長を変えられない。
- 表示は **richの単一のConsole** に統一する(ログ・警告・本文・プログレスバーを同じConsole経由にして、表示を崩さない)。
  標準出力がTTYでない場合は、プログレスバーを無効にして、本文だけを出す。
- 起動時に `sys.stdout.reconfigure(encoding="utf-8")` を行う。ファイルもUTF-8で書く。
- txtは1セグメントごとにflushする(クラッシュしても失われないように)。
- **Ctrl+Cの扱い**: メインスレッドはイベント待ちのループ(`Event.wait(timeout)` を短い間隔で繰り返す)にし、
  文字起こしは別スレッドで実行する(ネイティブの推論呼び出し中はPythonのシグナル処理が遅れるため)。
  1回目: 録音を止めて、バックログを処理してから終了。2回目: sinkをflushして `os._exit(130)`。
- 型ヒントを付ける。`ruff` の指摘は残さない。コメントとdocstringは日本語でよい。

## テスト

- **サンプルwavでの文字起こしテスト**を軸にする。`WavFileSource` で本番と同じ経路を通す。
- 通常のテストは、偽の `Transcriber` を使って高速に回す(チャンク境界、担当割当て、出力形式、進捗の計算)。
- 実モデルを使うテストには `@pytest.mark.slow` を付ける(既定ではスキップ)。結果は完全一致ではなくCERで判定する。
- テスト用のwavは数秒の短いものにして `tests/data/` に置く。**著作権・プライバシーに問題のない音声**だけを使う
  (音声ファイルは、ユーザーが用意する想定)。

## 実装の推奨順序

1. `constants.py`、`errors.py`、`chunker.py`、`buffer.py`、出力の整形 + 単体テスト
2. `WavFileSource`、`convert.py`、`pipeline.py`(偽のTranscriberで通す)
3. faster-whisper実装の `Transcriber` と `model.py` + slowテスト
4. `progress.py` と `cli.py`。WSLでの手動確認用に、同じパイプラインでwavを流す開発用スクリプト
   `scripts/run_wav.py` を作る(**CLIオプションにはしない**)
5. `wasapi.py` と `devices` サブコマンド(Windows実機が必要)
6. Windows実機での検証(下記)

## Windows実機で確認が必要なこと

- 無音のときにループバックのコールバックが来ない場合の挙動(タイムスタンプが実時間とずれる可能性)
- 出力デバイスの切り替え・切断時の挙動
- Ctrl+Cの2段階動作、長時間(数時間)の実行時のメモリ
- `uv tool install git+...` でのインストールと、Python 3.13向けwheel(`pyaudiowpatch`、`soxr`)の有無

## やらないこと

- 上記の「守るべき決定事項」に反する変更(オプションの追加、バックログの破棄など)を、確認なしに入れること
- 実在する個人の音声や、権利のある音声をリポジトリに含めること
- 設計資料に書かれていない大きな依存の追加(必要なら理由を添えてユーザーに確認する)

## 未確定事項

設計資料の「リスク・未確定事項」を参照してください。決まった内容は、設計資料の決定ログに追記してください。

## その他
技術的な知見は /home/chiba/workspaces/knowledge-base を参照すること。
まず INDEX.md を読み、rg でキーワード検索する。詳細は同ディレクトリの AGENTS.md に従う。
