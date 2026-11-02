# loopscribe

Windowsで再生中の音声(システム音声)を、**ローカルで日本語に文字起こし**するCLIツールです。
GPU不要・従量課金なし。音声は外部に送信されません(通信はモデルのダウンロード時のみ)。

> **現在の状態: 設計済み・未実装。**
> 要件と設計は [`docs/requirements-and-design.md`](docs/requirements-and-design.md) にまとまっています。
> 実装を始める前に、まず「モデル検証」(下記)を行ってください。

## 特徴(予定)

- WASAPIループバックで、再生中の音声をそのまま取得(Windowsのみ)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) + kotoba-whisper で日本語を文字起こし(CPU / int8)
- 約30秒ごとにまとめて処理(リアルタイム性より、低スペックPCでの安定動作を優先)
- 結果をコンソールとtxtファイル(タイムスタンプ付き)に出力
- 進捗(処理済み時間 / 録音時間)をプログレスバーで表示
- 文字起こしが遅れても音声は破棄しない(メモリに保持して順に処理)

## 動作環境

| 項目 | 内容 |
|---|---|
| OS | Windows(実行時)。開発はWSL2でも可 |
| Python | 3.13以降 |
| 想定PC | Core i5-8350U 相当(4コア8スレッド)、RAM 8GB、GPUなし |

## インストール(実装後)

```powershell
# Windows側で実行
uv tool install git+https://github.com/<account>/loopscribe
loopscribe download-model   # 省略可。未取得なら start 時に自動で取得される
```

## 使い方(予定)

```powershell
loopscribe devices                       # ループバック対象の出力デバイス一覧
loopscribe start                         # 既定の出力デバイスの音声を文字起こし(Ctrl+Cで停止)
loopscribe start --device "Speakers"     # デバイスを指定(番号または名前の一部)
loopscribe start -o D:\notes             # txtの保存先(既定: カレントディレクトリ)
```

- Ctrl+C を1回押すと録音を止め、未処理の音声を最後まで処理してから終了します。
  もう一度押すと、その場で中断します。
- 出力例:

  ```
  [00:00:12] こんにちは、今日は…
  [00:00:41] 次の話題ですが…
  ```

設定値(モデル、チャンク長など)はCLIオプションではなく、`src/loopscribe/constants.py` で定義します。

## 開発

```bash
uv sync                  # 依存を取得(WSLでも可。Windows専用のpyaudiowpatchは自動でスキップされる)
uv run pytest            # 通常のテスト(実モデルを使う slow は除外)
uv run pytest -m slow    # 実モデルを使うテスト(重い)
uv run ruff check . && uv run ruff format --check .
```

開発は **WSL2、音声取得の最終確認だけWindows** という進め方を想定しています。

- コア部分(チャンク処理・文字起こし・出力・CLI)は、サンプルwavを入力にしてWSLで開発・テストできます。
- ループバック取得は、Windows実機で確認が必要です。GitHubにpushし、Windows側で
  `uv tool install --reinstall git+https://github.com/<account>/loopscribe` を実行して確認します。
- リポジトリは WSL側(`~/`配下)に置いてください。`/mnt/c` 配下はファイルI/Oが遅く、仮想環境も壊れやすいです。
- WSL2のメモリ上限(既定は物理メモリの約半分)が足りない場合は、`%UserProfile%\.wslconfig` で調整します。

## モデル検証(最初のタスク)

採用するモデルを決める前に、サンプル音声で処理速度(RTF)・メモリ・精度(CER)を測ります。
`scripts/bench_models.py` は、アプリ本体と同じ条件(30秒チャンク + 3秒の重なり)で処理します。

```bash
# サンプルを作る(ReazonSpeech。出力は scripts/data/ で、Git には含めない)
uv run scripts/prepare_bench_audio.py

# 段階1: モデルだけを比較(既定: kotoba-whisper と small)
uv run scripts/bench_models.py scripts/data/sample.wav --reference scripts/data/sample.txt

# 段階2: 有力なモデルに絞って、beam_size・スレッド数・VADを振る
uv run scripts/bench_models.py scripts/data/sample.wav --reference scripts/data/sample.txt \
    --models kotoba-tech/kotoba-whisper-v2.0-faster \
    --beam-sizes 1 5 --threads 4 8 --vad on off --json result.json
```

- サンプルは、ReazonSpeech の先頭から作った 1〜3分程度・16bit PCM の wav です。
  生成した音声とテキストはローカル専用です。リポジトリには含めません。
- 取得の前に、ブラウザで [ReazonSpeech の利用条件](https://huggingface.co/datasets/reazon-research/reazonspeech) へ同意してください。
  未ログインなら、スクリプトがブラウザログインを始めます。
- 計測中は他の重い処理を止めてください(ノートPCは熱でクロックが下がりやすいです)。
- RTFが1.0を超えると、録音1時間あたり `(RTF − 1) × 60` 分ずつ処理が遅れます。判断基準は設計資料の「モデル検証」を参照してください。

## ドキュメント

- [`docs/requirements-and-design.md`](docs/requirements-and-design.md): 要件・設計資料(実装担当への引継ぎ)
- [`AGENTS.md`](AGENTS.md): コーディングエージェント向けの作業ルール
