"""起動時の致命的エラーなど、アプリ固有の例外。"""


class LoopscribeError(Exception):
    """loopscribe の基底例外。"""


class FatalStartupError(LoopscribeError):
    """起動時に続行できない状況。終了コード 1 で終了する。"""


class DeviceNotFoundError(FatalStartupError):
    """対象の出力デバイスが見つからない。"""


class ModelUnavailableError(FatalStartupError):
    """モデルが未取得で、取得にも失敗した。"""


class InvalidOutputNameError(FatalStartupError):
    """`-n/--output-name` の値が、ファイル名として使えない。"""


class ConfigError(LoopscribeError):
    """設定ファイルの読み書きに失敗した、または項目名・値が不正。

    起動時は警告を出して無視する。`config` で更新するときは、エラーを出して終了コード 1 にする。
    """
