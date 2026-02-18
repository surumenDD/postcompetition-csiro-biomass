# timing.py
import math
import os
import sys
import time
from contextlib import contextmanager

import psutil


@contextmanager
def measure_time_and_memory(title):
    """
    コードブロックの実行時間とRSSメモリ使用量を計測するコンテキストマネージャ。

    処理内容:
        1. 開始時刻とRSSメモリを取得
        2. withブロック内の処理を実行
        3. 終了時刻とRSSメモリを再取得
        4. 経過時間とメモリ増減を計算
        5. 計測結果をstderrへ出力

    出力形式:
        [現在GB(+/-増減GB):経過秒] title

    用途:
        重い前処理・学習・推論などの時間およびメモリ変化の可視化。

    使い方例:
        >>> with measure_time_and_memory("wait"):
                time.sleep(2.0)

    @contextmanagerやyieldの使い方の参考:
        https://qiita.com/shigezou46/items/75862fa52c478d614054
        
    """
    t0 = time.time()
    p = psutil.Process(os.getpid())
    m0 = p.memory_info().rss / 2.0**30
    yield
    m1 = p.memory_info().rss / 2.0**30
    delta = m1 - m0
    sign = "+" if delta >= 0 else "-"
    delta = math.fabs(delta)
    print(
        f"[{m1:.1f}GB({sign}{delta:.1f}GB):{time.time() - t0:.1f}sec] {title} ",
        file=sys.stderr,
    )


@contextmanager
def timer(name):
    """
    Examples:
        >>> with timer("wait"):
                time.sleep(2.0)
    """
    t0 = time.time()
    yield
    elapsed_time = time.time() - t0
    print(f"[{name}] done in {elapsed_time:.1f} s")