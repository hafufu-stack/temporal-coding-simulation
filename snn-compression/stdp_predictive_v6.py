"""
STDP予測圧縮 v6 - ハイブリッド最適化版
========================================

最終形態:
1. 複数の圧縮手法を試して最小を選択
2. デルタ符号化（差分圧縮）
3. XOR圧縮（類似データに強い）
4. マルチパス予測

Author: ろーる (cell_activation)
"""

import numpy as np
import time
import struct
import zlib
from typing import List, Tuple
from collections import Counter


def delta_encode(data: bytes) -> bytes:
    """デルタ符号化: 差分を記録"""
    if not data:
        return b''
    result = bytearray([data[0]])
    for i in range(1, len(data)):
        diff = (data[i] - data[i-1]) % 256
        result.append(diff)
    return bytes(result)


def delta_decode(data: bytes) -> bytes:
    """デルタ復号"""
    if not data:
        return b''
    result = bytearray([data[0]])
    for i in range(1, len(data)):
        val = (result[i-1] + data[i]) % 256
        result.append(val)
    return bytes(result)


def xor_encode(data: bytes) -> bytes:
    """XOR符号化: 直前バイトとXOR"""
    if not data:
        return b''
    result = bytearray([data[0]])
    for i in range(1, len(data)):
        result.append(data[i] ^ data[i-1])
    return bytes(result)


def xor_decode(data: bytes) -> bytes:
    """XOR復号"""
    if not data:
        return b''
    result = bytearray([data[0]])
    for i in range(1, len(data)):
        result.append(data[i] ^ result[i-1])
    return bytes(result)


class MarkovPredictor:
    """1次マルコフ予測器"""
    
    def __init__(self):
        self.transitions = {}  # byte -> Counter
    
    def reset(self):
        self.transitions = {}
    
    def predict(self, context: int) -> int:
        if context in self.transitions and self.transitions[context]:
            return self.transitions[context].most_common(1)[0][0]
        return context
    
    def train(self, prev: int, curr: int):
        if prev not in self.transitions:
            self.transitions[prev] = Counter()
        self.transitions[prev][curr] += 1


class STDPPredictiveCodecV6:
    """STDP予測圧縮コーデック v6 - 最終形態"""
    
    MAGIC = b'STD6'
    
    METHODS = {
        0: "raw",
        1: "delta",
        2: "xor",
        3: "markov",
        4: "delta+markov",
    }
    
    def __init__(self):
        self.predictor = MarkovPredictor()
    
    def compress(self, data: bytes, verbose: bool = True) -> bytes:
        if len(data) == 0:
            return b''
        
        if verbose:
            print(f"入力: {len(data)} bytes")
        
        start_time = time.time()
        
        # 各圧縮手法を試す
        candidates = []
        
        # 方法0: 生データ
        raw_comp = zlib.compress(data, 9)
        candidates.append((0, raw_comp, "raw"))
        
        # 方法1: デルタ符号化
        delta_data = delta_encode(data)
        delta_comp = zlib.compress(delta_data, 9)
        candidates.append((1, delta_comp, "delta"))
        
        # 方法2: XOR符号化
        xor_data = xor_encode(data)
        xor_comp = zlib.compress(xor_data, 9)
        candidates.append((2, xor_comp, "xor"))
        
        # 方法3: マルコフ予測残差
        self.predictor.reset()
        residuals = []
        last = 0
        for b in data:
            pred = self.predictor.predict(last)
            res = (b - pred) % 256
            residuals.append(res)
            self.predictor.train(last, b)
            last = b
        markov_comp = zlib.compress(bytes(residuals), 9)
        candidates.append((3, markov_comp, "markov"))
        
        # 方法4: デルタ + マルコフ
        delta_data = delta_encode(data)
        self.predictor.reset()
        residuals = []
        last = 0
        for b in delta_data:
            pred = self.predictor.predict(last)
            res = (b - pred) % 256
            residuals.append(res)
            self.predictor.train(last, b)
            last = b
        delta_markov_comp = zlib.compress(bytes(residuals), 9)
        candidates.append((4, delta_markov_comp, "delta+markov"))
        
        # 最小を選択
        best_method, best_data, best_name = min(candidates, key=lambda x: len(x[1]))
        
        # パック
        compressed = self.MAGIC + struct.pack('<BI', best_method, len(data)) + best_data
        
        if verbose:
            ratio = len(compressed) / len(data) * 100
            print(f"圧縮: {len(compressed)} bytes ({ratio:.1f}%) [{best_name}]")
            print(f"時間: {time.time() - start_time:.3f}秒")
        
        return compressed
    
    def decompress(self, compressed: bytes, verbose: bool = True) -> bytes:
        if len(compressed) == 0:
            return b''
        
        if verbose:
            print(f"圧縮データ: {len(compressed)} bytes")
        
        start_time = time.time()
        
        if compressed[:4] != self.MAGIC:
            raise ValueError("Invalid format")
        
        method, orig_len = struct.unpack('<BI', compressed[4:9])
        payload = zlib.decompress(compressed[9:])
        
        if method == 0:  # raw
            result = payload
        elif method == 1:  # delta
            result = delta_decode(payload)
        elif method == 2:  # xor
            result = xor_decode(payload)
        elif method == 3:  # markov
            result = self._decode_markov(payload)
        elif method == 4:  # delta+markov
            decoded_residuals = self._decode_markov(payload)
            result = delta_decode(decoded_residuals)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if verbose:
            print(f"復元: {len(result)} bytes")
            print(f"時間: {time.time() - start_time:.3f}秒")
        
        return result
    
    def _decode_markov(self, residuals: bytes) -> bytes:
        self.predictor.reset()
        result = bytearray()
        last = 0
        for res in residuals:
            pred = self.predictor.predict(last)
            val = (pred + res) % 256
            result.append(val)
            self.predictor.train(last, val)
            last = val
        return bytes(result)


# =============================================================================
# テスト
# =============================================================================

def run_comprehensive_test():
    print("=" * 80)
    print("STDP予測圧縮 v6 - ハイブリッド最適化版")
    print("=" * 80)
    
    test_cases = [
        ("テキスト短", b"The quick brown fox jumps over the lazy dog."),
        ("テキスト長", b"The quick brown fox jumps over the lazy dog. " * 10),
        ("繰り返しABC", b"ABCABCABCABCABCABCABCABC" * 10),
        ("数字列", b"0123456789" * 20),
        ("英文反復", b"Spiking neural networks process spikes. " * 10),
        ("バイナリ連番", bytes(range(256))),
        ("バイナリ繰返", bytes(range(256)) * 4),
        ("長文", b"Hello World! " * 100),
        ("日本語", "こんにちは世界！".encode('utf-8') * 20),
        ("ソースコード", b"def f(x):\n    return x * 2\n" * 20),
        ("ランダム風", bytes([i * 17 % 256 for i in range(500)])),
    ]
    
    print(f"\n{'データ':<16} {'元':>6} {'v6':>6} {'v6%':>6} {'zlib':>6} {'zlib%':>6} {'勝者':>8} {'手法':<15}")
    print("-" * 90)
    
    wins = 0
    total_v6 = 0
    total_zlib = 0
    
    for name, data in test_cases:
        codec = STDPPredictiveCodecV6()
        compressed = codec.compress(data, verbose=False)
        restored = codec.decompress(compressed, verbose=False)
        
        zlib_comp = zlib.compress(data, level=9)
        
        success = (data == restored)
        ratio_v6 = len(compressed) / len(data) * 100
        ratio_zlib = len(zlib_comp) / len(data) * 100
        
        total_v6 += len(compressed)
        total_zlib += len(zlib_comp)
        
        method = compressed[4]
        method_name = STDPPredictiveCodecV6.METHODS.get(method, "?")
        
        if ratio_v6 <= ratio_zlib:
            winner = "v6🏆"
            wins += 1
        else:
            winner = "zlib"
        
        status = "✅" if success else "❌"
        print(f"{name:<16} {len(data):>6} {len(compressed):>6} {ratio_v6:>5.1f}% {len(zlib_comp):>6} {ratio_zlib:>5.1f}% {winner:>8} {method_name:<15} {status}")
    
    print("-" * 90)
    print(f"v6勝利: {wins}/{len(test_cases)}")
    print(f"合計サイズ: v6={total_v6} bytes, zlib={total_zlib} bytes")
    avg_ratio = total_v6 / total_zlib * 100
    print(f"平均比率: v6はzlibの{avg_ratio:.1f}%")


if __name__ == "__main__":
    run_comprehensive_test()
