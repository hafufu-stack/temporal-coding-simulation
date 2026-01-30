"""
STDP予測圧縮システム v2 (STDP Predictive Compression)
======================================================

真の圧縮（100%以下）を実現する改良版

原理:
1. SNNリザーバが次のバイトを予測
2. 予測と実際の差（残差）のみを保存
3. 残差は0付近に集中 → zlib圧縮が効く

Author: ろーる (cell_activation)
"""

import numpy as np
import time
import struct
import zlib
from typing import List


class SNNPredictor:
    """SNNリザーバ予測器"""
    
    def __init__(self, n_reservoir: int = 100, n_history: int = 8):
        np.random.seed(42)
        self.n_reservoir = n_reservoir
        self.n_history = n_history
        
        # リザーバ初期化
        W = np.random.randn(n_reservoir, n_reservoir)
        rho = max(abs(np.linalg.eigvals(W)))
        self.W_res = W * (0.9 / rho)
        mask = np.random.rand(n_reservoir, n_reservoir) < 0.2
        self.W_res *= mask
        
        self.W_in = np.random.randn(n_reservoir, n_history) * 0.5
        self.W_out = np.zeros(n_reservoir)
        
        self.state = np.zeros(n_reservoir)
        self.history = np.zeros(n_history)
        self.lr = 0.01
    
    def reset(self):
        """状態と学習済み重みをリセット"""
        self.state = np.zeros(self.n_reservoir)
        self.history = np.zeros(self.n_history)
        self.W_out = np.zeros(self.n_reservoir)  # ← これが重要！
    
    def predict(self, input_val: int) -> int:
        x = input_val / 255.0
        self.history = np.roll(self.history, -1)
        self.history[-1] = x
        pre_activation = self.W_res @ self.state + self.W_in @ self.history
        self.state = np.tanh(pre_activation)
        pred_norm = np.dot(self.W_out, self.state)
        return int(np.clip(pred_norm * 255, 0, 255))
    
    def train(self, target_val: int):
        target_norm = target_val / 255.0
        pred_norm = np.dot(self.W_out, self.state)
        error = target_norm - pred_norm
        self.W_out += self.lr * error * self.state


class STDPPredictiveCodec:
    """STDP予測圧縮コーデック"""
    
    MAGIC = b'STDV'
    VERSION = 2
    
    def __init__(self, n_reservoir: int = 100):
        self.n_reservoir = n_reservoir
        self.predictor = SNNPredictor(n_reservoir=n_reservoir)
    
    def compress(self, data: bytes, verbose: bool = True) -> bytes:
        if len(data) == 0:
            return b''
        
        if verbose:
            print(f"入力データサイズ: {len(data)} bytes")
        
        start_time = time.time()
        self.predictor.reset()  # ← 圧縮開始時にリセット
        
        residuals = []
        last_val = 0
        for val in data:
            pred = self.predictor.predict(last_val)
            res = (val - pred) % 256
            residuals.append(res)
            self.predictor.train(val)
            last_val = val
        
        compressed = self._pack(residuals)
        
        if verbose:
            ratio = len(compressed) / len(data) * 100
            print(f"圧縮データサイズ: {len(compressed)} bytes ({ratio:.1f}%)")
            print(f"処理時間: {time.time() - start_time:.3f}秒")
        
        return compressed
    
    def decompress(self, compressed: bytes, verbose: bool = True) -> bytes:
        if len(compressed) == 0:
            return b''
        
        if verbose:
            print(f"圧縮データサイズ: {len(compressed)} bytes")
        
        start_time = time.time()
        residuals = self._unpack(compressed)
        self.predictor.reset()  # ← 解凍開始時も同様にリセット
        
        restored = bytearray()
        last_val = 0
        for res in residuals:
            pred = self.predictor.predict(last_val)
            val = (pred + res) % 256
            restored.append(val)
            self.predictor.train(val)
            last_val = val
        
        if verbose:
            print(f"復元データサイズ: {len(restored)} bytes")
            print(f"処理時間: {time.time() - start_time:.3f}秒")
        
        return bytes(restored)
    
    def _pack(self, residuals: List[int]) -> bytes:
        parts = [self.MAGIC, struct.pack('<HI', self.VERSION, len(residuals)), bytes(residuals)]
        return zlib.compress(b''.join(parts), level=9)
    
    def _unpack(self, compressed: bytes) -> List[int]:
        raw = zlib.decompress(compressed)
        if raw[:4] != self.MAGIC:
            raise ValueError("Invalid file format")
        _, data_len = struct.unpack('<HI', raw[4:10])
        return list(raw[10:10 + data_len])


# =============================================================================
# テスト
# =============================================================================

def run_demo():
    print("=" * 60)
    print("STDP予測圧縮システム v2")
    print("=" * 60)
    
    test_data = b"Hello, STDP World! This is a predictive compression demo."
    print(f"\n元データ: {test_data[:50]}...")
    print(f"データ長: {len(test_data)} bytes")
    
    codec = STDPPredictiveCodec(n_reservoir=100)
    
    print("\n--- 圧縮 ---")
    compressed = codec.compress(test_data)
    
    print("\n--- 解凍 ---")
    restored = codec.decompress(compressed)
    
    print("\n--- 結果 ---")
    if test_data == restored:
        print("✅ 完全一致！")
    else:
        matches = sum(1 for a, b in zip(test_data, restored) if a == b)
        print(f"⚠️ 一致率: {matches}/{len(test_data)}")


def test_various_data():
    print("\n" + "=" * 60)
    print("圧縮率比較テスト")
    print("=" * 60)
    
    test_cases = [
        ("ASCIIテキスト", b"The quick brown fox jumps over the lazy dog. " * 5),
        ("繰り返し", b"ABCABCABCABCABCABCABCABC" * 10),
        ("数字列", b"0123456789" * 20),
        ("英文", b"Spiking neural networks are efficient. " * 5),
        ("バイナリ", bytes(range(256))),
    ]
    
    print(f"\n{'データ':<15} {'元サイズ':>10} {'圧縮後':>10} {'圧縮率':>10} {'復元':>6}")
    print("-" * 60)
    
    for name, data in test_cases:
        codec = STDPPredictiveCodec()
        compressed = codec.compress(data, verbose=False)
        restored = codec.decompress(compressed, verbose=False)
        
        success = (data == restored)
        ratio = len(compressed) / len(data) * 100
        status = "✅" if success else "❌"
        print(f"{name:<15} {len(data):>10} {len(compressed):>10} {ratio:>9.1f}% {status:>6}")


def compare_with_zlib():
    print("\n" + "=" * 60)
    print("zlib との比較")
    print("=" * 60)
    
    text = b"The spiking neural network uses temporal coding. " * 10
    
    codec = STDPPredictiveCodec()
    stdp_compressed = codec.compress(text, verbose=False)
    zlib_compressed = zlib.compress(text, level=9)
    
    print(f"\n元データ: {len(text)} bytes")
    print(f"STDP予測: {len(stdp_compressed)} bytes ({len(stdp_compressed)/len(text)*100:.1f}%)")
    print(f"純zlib:   {len(zlib_compressed)} bytes ({len(zlib_compressed)/len(text)*100:.1f}%)")


if __name__ == "__main__":
    run_demo()
    test_various_data()
    compare_with_zlib()
