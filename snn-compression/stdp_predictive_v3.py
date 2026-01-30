"""
STDP予測圧縮 v3 - 高精度予測版
================================

改良点:
1. 拡張コンテキスト (16バイト履歴)
2. バイト頻度統計の活用
3. 適応学習率
4. デルタ符号化（差分圧縮）

Author: ろーる (cell_activation)
"""

import numpy as np
import time
import struct
import zlib
from typing import List, Dict
from collections import Counter


class AdaptivePredictor:
    """
    適応型次バイト予測器
    
    3つの予測手法を組み合わせ:
    1. 直前バイト予測（最もシンプル）
    2. 2gram統計（直前2バイトからの予測）
    3. SNNリザーバ予測（パターン学習）
    """
    
    def __init__(self, n_reservoir: int = 64, n_history: int = 16):
        np.random.seed(42)
        
        self.n_reservoir = n_reservoir
        self.n_history = n_history
        
        # --- SNN リザーバ ---
        W = np.random.randn(n_reservoir, n_reservoir) * 0.1
        # スペクトル半径を小さくして安定化
        self.W_res = W
        mask = np.random.rand(n_reservoir, n_reservoir) < 0.1
        self.W_res *= mask
        
        self.W_in = np.random.randn(n_reservoir, n_history) * 0.3
        self.W_out = np.zeros(n_reservoir)
        
        self.state = np.zeros(n_reservoir)
        self.history = np.zeros(n_history)
        
        # 適応学習率
        self.lr = 0.1
        self.lr_decay = 0.999
        
        # --- 統計ベース予測 ---
        self.byte_counts = np.zeros(256)  # 各バイトの出現頻度
        self.pair_counts: Dict[int, Counter] = {}  # 2gram統計
        
        # 前回の予測情報
        self.last_byte = 0
        self.prediction_errors = []
    
    def reset(self):
        """状態リセット"""
        np.random.seed(42)
        self.state = np.zeros(self.n_reservoir)
        self.history = np.zeros(self.n_history)
        self.W_out = np.zeros(self.n_reservoir)
        self.byte_counts = np.zeros(256)
        self.pair_counts = {}
        self.last_byte = 0
        self.lr = 0.1
        self.prediction_errors = []
    
    def predict(self, context_byte: int) -> int:
        """
        複数手法を組み合わせて予測
        """
        x = context_byte / 255.0
        
        # 履歴更新
        self.history = np.roll(self.history, -1)
        self.history[-1] = x
        
        # --- 方法1: 直前バイト予測（差分0を期待）---
        pred_last = context_byte
        
        # --- 方法2: 2gram統計予測 ---
        pred_stat = context_byte
        if context_byte in self.pair_counts and self.pair_counts[context_byte]:
            most_common = self.pair_counts[context_byte].most_common(1)
            if most_common:
                pred_stat = most_common[0][0]
        
        # --- 方法3: SNN予測 ---
        pre_activation = self.W_res @ self.state + self.W_in @ self.history
        self.state = np.tanh(pre_activation)
        pred_snn_norm = np.dot(self.W_out, self.state)
        pred_snn = int(np.clip(pred_snn_norm * 255, 0, 255))
        
        # --- 予測の重み付け平均 ---
        # 序盤は統計が不足 → 直前バイトを重視
        # 後半は統計が充実 → 統計予測を重視
        total_bytes = np.sum(self.byte_counts)
        if total_bytes < 50:
            # 序盤: 直前バイト中心
            final_pred = pred_last
        elif total_bytes < 200:
            # 中盤: 混合
            final_pred = int(0.5 * pred_last + 0.3 * pred_stat + 0.2 * pred_snn)
        else:
            # 後半: 統計+SNN中心
            final_pred = int(0.3 * pred_last + 0.4 * pred_stat + 0.3 * pred_snn)
        
        final_pred = max(0, min(255, final_pred))
        self.last_byte = context_byte
        return final_pred
    
    def train(self, actual_byte: int, context_byte: int):
        """実際の値で学習"""
        # バイト頻度更新
        self.byte_counts[actual_byte] += 1
        
        # 2gram統計更新
        if context_byte not in self.pair_counts:
            self.pair_counts[context_byte] = Counter()
        self.pair_counts[context_byte][actual_byte] += 1
        
        # SNN学習
        target_norm = actual_byte / 255.0
        pred_norm = np.dot(self.W_out, self.state)
        error = target_norm - pred_norm
        self.W_out += self.lr * error * self.state
        
        # 学習率減衰
        self.lr *= self.lr_decay


class STDPPredictiveCodecV3:
    """STDP予測圧縮コーデック v3"""
    
    MAGIC = b'STD3'
    VERSION = 3
    
    def __init__(self):
        self.predictor = AdaptivePredictor()
    
    def compress(self, data: bytes, verbose: bool = True) -> bytes:
        if len(data) == 0:
            return b''
        
        if verbose:
            print(f"入力: {len(data)} bytes")
        
        start_time = time.time()
        self.predictor.reset()
        
        residuals = []
        last_byte = 0
        
        for byte_val in data:
            pred = self.predictor.predict(last_byte)
            res = (byte_val - pred) % 256
            residuals.append(res)
            self.predictor.train(byte_val, last_byte)
            last_byte = byte_val
        
        # 残差の統計
        residuals_arr = np.array(residuals)
        zero_count = np.sum(residuals_arr == 0)
        
        compressed = self._pack(residuals)
        
        if verbose:
            ratio = len(compressed) / len(data) * 100
            print(f"圧縮: {len(compressed)} bytes ({ratio:.1f}%)")
            print(f"予測的中: {zero_count}/{len(data)} ({zero_count/len(data)*100:.1f}%)")
            print(f"時間: {time.time() - start_time:.3f}秒")
        
        return compressed
    
    def decompress(self, compressed: bytes, verbose: bool = True) -> bytes:
        if len(compressed) == 0:
            return b''
        
        if verbose:
            print(f"圧縮データ: {len(compressed)} bytes")
        
        start_time = time.time()
        residuals = self._unpack(compressed)
        self.predictor.reset()
        
        restored = bytearray()
        last_byte = 0
        
        for res in residuals:
            pred = self.predictor.predict(last_byte)
            byte_val = (pred + res) % 256
            restored.append(byte_val)
            self.predictor.train(byte_val, last_byte)
            last_byte = byte_val
        
        if verbose:
            print(f"復元: {len(restored)} bytes")
            print(f"時間: {time.time() - start_time:.3f}秒")
        
        return bytes(restored)
    
    def _pack(self, residuals: List[int]) -> bytes:
        parts = [self.MAGIC, struct.pack('<I', len(residuals)), bytes(residuals)]
        return zlib.compress(b''.join(parts), level=9)
    
    def _unpack(self, compressed: bytes) -> List[int]:
        raw = zlib.decompress(compressed)
        if raw[:4] != self.MAGIC:
            raise ValueError("Invalid format")
        data_len = struct.unpack('<I', raw[4:8])[0]
        return list(raw[8:8 + data_len])


# =============================================================================
# テスト
# =============================================================================

def run_tests():
    print("=" * 60)
    print("STDP予測圧縮 v3 - 高精度予測版")
    print("=" * 60)
    
    test_cases = [
        ("テキスト", b"The quick brown fox jumps over the lazy dog. " * 5),
        ("繰り返し", b"ABCABCABCABCABCABCABCABC" * 10),
        ("数字", b"0123456789" * 20),
        ("英文", b"Spiking neural networks process information using spikes. " * 5),
        ("バイナリ", bytes(range(256))),
        ("長文", b"Hello World! " * 50),
    ]
    
    print(f"\n{'データ':<12} {'元':>8} {'v3':>8} {'v3%':>8} {'zlib':>8} {'zlib%':>8} {'復元':>6}")
    print("-" * 70)
    
    for name, data in test_cases:
        codec = STDPPredictiveCodecV3()
        compressed = codec.compress(data, verbose=False)
        restored = codec.decompress(compressed, verbose=False)
        
        zlib_comp = zlib.compress(data, level=9)
        
        success = (data == restored)
        ratio_v3 = len(compressed) / len(data) * 100
        ratio_zlib = len(zlib_comp) / len(data) * 100
        
        status = "✅" if success else "❌"
        winner = "🏆" if ratio_v3 < ratio_zlib else ""
        
        print(f"{name:<12} {len(data):>8} {len(compressed):>8} {ratio_v3:>7.1f}% {len(zlib_comp):>8} {ratio_zlib:>7.1f}% {status:>6} {winner}")


def detailed_demo():
    print("\n" + "=" * 60)
    print("詳細デモ")
    print("=" * 60)
    
    data = b"Hello, STDP World! This is v3 with adaptive prediction."
    print(f"\n元データ: {data}")
    
    codec = STDPPredictiveCodecV3()
    
    print("\n--- 圧縮 ---")
    compressed = codec.compress(data)
    
    print("\n--- 解凍 ---")
    restored = codec.decompress(compressed)
    
    print(f"\n復元: {restored}")
    print(f"一致: {'✅' if data == restored else '❌'}")


if __name__ == "__main__":
    run_tests()
    detailed_demo()
