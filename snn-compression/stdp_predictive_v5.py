"""
STDP予測圧縮 v5 - PPM風高次コンテキスト予測
=============================================

PPM (Prediction by Partial Matching) を参考に:
1. 長いコンテキスト（4バイト）から予測を試みる
2. 見つからなければ短いコンテキストにフォールバック
3. 最終的には直前バイトを使う

Author: ろーる (cell_activation)
"""

import numpy as np
import time
import struct
import zlib
from typing import List, Dict, Optional
from collections import Counter


class PPMPredictor:
    """
    PPM風予測器
    
    コンテキスト長: 4 → 3 → 2 → 1 → 0 (デフォルト)
    """
    
    def __init__(self, max_order: int = 4):
        self.max_order = max_order
        # 各オーダーの統計: context_tuple -> Counter
        self.contexts: List[Dict[tuple, Counter]] = [
            {} for _ in range(max_order + 1)
        ]
        self.history = []
    
    def reset(self):
        self.contexts = [{} for _ in range(self.max_order + 1)]
        self.history = []
    
    def predict(self) -> int:
        """
        最長マッチするコンテキストから予測
        """
        # 長いコンテキストから試す
        for order in range(min(self.max_order, len(self.history)), -1, -1):
            if order == 0:
                # オーダー0: 全体統計
                if self.contexts[0] and () in self.contexts[0]:
                    mc = self.contexts[0][()].most_common(1)
                    if mc:
                        return mc[0][0]
            else:
                # オーダーn: 直前nバイトをコンテキストとして使用
                ctx = tuple(self.history[-order:])
                if ctx in self.contexts[order]:
                    mc = self.contexts[order][ctx].most_common(1)
                    if mc:
                        return mc[0][0]
        
        # フォールバック: 直前バイト
        if self.history:
            return self.history[-1]
        return 0
    
    def train(self, byte_val: int):
        """統計を更新"""
        # 各オーダーで更新
        for order in range(min(self.max_order, len(self.history)) + 1):
            if order == 0:
                ctx = ()
            else:
                ctx = tuple(self.history[-order:])
            
            if ctx not in self.contexts[order]:
                self.contexts[order][ctx] = Counter()
            self.contexts[order][ctx][byte_val] += 1
        
        # 履歴に追加（最大長制限）
        self.history.append(byte_val)
        if len(self.history) > 100:
            self.history = self.history[-50:]


class STDPPredictiveCodecV5:
    """STDP予測圧縮コーデック v5 - PPM風"""
    
    MAGIC = b'STD5'
    
    def __init__(self, max_order: int = 4):
        self.max_order = max_order
        self.predictor = PPMPredictor(max_order=max_order)
    
    def compress(self, data: bytes, verbose: bool = True) -> bytes:
        if len(data) == 0:
            return b''
        
        if verbose:
            print(f"入力: {len(data)} bytes")
        
        start_time = time.time()
        self.predictor.reset()
        
        residuals = []
        for byte_val in data:
            pred = self.predictor.predict()
            res = (byte_val - pred) % 256
            residuals.append(res)
            self.predictor.train(byte_val)
        
        residuals_bytes = bytes(residuals)
        
        # 方法選択: 生データ vs 残差
        comp_raw = zlib.compress(data, 9)
        comp_res = zlib.compress(residuals_bytes, 9)
        
        if len(comp_res) < len(comp_raw):
            compressed = self.MAGIC + b'\x01' + struct.pack('<I', len(data)) + comp_res
            method = "pred"
        else:
            compressed = self.MAGIC + b'\x00' + struct.pack('<I', len(data)) + comp_raw
            method = "raw"
        
        if verbose:
            ratio = len(compressed) / len(data) * 100
            zeros = sum(1 for r in residuals if r == 0)
            print(f"圧縮: {len(compressed)} bytes ({ratio:.1f}%) [{method}]")
            print(f"予測的中: {zeros}/{len(data)} ({zeros/len(data)*100:.1f}%)")
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
        
        method = compressed[4]
        orig_len = struct.unpack('<I', compressed[5:9])[0]
        payload = zlib.decompress(compressed[9:])
        
        if method == 0:  # raw
            result = payload
        else:  # pred
            self.predictor.reset()
            result = bytearray()
            for res in payload:
                pred = self.predictor.predict()
                byte_val = (pred + res) % 256
                result.append(byte_val)
                self.predictor.train(byte_val)
            result = bytes(result)
        
        if verbose:
            print(f"復元: {len(result)} bytes")
            print(f"時間: {time.time() - start_time:.3f}秒")
        
        return result


# =============================================================================
# テスト
# =============================================================================

def run_tests():
    print("=" * 75)
    print("STDP予測圧縮 v5 - PPM風高次コンテキスト予測")
    print("=" * 75)
    
    test_cases = [
        ("テキスト", b"The quick brown fox jumps over the lazy dog. " * 5),
        ("繰り返し", b"ABCABCABCABCABCABCABCABC" * 10),
        ("数字", b"0123456789" * 20),
        ("英文", b"Spiking neural networks process information using spikes. " * 5),
        ("バイナリ", bytes(range(256))),
        ("長文", b"Hello World! " * 50),
        ("日本語", "こんにちはスパイキングニューラルネットワーク！".encode('utf-8') * 10),
        ("ソースコード", b"def hello():\n    print('Hello')\n" * 20),
    ]
    
    print(f"\n{'データ':<14} {'元':>7} {'v5':>7} {'v5%':>7} {'zlib':>7} {'zlib%':>7} {'勝者':>6}")
    print("-" * 75)
    
    wins = 0
    
    for name, data in test_cases:
        codec = STDPPredictiveCodecV5(max_order=4)
        compressed = codec.compress(data, verbose=False)
        restored = codec.decompress(compressed, verbose=False)
        
        zlib_comp = zlib.compress(data, level=9)
        
        success = (data == restored)
        ratio_v5 = len(compressed) / len(data) * 100
        ratio_zlib = len(zlib_comp) / len(data) * 100
        
        if ratio_v5 <= ratio_zlib:
            winner = "v5🏆"
            wins += 1
        else:
            winner = "zlib"
        
        status = "✅" if success else "❌"
        print(f"{name:<14} {len(data):>7} {len(compressed):>7} {ratio_v5:>6.1f}% {len(zlib_comp):>7} {ratio_zlib:>6.1f}% {winner:>6} {status}")
    
    print("-" * 75)
    print(f"v5勝利: {wins}/{len(test_cases)}")


def demo_prediction():
    print("\n" + "=" * 75)
    print("予測デモ")
    print("=" * 75)
    
    data = b"ABCABCABCABC"
    print(f"\n入力: {data}")
    
    pred = PPMPredictor(max_order=4)
    
    print("\n予測過程:")
    for i, byte_val in enumerate(data):
        prediction = pred.predict()
        correct = "✅" if prediction == byte_val else "❌"
        print(f"  位置{i}: 予測={chr(prediction) if 32<=prediction<127 else '?'} 実際={chr(byte_val)} {correct}")
        pred.train(byte_val)


if __name__ == "__main__":
    run_tests()
    demo_prediction()
