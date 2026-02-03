"""
STDP予測圧縮 v4 - ランレングス符号化 + 適応予測
================================================

改良点:
1. ランレングス符号化（RLE）との組み合わせ
2. より攻撃的な予測（直前バイトをそのまま使う）
3. テキスト特化の最適化

Author: ろーる (cell_activation)
"""

import numpy as np
import time
import struct
import zlib
from typing import List, Tuple
from collections import Counter


class SmartPredictor:
    """
    スマート予測器
    
    戦略:
    - 直前バイトは次も同じ可能性が高い → 残差が0になりやすい
    - 2gram統計でパターン学習
    """
    
    def __init__(self):
        self.pair_counts = {}  # 2gram: context -> Counter
        self.last_byte = 0
    
    def reset(self):
        self.pair_counts = {}
        self.last_byte = 0
    
    def predict(self, context: int) -> int:
        """最も可能性の高い次バイトを予測"""
        # 2gram統計があればそれを使う
        if context in self.pair_counts and self.pair_counts[context]:
            mc = self.pair_counts[context].most_common(1)
            if mc:
                return mc[0][0]
        # なければ直前バイトをそのまま
        return context
    
    def train(self, actual: int, context: int):
        """統計を更新"""
        if context not in self.pair_counts:
            self.pair_counts[context] = Counter()
        self.pair_counts[context][actual] += 1


def run_length_encode(data: bytes) -> bytes:
    """ランレングス符号化"""
    if not data:
        return b''
    
    result = bytearray()
    i = 0
    while i < len(data):
        byte_val = data[i]
        run_length = 1
        
        # 同じバイトが連続する長さを計測
        while i + run_length < len(data) and data[i + run_length] == byte_val and run_length < 255:
            run_length += 1
        
        if run_length >= 4:
            # RLEマーカー（0xFF） + バイト値 + 長さ
            result.extend([0xFF, byte_val, run_length])
            i += run_length
        else:
            # そのまま出力（0xFFの場合はエスケープ）
            if byte_val == 0xFF:
                result.extend([0xFF, 0xFF, 1])
            else:
                result.append(byte_val)
            i += 1
    
    return bytes(result)


def run_length_decode(data: bytes) -> bytes:
    """ランレングス復号"""
    if not data:
        return b''
    
    result = bytearray()
    i = 0
    while i < len(data):
        if data[i] == 0xFF and i + 2 < len(data):
            byte_val = data[i + 1]
            run_length = data[i + 2]
            result.extend([byte_val] * run_length)
            i += 3
        else:
            result.append(data[i])
            i += 1
    
    return bytes(result)


class STDPPredictiveCodecV4:
    """STDP予測圧縮コーデック v4"""
    
    MAGIC = b'STD4'
    
    def __init__(self):
        self.predictor = SmartPredictor()
    
    def compress(self, data: bytes, verbose: bool = True) -> bytes:
        if len(data) == 0:
            return b''
        
        if verbose:
            print(f"入力: {len(data)} bytes")
        
        start_time = time.time()
        self.predictor.reset()
        
        # 方法1: 予測圧縮
        residuals = []
        last_byte = 0
        for byte_val in data:
            pred = self.predictor.predict(last_byte)
            res = (byte_val - pred) % 256
            residuals.append(res)
            self.predictor.train(byte_val, last_byte)
            last_byte = byte_val
        
        residuals_bytes = bytes(residuals)
        
        # 方法2: 残差にRLEを適用
        rle_residuals = run_length_encode(residuals_bytes)
        
        # 方法3: 生データにRLEを適用
        rle_raw = run_length_encode(data)
        
        # 各方法をzlib圧縮して最小を選択
        candidates = [
            (b'\x00' + zlib.compress(data, 9), "raw+zlib"),
            (b'\x01' + zlib.compress(residuals_bytes, 9), "pred+zlib"),
            (b'\x02' + zlib.compress(rle_raw, 9), "rle+zlib"),
            (b'\x03' + zlib.compress(rle_residuals, 9), "pred+rle+zlib"),
        ]
        
        best = min(candidates, key=lambda x: len(x[0]))
        compressed = self.MAGIC + struct.pack('<I', len(data)) + best[0]
        
        if verbose:
            ratio = len(compressed) / len(data) * 100
            print(f"圧縮: {len(compressed)} bytes ({ratio:.1f}%) [{best[1]}]")
            print(f"時間: {time.time() - start_time:.3f}秒")
        
        return compressed
    
    def decompress(self, compressed: bytes, verbose: bool = True) -> bytes:
        if len(compressed) == 0:
            return b''
        
        if verbose:
            print(f"圧縮データ: {len(compressed)} bytes")
        
        start_time = time.time()
        
        # ヘッダ解析
        if compressed[:4] != self.MAGIC:
            raise ValueError("Invalid format")
        
        orig_len = struct.unpack('<I', compressed[4:8])[0]
        method = compressed[8]
        payload = zlib.decompress(compressed[9:])
        
        if method == 0:  # raw+zlib
            result = payload
        elif method == 1:  # pred+zlib
            result = self._decode_residuals(payload)
        elif method == 2:  # rle+zlib
            result = run_length_decode(payload)
        elif method == 3:  # pred+rle+zlib
            residuals = run_length_decode(payload)
            result = self._decode_residuals(residuals)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if verbose:
            print(f"復元: {len(result)} bytes")
            print(f"時間: {time.time() - start_time:.3f}秒")
        
        return result
    
    def _decode_residuals(self, residuals: bytes) -> bytes:
        """残差から元データを復元"""
        self.predictor.reset()
        result = bytearray()
        last_byte = 0
        
        for res in residuals:
            pred = self.predictor.predict(last_byte)
            byte_val = (pred + res) % 256
            result.append(byte_val)
            self.predictor.train(byte_val, last_byte)
            last_byte = byte_val
        
        return bytes(result)


# =============================================================================
# テスト
# =============================================================================

def run_tests():
    print("=" * 70)
    print("STDP予測圧縮 v4 - 適応選択版")
    print("=" * 70)
    
    test_cases = [
        ("テキスト", b"The quick brown fox jumps over the lazy dog. " * 5),
        ("繰り返し", b"ABCABCABCABCABCABCABCABC" * 10),
        ("数字", b"0123456789" * 20),
        ("英文", b"Spiking neural networks process information using spikes. " * 5),
        ("バイナリ", bytes(range(256))),
        ("長文", b"Hello World! " * 50),
        ("日本語", "こんにちはスパイキングニューラルネットワーク！".encode('utf-8') * 10),
    ]
    
    print(f"\n{'データ':<12} {'元':>8} {'v4':>8} {'v4%':>8} {'zlib':>8} {'zlib%':>8} {'勝者':>6}")
    print("-" * 70)
    
    total_v4 = 0
    total_zlib = 0
    wins = 0
    
    for name, data in test_cases:
        codec = STDPPredictiveCodecV4()
        compressed = codec.compress(data, verbose=False)
        restored = codec.decompress(compressed, verbose=False)
        
        zlib_comp = zlib.compress(data, level=9)
        
        success = (data == restored)
        ratio_v4 = len(compressed) / len(data) * 100
        ratio_zlib = len(zlib_comp) / len(data) * 100
        
        total_v4 += len(compressed)
        total_zlib += len(zlib_comp)
        
        if ratio_v4 <= ratio_zlib:
            winner = "v4🏆"
            wins += 1
        else:
            winner = "zlib"
        
        status = "✅" if success else "❌"
        print(f"{name:<12} {len(data):>8} {len(compressed):>8} {ratio_v4:>7.1f}% {len(zlib_comp):>8} {ratio_zlib:>7.1f}% {winner:>6} {status}")
    
    print("-" * 70)
    print(f"v4勝利: {wins}/{len(test_cases)}")


if __name__ == "__main__":
    run_tests()
