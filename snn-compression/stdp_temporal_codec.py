"""
STDP時間符号化圧縮・解凍システム (STDP Temporal Codec)
======================================================

概念実証版: スパイクタイミングによるデータ符号化

動作原理:
1. データバイト → 整数ペア（phase, isi）として符号化
2. STDPでネットワーク重みにパターンを学習
3. 圧縮ファイル = 符号化データ + 学習済み重み

Author: ろーる (cell_activation)
"""

import numpy as np
import time
import struct
import zlib
from typing import List, Tuple


# =============================================================================
# 時間符号化エンコーダ/デコーダ（整数ベース - 精度保証）
# =============================================================================

class TemporalEncoder:
    """データを時間符号化に変換"""
    
    def encode_byte(self, value: int) -> Tuple[int, int]:
        """
        1バイトを2つの4bit整数に分解
        
        value (0-255) → (phase, isi) where:
        - phase = upper 4 bits (0-15)
        - isi = lower 4 bits (0-15)
        """
        phase = (value >> 4) & 0x0F  # 上位4bit
        isi = value & 0x0F           # 下位4bit
        return (phase, isi)
    
    def encode_data(self, data: bytes) -> List[Tuple[int, int]]:
        """バイト列全体を変換"""
        return [self.encode_byte(b) for b in data]


class TemporalDecoder:
    """符号化データを復元"""
    
    def decode(self, encoded: Tuple[int, int]) -> int:
        """2つの4bit整数から1バイトを復元"""
        phase, isi = encoded
        return ((phase & 0x0F) << 4) | (isi & 0x0F)


# =============================================================================
# STDP学習ネットワーク
# =============================================================================

class STDPNetwork:
    """シンプルなSTDP学習ネットワーク"""
    
    def __init__(self, n_neurons: int = 20):
        self.n_neurons = n_neurons
        np.random.seed(42)
        
        # 重み行列
        self.weights = np.random.uniform(0.1, 0.3, (n_neurons, n_neurons))
        np.fill_diagonal(self.weights, 0)
        
        # STDPパラメータ
        self.A_plus = 0.01
        self.A_minus = 0.012
        self.tau = 20.0
    
    def train(self, patterns: List[Tuple[int, int]]):
        """パターンでSTDP学習"""
        for phase, isi in patterns:
            # 仮想的なスパイクタイミングに変換
            t_pre = phase * 3.0   # 0-45ms
            t_post = t_pre + isi * 0.5 + 5.0  # 5-12.5ms後
            
            delta_t = t_post - t_pre
            
            # 全ペアでSTDP更新
            for i in range(min(self.n_neurons, 10)):
                for j in range(min(self.n_neurons, 10)):
                    if i == j:
                        continue
                    
                    if delta_t > 0:
                        dw = self.A_plus * np.exp(-delta_t / self.tau)
                    else:
                        dw = -self.A_minus * np.exp(delta_t / self.tau)
                    
                    self.weights[j, i] += dw * 0.1
                    self.weights[j, i] = np.clip(self.weights[j, i], 0, 1)
    
    def get_weights_bytes(self) -> bytes:
        """重みを量子化してバイト列に"""
        w_quantized = np.clip(self.weights * 255, 0, 255).astype(np.uint8)
        return w_quantized.tobytes()


# =============================================================================
# メインコーデック
# =============================================================================

class STDPTemporalCodec:
    """
    STDP時間符号化コーデック
    
    Usage:
        codec = STDPTemporalCodec()
        compressed = codec.compress(data)
        restored = codec.decompress(compressed)
    """
    
    MAGIC = b'STDC'
    VERSION = 1
    
    def __init__(self, n_neurons: int = 20):
        self.n_neurons = n_neurons
        self.encoder = TemporalEncoder()
        self.decoder = TemporalDecoder()
        self.network = STDPNetwork(n_neurons)
    
    def compress(self, data: bytes, verbose: bool = True) -> bytes:
        """データを圧縮"""
        if verbose:
            print(f"入力データサイズ: {len(data)} bytes")
        
        start_time = time.time()
        
        # 1. データを符号化
        encoded = self.encoder.encode_data(data)
        
        # 2. STDPでネットワークを学習
        self.network.train(encoded)
        
        # 3. パック
        compressed = self._pack(encoded)
        
        elapsed = time.time() - start_time
        
        if verbose:
            ratio = len(compressed) / len(data) * 100
            print(f"圧縮データサイズ: {len(compressed)} bytes ({ratio:.1f}%)")
            print(f"処理時間: {elapsed:.3f}秒")
        
        return compressed
    
    def decompress(self, compressed: bytes, verbose: bool = True) -> bytes:
        """圧縮データを解凍"""
        if verbose:
            print(f"圧縮データサイズ: {len(compressed)} bytes")
        
        start_time = time.time()
        
        # 1. アンパック
        encoded = self._unpack(compressed)
        
        # 2. 符号化データから復元
        restored = bytearray()
        for pair in encoded:
            value = self.decoder.decode(pair)
            restored.append(value)
        
        elapsed = time.time() - start_time
        
        if verbose:
            print(f"復元データサイズ: {len(restored)} bytes")
            print(f"処理時間: {elapsed:.3f}秒")
        
        return bytes(restored)
    
    def _pack(self, encoded: List[Tuple[int, int]]) -> bytes:
        """圧縮フォーマットにパック"""
        parts = []
        
        # ヘッダ
        parts.append(self.MAGIC)
        parts.append(struct.pack('<HI', self.VERSION, len(encoded)))
        
        # 符号化データ（2つの4bit値を1バイトに）
        # phase (4bit) + isi (4bit) = 1 byte per original byte
        data_bytes = bytearray()
        for phase, isi in encoded:
            packed = ((phase & 0x0F) << 4) | (isi & 0x0F)
            data_bytes.append(packed)
        parts.append(bytes(data_bytes))
        
        # 重み
        weights = self.network.get_weights_bytes()
        parts.append(struct.pack('<I', len(weights)))
        parts.append(weights)
        
        # zlib圧縮
        return zlib.compress(b''.join(parts), level=9)
    
    def _unpack(self, compressed: bytes) -> List[Tuple[int, int]]:
        """アンパック"""
        raw = zlib.decompress(compressed)
        offset = 0
        
        # ヘッダ
        magic = raw[offset:offset+4]
        offset += 4
        if magic != self.MAGIC:
            raise ValueError("Invalid file format")
        
        version, data_len = struct.unpack('<HI', raw[offset:offset+6])
        offset += 6
        
        # 符号化データ
        encoded = []
        for i in range(data_len):
            packed = raw[offset + i]
            phase = (packed >> 4) & 0x0F
            isi = packed & 0x0F
            encoded.append((phase, isi))
        
        return encoded


# =============================================================================
# テスト
# =============================================================================

def run_demo():
    """デモ実行"""
    print("=" * 60)
    print("STDP時間符号化圧縮システム デモ")
    print("=" * 60)
    
    # テストデータ
    test_data = b"Hello, STDP World!"
    print(f"\n元データ: {test_data}")
    print(f"データ長: {len(test_data)} bytes")
    
    # コーデック
    codec = STDPTemporalCodec(n_neurons=20)
    
    # 圧縮
    print("\n--- 圧縮 ---")
    compressed = codec.compress(test_data)
    
    # 解凍
    print("\n--- 解凍 ---")
    restored = codec.decompress(compressed)
    
    # 結果
    print("\n--- 結果 ---")
    print(f"元データ:   {test_data}")
    print(f"復元データ: {restored}")
    
    # 一致確認
    if test_data == restored:
        print("\n✅ 完全一致！圧縮・解凍成功！")
    else:
        matches = sum(1 for a, b in zip(test_data, restored) if a == b)
        print(f"\n⚠️ 一致率: {matches}/{len(test_data)}")


def test_various_data():
    """様々なデータでテスト"""
    print("\n" + "=" * 60)
    print("各種データテスト")
    print("=" * 60)
    
    test_cases = [
        ("ASCII テキスト", b"The quick brown fox jumps over the lazy dog"),
        ("数字", b"0123456789"),
        ("日本語UTF-8", "こんにちは".encode('utf-8')),
        ("バイナリ", bytes(range(256))),
        ("繰り返し", b"AAAAAAAAAA"),
    ]
    
    codec = STDPTemporalCodec()
    
    all_passed = True
    for name, data in test_cases:
        compressed = codec.compress(data, verbose=False)
        restored = codec.decompress(compressed, verbose=False)
        
        success = (data == restored)
        ratio = len(compressed) / len(data) * 100
        
        status = "✅" if success else "❌"
        print(f"{status} {name}: 復元={success}, 圧縮率={ratio:.1f}%")
        
        if not success:
            all_passed = False
    
    if all_passed:
        print("\n🎉 全テストパス！")
    else:
        print("\n⚠️ 一部テスト失敗")


if __name__ == "__main__":
    run_demo()
    test_various_data()
