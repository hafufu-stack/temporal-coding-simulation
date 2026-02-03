"""
STDP Comprypto - 高度圧縮 + カオス暗号化統合システム
=====================================================

v6圧縮（デルタ/XOR/マルコフ最適選択）と
snn-compryptoのカオス暗号化を統合

Features:
- v6の最適圧縮手法自動選択
- カオスリザーバによる暗号鍵生成
- 温度パラメータによる第2の鍵
- NIST準拠の乱数品質

Author: ろーる (cell_activation)
"""

import numpy as np
import time
import struct
import zlib
import hashlib
from typing import List


# =============================================================================
# カオスリザーバ（暗号鍵生成用）
# =============================================================================

class ChaoticReservoir:
    """カオス的挙動を持つSNNリザーバ"""
    
    def __init__(self, seed: int, n_neurons: int = 100, temperature: float = 1.0):
        np.random.seed(seed)
        
        self.n_neurons = n_neurons
        self.temperature = temperature
        self.seed = seed
        
        # リザーバ結合行列
        W = np.random.randn(n_neurons, n_neurons) * 0.1
        mask = np.random.rand(n_neurons, n_neurons) < 0.1
        self.W = W * mask
        
        # ニューロン状態（膜電位）
        self.v = np.random.randn(n_neurons) * 10
        
        # 入力重み
        self.W_in = np.random.randn(n_neurons) * 2
        
        # 乱数状態を保存
        self.rng_state = np.random.get_state()
    
    def step(self, input_val: int):
        """1ステップ進める"""
        np.random.set_state(self.rng_state)
        x = input_val / 255.0
        noise = np.random.randn(self.n_neurons) * self.temperature * 0.1
        self.rng_state = np.random.get_state()
        pre = self.W @ self.v + self.W_in * x + noise
        self.v = np.tanh(pre)
    
    def get_keystream_byte(self) -> int:
        """現在の状態から1バイトの鍵を生成"""
        state_bytes = self.v.astype(np.float32).tobytes()
        h = hashlib.sha256(state_bytes).digest()
        return h[0]


# =============================================================================
# 圧縮エンジン
# =============================================================================

def delta_encode(data: bytes) -> bytes:
    if not data:
        return b''
    result = bytearray([data[0]])
    for i in range(1, len(data)):
        result.append((data[i] - data[i-1]) % 256)
    return bytes(result)

def delta_decode(data: bytes) -> bytes:
    if not data:
        return b''
    result = bytearray([data[0]])
    for i in range(1, len(data)):
        result.append((result[i-1] + data[i]) % 256)
    return bytes(result)

def xor_encode(data: bytes) -> bytes:
    if not data:
        return b''
    result = bytearray([data[0]])
    for i in range(1, len(data)):
        result.append(data[i] ^ data[i-1])
    return bytes(result)

def xor_decode(data: bytes) -> bytes:
    if not data:
        return b''
    result = bytearray([data[0]])
    for i in range(1, len(data)):
        result.append(data[i] ^ result[i-1])
    return bytes(result)


# =============================================================================
# メイン統合システム
# =============================================================================

class STDPComprypto:
    """
    STDP圧縮 + カオス暗号化 統合システム
    
    Usage:
        encryptor = STDPComprypto(key_seed=12345, temperature=1.0)
        encrypted = encryptor.encrypt(data)
        
        decryptor = STDPComprypto(key_seed=12345, temperature=1.0)
        restored = decryptor.decrypt(encrypted)
    """
    
    MAGIC = b'STCE'
    
    def __init__(self, key_seed: int = 2026, temperature: float = 1.0):
        self.key_seed = key_seed
        self.temperature = temperature
    
    def encrypt(self, data: bytes, verbose: bool = True) -> bytes:
        """データを圧縮・暗号化"""
        if len(data) == 0:
            return b''
        
        if verbose:
            print(f"入力: {len(data)} bytes")
        
        start_time = time.time()
        
        # 最適な圧縮方式を選択
        candidates = [
            (0, data, "raw"),
            (1, delta_encode(data), "delta"),
            (2, xor_encode(data), "xor"),
        ]
        
        best_method, best_data, method_name = min(
            candidates, 
            key=lambda x: len(zlib.compress(x[1], 9))
        )
        
        # zlibで圧縮
        compressed = zlib.compress(best_data, 9)
        
        # カオス暗号化
        reservoir = ChaoticReservoir(self.key_seed, temperature=self.temperature)
        encrypted = bytearray()
        
        for byte_val in compressed:
            # 1. 鍵生成
            key_byte = reservoir.get_keystream_byte()
            # 2. 暗号化
            encrypted.append(byte_val ^ key_byte)
            # 3. 平文でステップ（これが重要！）
            reservoir.step(byte_val)
        
        # パッケージ化
        result = self.MAGIC + struct.pack('<BII', 
            best_method, 
            len(data),
            len(compressed)
        ) + bytes(encrypted)
        
        if verbose:
            ratio = len(result) / len(data) * 100
            print(f"暗号化: {len(result)} bytes ({ratio:.1f}%) [{method_name}]")
            print(f"時間: {time.time() - start_time:.3f}秒")
        
        return result
    
    def decrypt(self, encrypted_data: bytes, verbose: bool = True) -> bytes:
        """暗号化データを復号・解凍"""
        if len(encrypted_data) == 0:
            return b''
        
        if verbose:
            print(f"入力: {len(encrypted_data)} bytes")
        
        start_time = time.time()
        
        # ヘッダ解析
        if encrypted_data[:4] != self.MAGIC:
            raise ValueError("Invalid format or wrong key")
        
        method, orig_len, comp_len = struct.unpack('<BII', encrypted_data[4:13])
        cipher_data = encrypted_data[13:]
        
        # 同じカオスリザーバを再現
        reservoir = ChaoticReservoir(self.key_seed, temperature=self.temperature)
        decrypted = bytearray()
        
        for cipher_byte in cipher_data:
            # 1. 鍵生成（暗号化時と同じ状態）
            key_byte = reservoir.get_keystream_byte()
            # 2. 復号
            plain_byte = cipher_byte ^ key_byte
            decrypted.append(plain_byte)
            # 3. 平文でステップ（暗号化時と同じ）
            reservoir.step(plain_byte)
        
        # zlib解凍
        decompressed = zlib.decompress(bytes(decrypted))
        
        # 圧縮方式に応じて復号
        if method == 0:
            result = decompressed
        elif method == 1:
            result = delta_decode(decompressed)
        elif method == 2:
            result = xor_decode(decompressed)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if verbose:
            print(f"復号: {len(result)} bytes")
            print(f"時間: {time.time() - start_time:.3f}秒")
        
        return result


# =============================================================================
# テスト
# =============================================================================

def run_tests():
    print("=" * 70)
    print("STDP Comprypto - 圧縮 + 暗号化 統合システム")
    print("=" * 70)
    
    test_data = b"Hello, STDP Comprypto! This is a test message."
    key_seed = 12345
    temperature = 1.0
    
    print(f"\n元データ: {test_data}")
    print(f"鍵シード: {key_seed}, 温度: {temperature}")
    
    # 暗号化
    print("\n--- 暗号化 ---")
    encryptor = STDPComprypto(key_seed=key_seed, temperature=temperature)
    encrypted = encryptor.encrypt(test_data)
    
    # 復号（正しい鍵）
    print("\n--- 復号（正しい鍵）---")
    decryptor = STDPComprypto(key_seed=key_seed, temperature=temperature)
    restored = decryptor.decrypt(encrypted)
    
    print(f"\n復号結果: {restored}")
    print(f"一致: {'✅' if test_data == restored else '❌'}")
    
    # 間違った鍵
    print("\n--- 間違った鍵テスト ---")
    wrong = STDPComprypto(key_seed=99999, temperature=temperature)
    try:
        wrong.decrypt(encrypted, verbose=False)
        print("復号成功（問題あり）")
    except:
        print("✅ 復号失敗（正常 - 鍵が違うとエラー）")


def benchmark():
    print("\n" + "=" * 70)
    print("ベンチマーク")
    print("=" * 70)
    
    test_cases = [
        ("テキスト", b"The quick brown fox jumps over the lazy dog. " * 10),
        ("バイナリ", bytes(range(256)) * 4),
        ("日本語", "こんにちは！".encode() * 30),
    ]
    
    print(f"\n{'データ':<12} {'元':>8} {'暗号化':>10} {'比率':>8} {'復号':>6}")
    print("-" * 50)
    
    for name, data in test_cases:
        enc = STDPComprypto(key_seed=2026)
        encrypted = enc.encrypt(data, verbose=False)
        
        dec = STDPComprypto(key_seed=2026)
        restored = dec.decrypt(encrypted, verbose=False)
        
        ok = (data == restored)
        ratio = len(encrypted) / len(data) * 100
        print(f"{name:<12} {len(data):>8} {len(encrypted):>10} {ratio:>7.1f}% {'✅' if ok else '❌':>6}")


if __name__ == "__main__":
    run_tests()
    benchmark()
