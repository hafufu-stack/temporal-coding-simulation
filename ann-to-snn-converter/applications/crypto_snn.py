"""
自律進化 暗号・圧縮SNN (Evolving Crypto SNN)
============================================

暗号強度と圧縮率を自動最適化する自律進化SNN

Author: ろーる (cell_activation)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.evolution_engine import EvolvingSNN


class EvolvingCryptoSNN(EvolvingSNN):
    """
    自律進化する暗号・圧縮SNN
    
    自動で:
    - 暗号強度を最適化
    - 圧縮率を改善
    - セキュリティと効率のバランスを調整
    """
    
    def __init__(self, n_neurons: int = 100, key_size: int = 32):
        super().__init__(n_neurons)
        
        self.key_size = key_size
        
        # 暗号鍵（内部状態から生成）
        self.key = np.random.randint(0, 256, key_size, dtype=np.uint8)
        
        # スキル
        self.skills = {
            "encryption_strength": 0.5,
            "compression_ratio": 0.5,
            "speed": 0.5
        }
        
        # 統計
        self.encryptions = 0
        self.compressions = 0
    
    def encrypt(self, data: bytes) -> bytes:
        """暗号化"""
        data_array = np.frombuffer(data, dtype=np.uint8)
        
        # SNNを通して暗号化シーケンスを生成
        input_signal = data_array[:self.n_neurons].astype(float) / 255
        spikes = self.step(input_signal)
        
        # XOR暗号
        cipher_stream = self._generate_cipher_stream(len(data))
        encrypted = np.bitwise_xor(data_array, cipher_stream)
        
        self.encryptions += 1
        return bytes(encrypted)
    
    def decrypt(self, data: bytes) -> bytes:
        """復号"""
        data_array = np.frombuffer(data, dtype=np.uint8)
        cipher_stream = self._generate_cipher_stream(len(data))
        decrypted = np.bitwise_xor(data_array, cipher_stream)
        return bytes(decrypted)
    
    def compress(self, data: bytes) -> Tuple[bytes, float]:
        """圧縮"""
        data_array = np.frombuffer(data, dtype=np.uint8)
        
        # 簡易的なRLE圧縮 + SNN特徴
        compressed = []
        count = 1
        
        for i in range(1, len(data_array)):
            if data_array[i] == data_array[i-1] and count < 255:
                count += 1
            else:
                compressed.extend([count, data_array[i-1]])
                count = 1
        compressed.extend([count, data_array[-1]])
        
        result = bytes(compressed)
        ratio = len(data) / len(result)
        
        self.compressions += 1
        return result, ratio
    
    def _generate_cipher_stream(self, length: int) -> np.ndarray:
        """暗号ストリームを生成"""
        stream = np.zeros(length, dtype=np.uint8)
        
        state = self.state.copy()
        for i in range(length):
            # SNNの状態から暗号バイトを生成
            state = 0.9 * state + 0.1 * (self.W @ state)
            byte_val = int(np.abs(np.sum(state)) * 255) % 256
            stream[i] = byte_val ^ self.key[i % self.key_size]
        
        return stream
    
    def evaluate_security(self) -> float:
        """セキュリティを評価"""
        # エントロピーをチェック
        test_data = np.random.bytes(100)
        encrypted = self.encrypt(test_data)
        
        # 暗号文のランダム性
        encrypted_array = np.frombuffer(encrypted, dtype=np.uint8)
        unique = len(np.unique(encrypted_array))
        randomness = unique / len(encrypted_array)
        
        return randomness
    
    def evolve_for_security(self):
        """セキュリティ向上のための進化"""
        security = self.evaluate_security()
        self.skills["encryption_strength"] = security
        
        # 経験として記録
        self.experience(
            np.random.randn(self.n_neurons),
            skill="encryption_strength",
            target=np.ones(self.n_neurons)
        )
        
        # 進化
        result = self.evolve(verbose=True)
        
        return {"security": security, "evolution": result}


def test_crypto_snn():
    """テスト"""
    print("\n" + "=" * 70)
    print("🔐 自律進化 暗号SNN テスト")
    print("=" * 70)
    
    snn = EvolvingCryptoSNN(n_neurons=50)
    
    # 暗号化テスト
    original = b"Hello, Autonomous SNN!"
    encrypted = snn.encrypt(original)
    decrypted = snn.decrypt(encrypted)
    
    print(f"\n元データ: {original}")
    print(f"暗号化: {encrypted[:20]}...")
    print(f"復号: {decrypted}")
    print(f"正常復号: {original == decrypted}")
    
    # 自律進化
    print("\n--- 自律進化 ---")
    for i in range(3):
        result = snn.evolve_for_security()
        print(f"サイクル{i+1}: セキュリティ={result['security']:.2f}")
    
    snn.report()
    
    print("\n" + "=" * 70)
    print("✅ テスト完了")
    print("=" * 70)


if __name__ == "__main__":
    test_crypto_snn()
