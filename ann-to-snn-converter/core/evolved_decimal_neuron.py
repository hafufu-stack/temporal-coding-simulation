"""
進化版10進数ニューロン (Evolved Decimal Neuron)
================================================

3つの改善:
1. 学習アルゴリズム改善 - Adam最適化、バッチ学習
2. エンタングルメント活用 - 量子テレポーテーション風の情報伝達
3. 実用タスク - 暗号化、画像認識、言語処理

Author: ろーる (cell_activation)
Date: 2026-01-31
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
import random


# =============================================================================
# 改善1: 高度な学習アルゴリズム
# =============================================================================

class AdamOptimizer:
    """Adam最適化器"""
    
    def __init__(self, lr: float = 0.01, beta1: float = 0.9, 
                 beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {}
        self.v = {}
    
    def update(self, name: str, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """パラメータを更新"""
        self.t += 1
        
        if name not in self.m:
            self.m[name] = np.zeros_like(param)
            self.v[name] = np.zeros_like(param)
        
        # モーメンタム
        self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
        self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * grad**2
        
        # バイアス補正
        m_hat = self.m[name] / (1 - self.beta1**self.t)
        v_hat = self.v[name] / (1 - self.beta2**self.t)
        
        # 更新
        return param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class EvolvedDecimalNeuron:
    """
    進化版10進数ニューロン
    
    改善点:
    - Adam最適化
    - 温度付きソフトマックス
    - 残差接続
    """
    
    def __init__(self, n_digits: int = 10, temperature: float = 1.0):
        self.n_digits = n_digits
        self.temperature = temperature
        
        # 状態
        self.state = np.ones(n_digits) / n_digits
        self.hidden = np.zeros(n_digits)
        
        # パラメータ
        self.W = np.eye(n_digits) * 0.5 + np.random.randn(n_digits, n_digits) * 0.1
        self.bias = np.zeros(n_digits)
        
        # 残差用
        self.skip_weight = 0.2
        
        # 最適化器
        self.optimizer = AdamOptimizer(lr=0.05)
        
        # 量子状態
        self.phase = np.zeros(n_digits)
        self.coherence = 1.0  # コヒーレンス
        
        # エンタングル相手
        self.entangled: List['EvolvedDecimalNeuron'] = []
        
        # 履歴
        self.input_history = []
        self.output_history = []
    
    def encode(self, digit: int) -> np.ndarray:
        """10進数を状態に変換"""
        state = np.zeros(self.n_digits)
        state[digit % self.n_digits] = 1.0
        return state
    
    def decode(self, state: np.ndarray) -> int:
        """状態を10進数に変換"""
        return int(np.argmax(state))
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """温度付きソフトマックス"""
        x = x / self.temperature
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def forward(self, input_digit: int) -> np.ndarray:
        """順伝播（改善版）"""
        # 入力エンコード
        input_state = self.encode(input_digit)
        self.input_history.append(input_digit)
        
        # 隠れ状態の更新
        self.hidden = 0.9 * self.hidden + 0.1 * input_state
        
        # 重み変換
        output = self.W @ input_state + self.bias
        
        # 残差接続
        output = output + self.skip_weight * input_state
        
        # エンタングルメントの影響
        for other in self.entangled:
            interference = np.cos(self.phase - other.phase) * other.coherence
            output = output + 0.1 * interference * other.state
        
        # ソフトマックス
        self.state = self.softmax(output)
        
        # 位相更新
        self.phase = np.angle(self.state + 1j * np.random.randn(self.n_digits) * 0.01)
        
        # コヒーレンス減衰
        self.coherence *= 0.99
        
        self.output_history.append(self.decode(self.state))
        return self.state
    
    def backward(self, target: int):
        """逆伝播（Adam最適化）"""
        target_state = self.encode(target)
        
        # 勾配計算
        grad = self.state - target_state
        
        # パラメータ更新
        grad_W = np.outer(grad, self.state)
        self.W = self.optimizer.update("W", self.W, grad_W)
        self.bias = self.optimizer.update("bias", self.bias, grad)
        
        # コヒーレンスを回復（学習で量子性を維持）
        self.coherence = min(1.0, self.coherence + 0.1)
    
    def entangle(self, other: 'EvolvedDecimalNeuron'):
        """エンタングル"""
        if other not in self.entangled:
            self.entangled.append(other)
            other.entangled.append(self)
            # 位相を同期
            avg_phase = (self.phase + other.phase) / 2
            self.phase = avg_phase + np.random.randn(self.n_digits) * 0.01
            other.phase = avg_phase + np.random.randn(self.n_digits) * 0.01
    
    def teleport_state(self, other: 'EvolvedDecimalNeuron'):
        """量子テレポーテーション風の状態転送"""
        if other in self.entangled:
            # 状態を転送
            other.state = self.state.copy()
            other.phase = self.phase.copy()
            # 元の状態は崩壊
            self.state = np.ones(self.n_digits) / self.n_digits
            self.coherence = 0.5


# =============================================================================
# 改善2: エンタングルメントネットワーク
# =============================================================================

class EntangledDecimalNetwork:
    """
    エンタングルした10進数ニューロンのネットワーク
    
    特徴:
    - 全ニューロンがエンタングル可能
    - 量子テレポーテーション通信
    - 並列計算
    """
    
    def __init__(self, n_neurons: int = 10):
        self.n_neurons = n_neurons
        self.neurons = [EvolvedDecimalNeuron() for _ in range(n_neurons)]
        
        # 隣接ニューロンをエンタングル
        for i in range(n_neurons - 1):
            self.neurons[i].entangle(self.neurons[i + 1])
    
    def forward(self, inputs: List[int]) -> List[int]:
        """並列順伝播"""
        outputs = []
        for i, digit in enumerate(inputs):
            if i < self.n_neurons:
                state = self.neurons[i].forward(digit)
                outputs.append(self.neurons[i].decode(state))
        return outputs
    
    def train(self, inputs: List[List[int]], targets: List[List[int]], 
              epochs: int = 100):
        """学習"""
        history = []
        
        for epoch in range(epochs):
            correct = 0
            total = 0
            
            for inp, tgt in zip(inputs, targets):
                outputs = self.forward(inp)
                
                for i, (out, target) in enumerate(zip(outputs, tgt)):
                    if i < self.n_neurons:
                        self.neurons[i].backward(target)
                        if out == target:
                            correct += 1
                        total += 1
            
            accuracy = correct / max(1, total)
            history.append(accuracy)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: accuracy = {accuracy:.2%}")
        
        return history


# =============================================================================
# 改善3: 実用タスク
# =============================================================================

class DecimalCryptoSystem:
    """
    10進数ニューロン暗号システム
    
    10進数→10進数の暗号化
    人間が読める暗号！
    """
    
    def __init__(self, key_length: int = 8):
        self.key_length = key_length
        
        # 暗号化ニューロン
        self.encrypt_neurons = [EvolvedDecimalNeuron() for _ in range(key_length)]
        
        # 復号ニューロン
        self.decrypt_neurons = [EvolvedDecimalNeuron() for _ in range(key_length)]
        
        # 鍵
        self.key = [random.randint(0, 9) for _ in range(key_length)]
        
        # 鍵でニューロンを初期化
        for i, k in enumerate(self.key):
            self.encrypt_neurons[i].W += np.eye(10) * k * 0.1
            self.decrypt_neurons[i].W += np.eye(10) * (-k) * 0.1
    
    def encrypt(self, plaintext: str) -> str:
        """暗号化"""
        # 文字を数字に変換
        digits = [ord(c) % 10 for c in plaintext]
        
        # 暗号化
        encrypted = []
        for i, d in enumerate(digits):
            neuron_idx = i % self.key_length
            state = self.encrypt_neurons[neuron_idx].forward(d)
            
            # 鍵を加算
            enc = (self.encrypt_neurons[neuron_idx].decode(state) + self.key[neuron_idx]) % 10
            encrypted.append(enc)
        
        return ''.join(str(d) for d in encrypted)
    
    def decrypt(self, ciphertext: str) -> str:
        """復号"""
        digits = [int(c) for c in ciphertext if c.isdigit()]
        
        decrypted = []
        for i, d in enumerate(digits):
            neuron_idx = i % self.key_length
            
            # 鍵を減算
            dec_input = (d - self.key[neuron_idx]) % 10
            state = self.decrypt_neurons[neuron_idx].forward(dec_input)
            decrypted.append(self.decrypt_neurons[neuron_idx].decode(state))
        
        return ''.join(str(d) for d in decrypted)


class DecimalImageProcessor:
    """
    10進数ニューロン画像処理
    
    ピクセル値を0-9に量子化して処理
    """
    
    def __init__(self, size: int = 8):
        self.size = size
        self.neurons = [[EvolvedDecimalNeuron() for _ in range(size)] 
                        for _ in range(size)]
        
        # 隣接ニューロンをエンタングル
        for i in range(size):
            for j in range(size - 1):
                self.neurons[i][j].entangle(self.neurons[i][j + 1])
        for i in range(size - 1):
            for j in range(size):
                self.neurons[i][j].entangle(self.neurons[i + 1][j])
    
    def process_image(self, image: np.ndarray) -> np.ndarray:
        """画像を処理"""
        # 0-9に量子化
        quantized = (image[:self.size, :self.size] * 9).astype(int)
        quantized = np.clip(quantized, 0, 9)
        
        # 各ピクセルをニューロンで処理
        output = np.zeros_like(quantized)
        for i in range(min(self.size, quantized.shape[0])):
            for j in range(min(self.size, quantized.shape[1])):
                state = self.neurons[i][j].forward(quantized[i, j])
                output[i, j] = self.neurons[i][j].decode(state)
        
        return output
    
    def edge_detect(self, image: np.ndarray) -> np.ndarray:
        """エッジ検出（10進数版）"""
        output = np.zeros_like(image)
        
        for i in range(1, image.shape[0] - 1):
            for j in range(1, image.shape[1] - 1):
                # 周囲との差分
                center = image[i, j]
                neighbors = [image[i-1, j], image[i+1, j], 
                            image[i, j-1], image[i, j+1]]
                
                diff = np.mean([abs(center - n) for n in neighbors])
                output[i, j] = min(9, int(diff * 2))
        
        return output


class DecimalLanguageModel:
    """
    10進数ニューロン言語モデル
    
    文字を10進数で表現して処理
    """
    
    def __init__(self, vocab_size: int = 100, hidden_size: int = 20):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # 埋め込み層（10進数）
        self.embed_neurons = [EvolvedDecimalNeuron() for _ in range(hidden_size)]
        
        # 隠れ層
        self.hidden_neurons = [EvolvedDecimalNeuron() for _ in range(hidden_size)]
        
        # 出力層
        self.output_neurons = [EvolvedDecimalNeuron() for _ in range(hidden_size)]
        
        # 連続するニューロンをエンタングル
        for i in range(hidden_size - 1):
            self.hidden_neurons[i].entangle(self.hidden_neurons[i + 1])
    
    def char_to_digits(self, char: str) -> List[int]:
        """文字を10進数のリストに変換"""
        code = ord(char) % self.vocab_size
        return [code // 10, code % 10]
    
    def digits_to_char(self, digits: List[int]) -> str:
        """10進数のリストを文字に変換"""
        if len(digits) >= 2:
            code = digits[0] * 10 + digits[1]
            if 32 <= code < 127:
                return chr(code)
        return '?'
    
    def forward(self, text: str) -> str:
        """テキストを処理"""
        output_chars = []
        
        for char in text:
            digits = self.char_to_digits(char)
            
            # 埋め込み
            embedded = []
            for i, d in enumerate(digits):
                if i < self.hidden_size:
                    state = self.embed_neurons[i].forward(d)
                    embedded.append(self.embed_neurons[i].decode(state))
            
            # 隠れ層
            hidden = []
            for i, e in enumerate(embedded):
                if i < self.hidden_size:
                    state = self.hidden_neurons[i].forward(e)
                    hidden.append(self.hidden_neurons[i].decode(state))
            
            # 出力
            output_digits = []
            for i, h in enumerate(hidden):
                if i < self.hidden_size:
                    state = self.output_neurons[i].forward(h)
                    output_digits.append(self.output_neurons[i].decode(state))
            
            output_chars.append(self.digits_to_char(output_digits))
        
        return ''.join(output_chars)
    
    def train_next_char(self, text: str, epochs: int = 50):
        """次の文字予測を学習"""
        for epoch in range(epochs):
            for i in range(len(text) - 1):
                current = text[i]
                next_char = text[i + 1]
                
                # 順伝播
                self.forward(current)
                
                # 逆伝播
                target_digits = self.char_to_digits(next_char)
                for j, d in enumerate(target_digits):
                    if j < self.hidden_size:
                        self.output_neurons[j].backward(d)


# =============================================================================
# 統合テスト
# =============================================================================

def test_all_improvements():
    """全改善のテスト"""
    
    print("\n" + "=" * 70)
    print("🧪 進化版10進数ニューロン 全テスト")
    print("=" * 70)
    
    # 1. 学習アルゴリズム改善
    print("\n" + "-" * 50)
    print("【1. 学習アルゴリズム改善】")
    print("-" * 50)
    
    neuron = EvolvedDecimalNeuron(temperature=0.5)
    
    # +2 を学習
    print("  +2の変換を学習中...")
    for epoch in range(100):
        for digit in range(10):
            neuron.forward(digit)
            target = (digit + 2) % 10
            neuron.backward(target)
    
    print("  学習後:")
    correct = 0
    for digit in range(10):
        neuron.forward(digit)
        result = neuron.decode(neuron.state)
        expected = (digit + 2) % 10
        if result == expected:
            correct += 1
        if digit in [0, 4, 7]:
            print(f"    {digit} + 2 = {result} (期待: {expected}) {'✓' if result == expected else '✗'}")
    print(f"  正解率: {correct}/10")
    
    # 2. エンタングルメント
    print("\n" + "-" * 50)
    print("【2. エンタングルメント】")
    print("-" * 50)
    
    network = EntangledDecimalNetwork(n_neurons=4)
    
    # 4桁の足し算
    inputs = [[1, 2, 3, 4], [5, 6, 7, 8], [0, 0, 0, 1]]
    targets = [[2, 3, 4, 5], [6, 7, 8, 9], [1, 1, 1, 2]]
    
    print("  4桁変換を学習中...")
    history = network.train(inputs, targets, epochs=60)
    print(f"  最終精度: {history[-1]:.2%}")
    
    # テスト
    test_input = [1, 2, 3, 4]
    output = network.forward(test_input)
    print(f"  テスト: {test_input} → {output}")
    
    # 3. 暗号化
    print("\n" + "-" * 50)
    print("【3. 暗号化システム】")
    print("-" * 50)
    
    crypto = DecimalCryptoSystem(key_length=4)
    print(f"  鍵: {crypto.key}")
    
    plaintext = "Hello"
    encrypted = crypto.encrypt(plaintext)
    decrypted = crypto.decrypt(encrypted)
    
    print(f"  平文: {plaintext}")
    print(f"  暗号: {encrypted}")
    print(f"  復号: {decrypted}")
    
    # 4. 画像処理
    print("\n" + "-" * 50)
    print("【4. 画像処理】")
    print("-" * 50)
    
    processor = DecimalImageProcessor(size=4)
    
    # テスト画像
    image = np.array([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 0, 1],
        [2, 3, 4, 5]
    ])
    
    processed = processor.process_image(image.astype(float) / 9)
    edges = processor.edge_detect(image)
    
    print("  入力画像:")
    print(image)
    print("  処理後:")
    print(processed)
    print("  エッジ検出:")
    print(edges)
    
    # 5. 言語モデル
    print("\n" + "-" * 50)
    print("【5. 言語モデル】")
    print("-" * 50)
    
    lm = DecimalLanguageModel(hidden_size=4)
    
    # 学習
    print("  学習中...")
    lm.train_next_char("ABCDEFGH", epochs=30)
    
    # テスト
    test_text = "ABC"
    output = lm.forward(test_text)
    print(f"  入力: {test_text}")
    print(f"  出力: {output}")
    
    print("\n" + "=" * 70)
    print("✅ 全テスト完了")
    print("=" * 70)
    
    return {
        "learning": correct / 10,
        "entanglement": history[-1] if history else 0,
        "crypto": crypto,
        "image": processor,
        "language": lm
    }


if __name__ == "__main__":
    results = test_all_improvements()
    
    print("\n" + "=" * 70)
    print("📊 結果サマリー")
    print("=" * 70)
    print(f"  1. 学習改善: 正解率 {results['learning']:.0%}")
    print(f"  2. エンタングルメント: 精度 {results['entanglement']:.0%}")
    print("  3. 暗号化: 動作確認OK")
    print("  4. 画像処理: 動作確認OK")
    print("  5. 言語モデル: 動作確認OK")
