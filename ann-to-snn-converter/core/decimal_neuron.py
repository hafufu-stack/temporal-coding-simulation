"""
10進数ニューロン (Decimal Neuron)
==================================

量子コンピュータ + DNN + SNN の良いところを取った新素子

特徴:
1. 10進数入出力 (0-9) - 人間に分かりやすい
2. 重ね合わせ状態 - 量子的な確率分布
3. スパイク時間符号化 - SNNの効率性
4. 勾配学習可能 - DNNの学習能力

Author: ろーる (cell_activation)
Date: 2026-01-31
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import matplotlib.pyplot as plt


# =============================================================================
# 10進数ニューロン
# =============================================================================

class DecimalNeuron:
    """
    10進数ニューロン
    
    入力: 0-9 の整数、または確率分布
    出力: 0-9 の整数、または確率分布
    
    量子的性質:
    - 重ね合わせ状態（10個の状態を同時に保持）
    - 測定時に1つに収束
    
    SNN的性質:
    - スパイクタイミングで情報を符号化
    - エネルギー効率が高い
    
    DNN的性質:
    - 勾配で学習可能
    - バックプロパゲーション対応
    """
    
    def __init__(self, n_digits: int = 10):
        self.n_digits = n_digits  # 通常は10（0-9）
        
        # 重ね合わせ状態（確率振幅）
        self.state = np.ones(n_digits) / n_digits  # 初期は均等
        
        # 重み（各入力数字から各出力数字への変換）
        self.W = np.eye(n_digits) + np.random.randn(n_digits, n_digits) * 0.1
        
        # バイアス
        self.bias = np.zeros(n_digits)
        
        # スパイクタイミング履歴
        self.spike_times: List[Tuple[int, float]] = []
        
        # 学習用
        self.grad_W = np.zeros_like(self.W)
        self.grad_bias = np.zeros_like(self.bias)
    
    def encode_decimal(self, digit: int) -> np.ndarray:
        """10進数を確率分布に変換"""
        if not 0 <= digit <= 9:
            raise ValueError(f"数字は0-9である必要があります: {digit}")
        
        # One-hot風だが、少しぼやかす（量子的揺らぎ）
        state = np.ones(self.n_digits) * 0.01
        state[digit] = 0.91
        return state / state.sum()
    
    def decode_decimal(self, state: np.ndarray) -> int:
        """確率分布を10進数に変換（測定）"""
        # 確率的に選択（量子測定）
        probs = np.abs(state)
        probs = probs / probs.sum()
        return np.random.choice(self.n_digits, p=probs)
    
    def decode_deterministic(self, state: np.ndarray) -> int:
        """確定的に最大確率を選択"""
        return np.argmax(state)
    
    def forward(self, input_digit: int) -> np.ndarray:
        """
        順伝播
        
        入力: 10進数 (0-9)
        出力: 出力状態（確率分布）
        """
        # 入力を状態に変換
        input_state = self.encode_decimal(input_digit)
        
        # 重み行列で変換（DNN的）
        output_state = self.W @ input_state + self.bias
        
        # ソフトマックス（確率に正規化）
        exp_state = np.exp(output_state - np.max(output_state))
        self.state = exp_state / exp_state.sum()
        
        # スパイクタイミングを記録（SNN的）
        # 確率が高いほど早くスパイク
        for i in range(self.n_digits):
            spike_time = 1.0 - self.state[i]  # 確率高い = 早い
            self.spike_times.append((i, spike_time))
        
        return self.state
    
    def measure(self) -> int:
        """量子測定（確率的に1つの値に収束）"""
        return self.decode_decimal(self.state)
    
    def backward(self, target_digit: int, learning_rate: float = 0.1):
        """逆伝播（学習）"""
        target = np.zeros(self.n_digits)
        target[target_digit] = 1.0
        
        # クロスエントロピー勾配
        grad = self.state - target
        
        # 重み更新
        self.W -= learning_rate * np.outer(grad, self.state)
        self.bias -= learning_rate * grad
    
    def __repr__(self):
        return f"DecimalNeuron(state={self.decode_deterministic(self.state)})"


# =============================================================================
# 10進数ニューロン層
# =============================================================================

class DecimalLayer:
    """10進数ニューロンの層"""
    
    def __init__(self, n_neurons: int):
        self.n_neurons = n_neurons
        self.neurons = [DecimalNeuron() for _ in range(n_neurons)]
    
    def forward(self, inputs: List[int]) -> List[np.ndarray]:
        """層全体の順伝播"""
        if len(inputs) != self.n_neurons:
            raise ValueError(f"入力数が一致しません: {len(inputs)} != {self.n_neurons}")
        
        outputs = []
        for i, neuron in enumerate(self.neurons):
            output = neuron.forward(inputs[i])
            outputs.append(output)
        
        return outputs
    
    def measure_all(self) -> List[int]:
        """全ニューロンを測定"""
        return [n.measure() for n in self.neurons]


# =============================================================================
# 10進数ニューラルネットワーク
# =============================================================================

class DecimalNeuralNetwork:
    """
    10進数ニューラルネットワーク
    
    各層が10進数を入出力
    量子 + SNN + DNN のハイブリッド
    """
    
    def __init__(self, layer_sizes: List[int]):
        """
        layer_sizes: 各層のニューロン数
        例: [4, 8, 4] = 入力4桁、隠れ層8、出力4桁
        """
        self.layer_sizes = layer_sizes
        self.layers: List[DecimalLayer] = []
        
        # 層間の接続重み
        self.inter_layer_weights: List[np.ndarray] = []
        
        for i in range(len(layer_sizes) - 1):
            # 各出力から各入力への変換行列
            W = np.random.randn(layer_sizes[i+1], layer_sizes[i], 10, 10) * 0.1
            self.inter_layer_weights.append(W)
        
        for size in layer_sizes:
            self.layers.append(DecimalLayer(size))
    
    def forward(self, input_digits: List[int]) -> List[int]:
        """順伝播"""
        current = input_digits
        
        for layer_idx, layer in enumerate(self.layers):
            # 層を通す
            if layer_idx == 0:
                # 入力層はそのまま
                states = layer.forward(current)
            else:
                # 前層の出力を次層の入力に変換
                prev_outputs = [self.layers[layer_idx-1].neurons[i].measure() 
                               for i in range(len(self.layers[layer_idx-1].neurons))]
                
                # 次層の入力を計算
                next_inputs = []
                for j in range(layer.n_neurons):
                    # 前層の全ニューロンからの入力を集約
                    aggregated = np.zeros(10)
                    for i, prev_out in enumerate(prev_outputs):
                        if i < self.inter_layer_weights[layer_idx-1].shape[1]:
                            W = self.inter_layer_weights[layer_idx-1][min(j, self.inter_layer_weights[layer_idx-1].shape[0]-1), i]
                            aggregated += W[prev_out]
                    
                    next_inputs.append(int(np.argmax(aggregated)))
                
                states = layer.forward(next_inputs)
            
            current = layer.measure_all()
        
        return current
    
    def train(self, inputs: List[List[int]], targets: List[List[int]], 
              epochs: int = 100, learning_rate: float = 0.1):
        """学習"""
        history = []
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            
            for input_digits, target_digits in zip(inputs, targets):
                # 順伝播
                outputs = self.forward(input_digits)
                
                # 損失計算
                for i, (out, target) in enumerate(zip(outputs, target_digits)):
                    if out == target:
                        correct += 1
                
                # 逆伝播（出力層のみ簡易版）
                for i, neuron in enumerate(self.layers[-1].neurons):
                    if i < len(target_digits):
                        neuron.backward(target_digits[i], learning_rate)
            
            accuracy = correct / (len(inputs) * len(inputs[0]))
            history.append(accuracy)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: accuracy = {accuracy:.2%}")
        
        return history


# =============================================================================
# 量子的10進数ニューロン（拡張版）
# =============================================================================

class QuantumDecimalNeuron(DecimalNeuron):
    """
    量子的性質を強化した10進数ニューロン
    
    追加機能:
    - エンタングルメント（他のニューロンとの絡み合い）
    - 干渉効果
    """
    
    def __init__(self, n_digits: int = 10):
        super().__init__(n_digits)
        
        # 位相（量子的）
        self.phase = np.zeros(n_digits)
        
        # エンタングル済みニューロン
        self.entangled_with: List['QuantumDecimalNeuron'] = []
    
    def entangle(self, other: 'QuantumDecimalNeuron'):
        """他のニューロンとエンタングルする"""
        self.entangled_with.append(other)
        other.entangled_with.append(self)
    
    def forward_quantum(self, input_digit: int) -> np.ndarray:
        """量子的順伝播"""
        # 通常の順伝播
        output = self.forward(input_digit)
        
        # 位相を更新
        self.phase = np.angle(self.state + 1j * np.random.randn(self.n_digits) * 0.1)
        
        # エンタングルしたニューロンの影響
        for other in self.entangled_with:
            # 干渉効果
            interference = np.cos(self.phase - other.phase)
            self.state = self.state * (1 + 0.1 * interference)
            self.state = self.state / self.state.sum()
        
        return self.state
    
    def superposition_input(self, digits: List[int], weights: List[float] = None):
        """複数の数字を重ね合わせ入力"""
        if weights is None:
            weights = [1.0 / len(digits)] * len(digits)
        
        total_state = np.zeros(self.n_digits)
        for digit, weight in zip(digits, weights):
            total_state += weight * self.encode_decimal(digit)
        
        self.state = total_state / total_state.sum()
        return self.state


# =============================================================================
# テスト
# =============================================================================

def test_decimal_neuron():
    """10進数ニューロンのテスト"""
    
    print("\n" + "=" * 70)
    print("🧪 10進数ニューロン テスト")
    print("=" * 70)
    
    # 基本テスト
    print("\n【基本テスト】")
    neuron = DecimalNeuron()
    
    for digit in [0, 5, 9]:
        output = neuron.forward(digit)
        measured = neuron.measure()
        deterministic = neuron.decode_deterministic(output)
        print(f"  入力={digit} → 確定={deterministic}, 測定={measured}")
    
    # 学習テスト
    print("\n【学習テスト】+1を学習")
    neuron = DecimalNeuron()
    
    # +1 の変換を学習 (0→1, 1→2, ..., 9→0)
    for epoch in range(50):
        for digit in range(10):
            neuron.forward(digit)
            target = (digit + 1) % 10
            neuron.backward(target, learning_rate=0.2)
    
    print("  学習後:")
    for digit in [0, 3, 8]:
        neuron.forward(digit)
        result = neuron.decode_deterministic(neuron.state)
        expected = (digit + 1) % 10
        print(f"    {digit} + 1 = {result} (期待: {expected}) {'✓' if result == expected else '✗'}")
    
    # 量子的テスト
    print("\n【量子的テスト】")
    q_neuron = QuantumDecimalNeuron()
    
    # 重ね合わせ入力
    q_neuron.superposition_input([3, 7], weights=[0.5, 0.5])
    print(f"  重ね合わせ(3, 7) → 状態: {q_neuron.state[:4].round(2)}...")
    
    # 測定（確率的）
    measurements = [q_neuron.measure() for _ in range(100)]
    print(f"  100回測定: 3の頻度={measurements.count(3)}, 7の頻度={measurements.count(7)}")
    
    # ネットワークテスト
    print("\n【ネットワークテスト】2桁の足し算")
    
    # 2桁入力 → 2桁出力 のネットワーク
    network = DecimalNeuralNetwork([2, 4, 2])
    
    # テストデータ: 単純な足し算
    inputs = [[1, 2], [3, 4], [5, 5], [0, 9]]
    targets = [[3, 0], [7, 0], [0, 1], [9, 0]]  # 各桁の和
    
    print("  訓練前:")
    for inp, tgt in zip(inputs[:2], targets[:2]):
        out = network.forward(inp)
        print(f"    {inp[0]}+{inp[1]} = {out[0]} (期待: {tgt[0]})")
    
    print("\n  訓練中...")
    network.train(inputs, targets, epochs=50, learning_rate=0.3)
    
    print("\n  訓練後:")
    for inp, tgt in zip(inputs, targets):
        out = network.forward(inp)
        print(f"    {inp[0]}+{inp[1]} = {out[0]} (期待: {tgt[0]})")
    
    print("\n" + "=" * 70)
    print("✅ テスト完了")
    print("=" * 70)


def demo_hybrid_comparison():
    """量子・DNN・SNNとの比較デモ"""
    
    print("\n" + "=" * 70)
    print("📊 ハイブリッド素子の比較")
    print("=" * 70)
    
    print("""
【従来の素子】

| 素子タイプ | 入出力 | 状態 | 学習 | 効率 |
|-----------|--------|------|------|------|
| Qubit     | 0/1    | 重ね合わせ | 量子ゲート | 高電力 |
| DNN       | 実数   | 連続値 | 勾配 | GPU依存 |
| SNN       | 0/1    | スパイク | STDP | 低電力 |

【新素子: 10進数ニューロン】

| 特徴 | 値 |
|------|-----|
| 入出力 | 0-9（人間可読！） |
| 状態 | 10状態の重ね合わせ |
| 学習 | 勾配（DNN的） |
| 効率 | 低電力（SNN的） |
| 特殊能力 | エンタングルメント（量子的） |

【メリット】
1. 人間が直感的に理解できる（0-9）
2. 確率的推論ができる（量子的重ね合わせ）
3. エネルギー効率が高い（スパイク符号化）
4. 勾配で学習できる（DNNの強み）
5. ニューロン間の絡み合いが可能（量子エンタングル）
""")


if __name__ == "__main__":
    demo_hybrid_comparison()
    test_decimal_neuron()
