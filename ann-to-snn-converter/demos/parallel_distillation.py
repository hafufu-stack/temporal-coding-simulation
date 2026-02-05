"""
並列処理版 10進数ニューロンLLM蒸留 (シミュレーション版)
======================================================

改善点:
- マルチスレッドで並列学習
- CPU使用率を最大化
- 早期停止で過学習防止

Author: ろーる (cell_activation)  
Date: 2026-02-01
"""

import os
import sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# 並列化された10進数ニューロン
# =============================================================================

class ParallelDecimalNeuron:
    """並列処理対応の10進数ニューロン"""
    
    def __init__(self, n_digits: int = 10, learning_rate: float = 0.01):
        self.n_digits = n_digits
        self.learning_rate = learning_rate
        self.lock = threading.Lock()
        
        # パラメータ
        self.W = np.eye(n_digits) * 0.5 + np.random.randn(n_digits, n_digits) * 0.05
        self.bias = np.zeros(n_digits)
        self.state = np.ones(n_digits) / n_digits
        
        # モメンタム
        self.momentum_W = np.zeros_like(self.W)
        self.momentum_bias = np.zeros_like(self.bias)
    
    def encode(self, digit: int) -> np.ndarray:
        state = np.zeros(self.n_digits)
        state[digit % self.n_digits] = 1.0
        return state
    
    def forward(self, input_digit: int) -> int:
        input_state = self.encode(input_digit)
        output = self.W @ input_state + self.bias
        exp_out = np.exp(output - np.max(output))
        self.state = exp_out / exp_out.sum()
        return int(np.argmax(self.state))
    
    def backward(self, target: int, momentum: float = 0.9):
        with self.lock:
            target_state = self.encode(target)
            grad = self.state - target_state
            
            self.momentum_W = momentum * self.momentum_W + self.learning_rate * np.outer(grad, self.state)
            self.momentum_bias = momentum * self.momentum_bias + self.learning_rate * grad
            
            self.W -= self.momentum_W
            self.bias -= self.momentum_bias


def process_layer_parallel(neurons, inputs, n_workers):
    """層を並列処理"""
    def process_single(args):
        idx, neuron, inp = args
        return neuron.forward(inp)
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        args_list = [(i, neurons[i], inputs[i % len(inputs)]) 
                     for i in range(len(neurons))]
        results = list(executor.map(process_single, args_list))
    
    return results


class ParallelDecimalLLM:
    """並列処理対応の10進数LLM"""
    
    def __init__(self, hidden_size: int = 64, n_layers: int = 4, 
                 learning_rate: float = 0.01):
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_workers = min(cpu_count(), hidden_size)
        
        # 層を作成
        self.embed = [ParallelDecimalNeuron(learning_rate=learning_rate) 
                      for _ in range(hidden_size)]
        self.hidden = [[ParallelDecimalNeuron(learning_rate=learning_rate) 
                        for _ in range(hidden_size)] 
                       for _ in range(n_layers)]
        self.output = [ParallelDecimalNeuron(learning_rate=learning_rate) 
                       for _ in range(hidden_size)]
        
        # 語彙
        self._build_vocab()
    
    def _build_vocab(self):
        self.vocab = {}
        self.inv_vocab = {}
        chars = [chr(i) for i in range(32, 127)]
        chars.extend([chr(i) for i in range(0x3040, 0x30A0)])
        for i, c in enumerate(chars):
            self.vocab[c] = i % 1000
            self.inv_vocab[i % 1000] = c
    
    def char_to_id(self, char: str) -> int:
        return self.vocab.get(char, ord(char) % 1000)
    
    def id_to_char(self, id: int) -> str:
        return self.inv_vocab.get(id % 1000, '?')
    
    def forward_parallel(self, input_ids: list) -> list:
        """並列順伝播"""
        output_ids = []
        
        for input_id in input_ids:
            # 入力を桁に分解
            digits = [(input_id // (10 ** i)) % 10 for i in range(3)]
            
            # 埋め込み層（並列）
            embed_out = process_layer_parallel(
                self.embed, digits, self.n_workers
            )
            
            # 隠れ層（並列）
            current = embed_out
            for layer in self.hidden:
                current = process_layer_parallel(layer, current, self.n_workers)
            
            # 出力層（並列）
            out = process_layer_parallel(self.output, current, self.n_workers)
            
            # 出力ID
            output_id = sum(out[i] * (10 ** i) for i in range(min(3, len(out))))
            output_ids.append(output_id % 1000)
        
        return output_ids
    
    def forward(self, text: str) -> str:
        input_ids = [self.char_to_id(c) for c in text]
        output_ids = self.forward_parallel(input_ids)
        return ''.join(self.id_to_char(id) for id in output_ids)
    
    def train_parallel(self, batch: list):
        """並列学習"""
        def train_single(item):
            input_text, target_text = item
            input_ids = [self.char_to_id(c) for c in input_text]
            target_ids = [self.char_to_id(c) for c in target_text]
            
            output_ids = self.forward_parallel(input_ids)
            
            # 学習
            for i, tgt in enumerate(target_ids[:len(output_ids)]):
                for j in range(min(3, len(self.output))):
                    target_digit = (tgt // (10 ** j)) % 10
                    self.output[j].backward(target_digit)
            
            correct = sum(1 for o, t in zip(output_ids, target_ids) if o == t)
            return correct, len(target_ids)
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            results = list(executor.map(train_single, batch))
        
        total_correct = sum(r[0] for r in results)
        total_count = sum(r[1] for r in results)
        
        return total_correct / max(1, total_count)
    
    def get_stats(self):
        return {
            "total_neurons": len(self.embed) + sum(len(l) for l in self.hidden) + len(self.output),
            "hidden_size": self.hidden_size,
            "n_layers": self.n_layers,
            "n_workers": self.n_workers
        }


# =============================================================================
# メイン
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("⚡ 並列処理版 10進数ニューロンLLM蒸留")
    print("=" * 70)
    
    print(f"\n【システム情報】")
    print(f"  CPU数: {cpu_count()}")
    
    # モデル作成
    print("\n【モデル作成】")
    model = ParallelDecimalLLM(hidden_size=64, n_layers=4)
    stats = model.get_stats()
    print(f"  ニューロン数: {stats['total_neurons']}")
    print(f"  隠れサイズ: {stats['hidden_size']}")
    print(f"  層数: {stats['n_layers']}")
    print(f"  並列ワーカー: {stats['n_workers']}")
    
    # シミュレーション学習データ
    print("\n【学習データ】")
    training_data = [
        ("こんにちは", "こんにちは！"),
        ("おはよう", "おはようございます"),
        ("ありがとう", "どういたしまして"),
        ("さようなら", "またね"),
        ("元気？", "元気です！"),
        ("1+1は？", "2です"),
        ("2+2は？", "4です"),
        ("天気は？", "晴れです"),
        ("名前は？", "AIです"),
        ("Hello", "Hi there!"),
        ("調子どう？", "いい感じ！"),
        ("何してる？", "学習中です"),
        ("暇？", "いいえ"),
        ("忙しい？", "はい"),
        ("眠い", "休んでください"),
        ("疲れた", "お疲れ様です"),
        ("嬉しい", "よかったね！"),
        ("悲しい", "大丈夫？"),
        ("楽しい", "いいね！"),
        ("すごい", "ありがとう！"),
    ]
    print(f"  {len(training_data)}個のサンプル")
    
    # 並列学習
    print("\n【並列学習】")
    start_time = time.time()
    
    best_acc = 0
    patience = 5
    no_improve = 0
    
    for epoch in range(50):
        np.random.shuffle(training_data)
        acc = model.train_parallel(training_data)
        
        if acc > best_acc:
            best_acc = acc
            no_improve = 0
        else:
            no_improve += 1
        
        if epoch % 10 == 0:
            print(f"    Epoch {epoch}: accuracy = {acc:.2%} (best: {best_acc:.2%})")
        
        if no_improve >= patience and epoch > 15:
            print(f"    早期停止 (改善なし {patience}エポック)")
            break
    
    elapsed = time.time() - start_time
    print(f"\n  ⏱ 学習時間: {elapsed:.1f}秒")
    print(f"  最高精度: {best_acc:.2%}")
    
    # 生成テスト
    print("\n【生成テスト】")
    test_prompts = ["こんにちは", "ありがとう", "Hello", "1+1"]
    for prompt in test_prompts:
        output = model.forward(prompt)
        print(f"  '{prompt}' → '{output}'")
    
    print("\n" + "=" * 70)
    print("✅ 並列蒸留完了！")
    print("=" * 70)
    
    return model


if __name__ == "__main__":
    main()
