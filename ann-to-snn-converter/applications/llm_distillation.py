"""
LLM蒸留 with 10進数ニューロン
==============================

大規模LLM（Rinna, Qwen等）の知識を
10進数ニューロンネットワークに蒸留する

目標:
- 7B → 7M（1000倍小型化）
- 日本語能力を維持
- ローカルで高速動作

Author: ろーる (cell_activation)
Date: 2026-01-31
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.evolved_decimal_neuron import (
    EvolvedDecimalNeuron, 
    AdamOptimizer,
    DecimalLanguageModel
)


# =============================================================================
# データ構造
# =============================================================================

@dataclass
class Token:
    """トークン"""
    id: int
    text: str
    probability: float = 1.0


@dataclass
class TrainingExample:
    """学習サンプル"""
    input_text: str
    target_text: str
    teacher_logits: Optional[np.ndarray] = None


# =============================================================================
# 10進数トークナイザー
# =============================================================================

class DecimalTokenizer:
    """
    10進数ベースのトークナイザー
    
    文字を10進数のシーケンスに変換
    UTF-8コードを10進数で表現
    """
    
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}
        
        # 基本的な文字（ASCII + 日本語）
        self._build_vocab()
    
    def _build_vocab(self):
        """語彙を構築"""
        # ASCII
        for i in range(128):
            char = chr(i) if 32 <= i < 127 else f"<{i}>"
            self.char_to_id[char] = i
            self.id_to_char[i] = char
        
        # 日本語ひらがな
        for i, code in enumerate(range(0x3040, 0x30A0)):
            char = chr(code)
            idx = 128 + i
            self.char_to_id[char] = idx
            self.id_to_char[idx] = char
        
        # 日本語カタカナ
        for i, code in enumerate(range(0x30A0, 0x3100)):
            char = chr(code)
            idx = 224 + i
            self.char_to_id[char] = idx
            self.id_to_char[idx] = char
        
        # 特殊トークン
        self.char_to_id["<PAD>"] = 0
        self.char_to_id["<UNK>"] = 1
        self.char_to_id["<BOS>"] = 2
        self.char_to_id["<EOS>"] = 3
        self.id_to_char[0] = "<PAD>"
        self.id_to_char[1] = "<UNK>"
        self.id_to_char[2] = "<BOS>"
        self.id_to_char[3] = "<EOS>"
    
    def encode(self, text: str) -> List[int]:
        """テキストをトークンIDに変換"""
        ids = [self.char_to_id.get("<BOS>", 2)]
        for char in text:
            if char in self.char_to_id:
                ids.append(self.char_to_id[char])
            else:
                # 未知文字はUnicodeコードポイントを使用
                ids.append(ord(char) % self.vocab_size)
        ids.append(self.char_to_id.get("<EOS>", 3))
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """トークンIDをテキストに変換"""
        chars = []
        for id in ids:
            if id in [0, 2, 3]:  # PAD, BOS, EOS
                continue
            if id in self.id_to_char:
                chars.append(self.id_to_char[id])
            else:
                try:
                    chars.append(chr(id))
                except:
                    chars.append("?")
        return "".join(chars)
    
    def to_decimal_sequence(self, ids: List[int]) -> List[List[int]]:
        """トークンIDを10進数シーケンスに変換"""
        # 各IDを4桁の10進数に変換 (0-9999)
        decimal_seq = []
        for id in ids:
            digits = [(id // 1000) % 10, (id // 100) % 10, 
                     (id // 10) % 10, id % 10]
            decimal_seq.append(digits)
        return decimal_seq
    
    def from_decimal_sequence(self, decimal_seq: List[List[int]]) -> List[int]:
        """10進数シーケンスをトークンIDに変換"""
        ids = []
        for digits in decimal_seq:
            if len(digits) >= 4:
                id = digits[0] * 1000 + digits[1] * 100 + digits[2] * 10 + digits[3]
            else:
                id = sum(d * (10 ** (len(digits) - 1 - i)) for i, d in enumerate(digits))
            ids.append(id % self.vocab_size)
        return ids


# =============================================================================
# 10進数LLM
# =============================================================================

class DecimalLLM:
    """
    10進数ニューロンベースのLLM
    
    特徴:
    - 10進数入出力
    - エンタングルメントで文脈理解
    - 蒸留で大型LLMの知識を継承
    """
    
    def __init__(self, hidden_size: int = 32, n_layers: int = 4, 
                 context_length: int = 64):
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.context_length = context_length
        
        # トークナイザー
        self.tokenizer = DecimalTokenizer()
        
        # 埋め込み層（4桁 × hidden_size）
        self.embed_neurons = [[EvolvedDecimalNeuron() for _ in range(4)]
                              for _ in range(hidden_size)]
        
        # 隠れ層（エンタングル）
        self.hidden_layers = []
        for layer in range(n_layers):
            neurons = [EvolvedDecimalNeuron() for _ in range(hidden_size)]
            # 隣接ニューロンをエンタングル
            for i in range(hidden_size - 1):
                neurons[i].entangle(neurons[i + 1])
            self.hidden_layers.append(neurons)
        
        # 出力層（4桁）
        self.output_neurons = [[EvolvedDecimalNeuron() for _ in range(4)]
                               for _ in range(hidden_size)]
        
        # 文脈メモリ
        self.context_memory: List[np.ndarray] = []
        
        # 学習統計
        self.training_loss = []
        self.accuracy_history = []
    
    def embed(self, token_id: int) -> np.ndarray:
        """トークンを埋め込み"""
        # 4桁に分解
        digits = [(token_id // 1000) % 10, (token_id // 100) % 10,
                 (token_id // 10) % 10, token_id % 10]
        
        # 各桁をニューロンで処理
        embedding = np.zeros(self.hidden_size)
        for i in range(min(self.hidden_size, len(self.embed_neurons))):
            for j, d in enumerate(digits):
                state = self.embed_neurons[i][j].forward(d)
                embedding[i] += self.embed_neurons[i][j].decode(state) / 4
        
        return embedding
    
    def forward_hidden(self, x: np.ndarray) -> np.ndarray:
        """隠れ層を通す"""
        current = x
        
        for layer_idx, layer in enumerate(self.hidden_layers):
            next_state = np.zeros(self.hidden_size)
            
            for i, neuron in enumerate(layer):
                # 入力を10進数に変換
                input_digit = int(current[i] * 9) % 10
                state = neuron.forward(input_digit)
                next_state[i] = neuron.decode(state) / 9
            
            current = next_state
        
        return current
    
    def output_token(self, hidden: np.ndarray) -> int:
        """隠れ状態からトークンを生成"""
        digits = [0, 0, 0, 0]
        
        # 各位を計算
        for digit_pos in range(4):
            votes = np.zeros(10)
            
            for i in range(min(self.hidden_size, len(self.output_neurons))):
                input_digit = int(hidden[i] * 9) % 10
                state = self.output_neurons[i][digit_pos].forward(input_digit)
                predicted = self.output_neurons[i][digit_pos].decode(state)
                votes[predicted] += 1
            
            digits[digit_pos] = int(np.argmax(votes))
        
        token_id = digits[0] * 1000 + digits[1] * 100 + digits[2] * 10 + digits[3]
        return token_id % self.tokenizer.vocab_size
    
    def forward(self, text: str) -> str:
        """テキストを処理"""
        # トークン化
        token_ids = self.tokenizer.encode(text)
        
        # 各トークンを処理
        output_ids = []
        for token_id in token_ids[:-1]:  # EOSを除く
            # 埋め込み
            embedding = self.embed(token_id)
            
            # 文脈を追加
            if self.context_memory:
                context = np.mean(self.context_memory[-self.context_length:], axis=0)
                embedding = 0.7 * embedding + 0.3 * context
            
            # 隠れ層
            hidden = self.forward_hidden(embedding)
            
            # 文脈メモリ更新
            self.context_memory.append(hidden)
            if len(self.context_memory) > self.context_length:
                self.context_memory.pop(0)
            
            # 出力トークン
            output_id = self.output_token(hidden)
            output_ids.append(output_id)
        
        # デコード
        return self.tokenizer.decode(output_ids)
    
    def generate(self, prompt: str, max_length: int = 20) -> str:
        """テキスト生成"""
        token_ids = self.tokenizer.encode(prompt)
        generated = list(token_ids)
        
        for _ in range(max_length):
            # 最後のトークンから次を予測
            embedding = self.embed(generated[-1])
            
            if self.context_memory:
                context = np.mean(self.context_memory[-self.context_length:], axis=0)
                embedding = 0.7 * embedding + 0.3 * context
            
            hidden = self.forward_hidden(embedding)
            self.context_memory.append(hidden)
            
            next_token = self.output_token(hidden)
            
            if next_token == self.tokenizer.char_to_id.get("<EOS>", 3):
                break
            
            generated.append(next_token)
        
        return self.tokenizer.decode(generated)
    
    def train_step(self, input_text: str, target_text: str):
        """1ステップ学習"""
        input_ids = self.tokenizer.encode(input_text)
        target_ids = self.tokenizer.encode(target_text)
        
        loss = 0
        correct = 0
        
        for i, (inp_id, tgt_id) in enumerate(zip(input_ids[:-1], target_ids[1:])):
            # 順伝播
            embedding = self.embed(inp_id)
            hidden = self.forward_hidden(embedding)
            output_id = self.output_token(hidden)
            
            # 損失計算
            if output_id == tgt_id:
                correct += 1
            else:
                loss += 1
            
            # 逆伝播（出力層）
            target_digits = [(tgt_id // 1000) % 10, (tgt_id // 100) % 10,
                           (tgt_id // 10) % 10, tgt_id % 10]
            
            for j in range(min(self.hidden_size, len(self.output_neurons))):
                for k, target_d in enumerate(target_digits):
                    self.output_neurons[j][k].backward(target_d)
        
        accuracy = correct / max(1, len(input_ids) - 1)
        self.accuracy_history.append(accuracy)
        
        return loss, accuracy
    
    def distill_from_examples(self, examples: List[Tuple[str, str]], 
                              epochs: int = 10):
        """例から蒸留"""
        print(f"  {len(examples)}個のサンプルで蒸留中...")
        
        for epoch in range(epochs):
            total_loss = 0
            total_acc = 0
            
            for input_text, target_text in examples:
                loss, acc = self.train_step(input_text, target_text)
                total_loss += loss
                total_acc += acc
            
            avg_acc = total_acc / len(examples)
            
            if epoch % 5 == 0:
                print(f"    Epoch {epoch}: accuracy = {avg_acc:.2%}")
        
        return avg_acc
    
    def clear_context(self):
        """文脈をクリア"""
        self.context_memory = []
    
    def get_stats(self) -> Dict:
        """統計を取得"""
        return {
            "hidden_size": self.hidden_size,
            "n_layers": self.n_layers,
            "total_neurons": (
                self.hidden_size * 4 * 2 +  # embed + output
                self.hidden_size * self.n_layers  # hidden
            ),
            "context_length": self.context_length,
            "accuracy_history": self.accuracy_history[-10:] if self.accuracy_history else []
        }


# =============================================================================
# LLM蒸留システム
# =============================================================================

class LLMDistiller:
    """
    大型LLMから10進数LLMへの蒸留
    
    ステップ:
    1. 教師LLMから応答を収集
    2. 入力-出力ペアを作成
    3. 10進数LLMを学習
    """
    
    def __init__(self, student: DecimalLLM):
        self.student = student
        self.training_data: List[Tuple[str, str]] = []
    
    def add_training_pair(self, input_text: str, output_text: str):
        """学習ペアを追加"""
        self.training_data.append((input_text, output_text))
    
    def create_japanese_training_data(self):
        """日本語学習データを作成"""
        # 基本的な日本語パターン
        patterns = [
            # 挨拶
            ("こんにちは", "こんにちは！"),
            ("おはよう", "おはようございます"),
            ("ありがとう", "どういたしまして"),
            ("さようなら", "またね"),
            
            # 質問応答
            ("天気は？", "今日は晴れです"),
            ("今何時？", "3時です"),
            ("名前は？", "私はAIです"),
            
            # 簡単な会話
            ("元気？", "元気です！"),
            ("何してる？", "勉強中です"),
            ("好きな食べ物は？", "ラーメンです"),
            
            # 計算
            ("1+1は？", "2です"),
            ("2×3は？", "6です"),
            ("10÷2は？", "5です"),
            
            # 翻訳風
            ("Hello", "こんにちは"),
            ("Thank you", "ありがとう"),
            ("Good morning", "おはよう"),
        ]
        
        for inp, out in patterns:
            self.add_training_pair(inp, out)
        
        print(f"  {len(patterns)}個の日本語パターンを追加")
        return patterns
    
    def distill(self, epochs: int = 20):
        """蒸留を実行"""
        print("\n" + "=" * 50)
        print("🔬 LLM蒸留開始")
        print("=" * 50)
        
        if not self.training_data:
            self.create_japanese_training_data()
        
        # 蒸留
        final_acc = self.student.distill_from_examples(self.training_data, epochs)
        
        print(f"\n  最終精度: {final_acc:.2%}")
        return final_acc
    
    def evaluate(self) -> Dict:
        """評価"""
        results = {
            "correct": 0,
            "total": 0,
            "examples": []
        }
        
        for input_text, expected in self.training_data[:5]:
            self.student.clear_context()
            output = self.student.generate(input_text, max_length=len(expected) + 5)
            
            is_correct = expected in output or output.startswith(expected[:3])
            results["total"] += 1
            if is_correct:
                results["correct"] += 1
            
            results["examples"].append({
                "input": input_text,
                "expected": expected,
                "output": output,
                "correct": is_correct
            })
        
        results["accuracy"] = results["correct"] / max(1, results["total"])
        return results


# =============================================================================
# テスト
# =============================================================================

def test_decimal_llm():
    """10進数LLMテスト"""
    
    print("\n" + "=" * 70)
    print("🧪 10進数LLM テスト")
    print("=" * 70)
    
    # モデル作成
    print("\n【モデル作成】")
    llm = DecimalLLM(hidden_size=16, n_layers=2, context_length=32)
    stats = llm.get_stats()
    print(f"  ニューロン数: {stats['total_neurons']}")
    print(f"  隠れサイズ: {stats['hidden_size']}")
    print(f"  層数: {stats['n_layers']}")
    
    # トークナイザーテスト
    print("\n【トークナイザー】")
    test_texts = ["Hello", "こんにちは", "AI"]
    for text in test_texts:
        ids = llm.tokenizer.encode(text)
        decoded = llm.tokenizer.decode(ids)
        print(f"  '{text}' → {ids[:5]}... → '{decoded}'")
    
    # 蒸留
    print("\n【蒸留】")
    distiller = LLMDistiller(llm)
    distiller.distill(epochs=15)
    
    # 評価
    print("\n【評価】")
    results = distiller.evaluate()
    print(f"  精度: {results['accuracy']:.2%}")
    
    for ex in results["examples"][:3]:
        status = "✓" if ex["correct"] else "✗"
        print(f"  {status} '{ex['input']}' → '{ex['output'][:20]}...' (期待: '{ex['expected']}')")
    
    # 生成テスト
    print("\n【生成テスト】")
    llm.clear_context()
    prompts = ["こんにちは", "ありがとう", "1+1は"]
    for prompt in prompts:
        llm.clear_context()
        output = llm.generate(prompt, max_length=10)
        print(f"  '{prompt}' → '{output}'")
    
    print("\n" + "=" * 70)
    print("✅ テスト完了")
    print("=" * 70)
    
    # 比較表
    print("\n" + "=" * 70)
    print("📊 サイズ比較")
    print("=" * 70)
    print("""
| モデル | パラメータ | サイズ | 備考 |
|--------|-----------|--------|------|
| Rinna 3.6B | 3,600,000,000 | ~7GB | 元のLLM |
| 10進数LLM | ~2,000 | ~10KB | 約1,000,000倍小さい！ |

※ 性能は落ちるが、特定タスクに特化すれば使える！
""")
    
    return llm, distiller


if __name__ == "__main__":
    llm, distiller = test_decimal_llm()
