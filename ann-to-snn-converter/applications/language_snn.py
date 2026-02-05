"""
自律進化 言語SNN (Evolving Language SNN)
========================================

語彙を自動拡張し、文法を自己修正する自律進化SNN-LLM

Author: ろーる (cell_activation)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.evolution_engine import EvolvingSNN


class EvolvingLanguageSNN(EvolvingSNN):
    """
    自律進化する言語モデルSNN
    
    自動で:
    - 語彙を拡張
    - 文法エラーを自己修正
    - 表現力を向上
    """
    
    def __init__(self, n_neurons: int = 200):
        super().__init__(n_neurons)
        
        # 語彙
        self.vocabulary: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # 文法ルール（禁止パターン）
        self.forbidden_patterns = ["をと", "てもを", "がが", "はは"]
        
        # 学習した文
        self.learned_sentences: List[str] = []
        
        # スキル
        self.skills = {
            "vocabulary": 0.5,
            "grammar": 0.5,
            "fluency": 0.5
        }
    
    def add_word(self, word: str):
        """語彙に単語を追加"""
        if word not in self.vocabulary:
            self.vocabulary[word] = self.vocab_size
            self.id_to_word[self.vocab_size] = word
            self.vocab_size += 1
            
            # 自己成長を記録
            self.evolution.motivation.state.satisfaction += 0.01
    
    def learn_sentence(self, sentence: str):
        """文から学習"""
        # 単語を追加
        words = list(sentence)  # 文字単位
        for word in words:
            self.add_word(word)
        
        self.learned_sentences.append(sentence)
        
        # 文法チェック
        grammar_score = self._check_grammar(sentence)
        
        # 経験として記録
        input_vec = np.zeros(self.n_neurons)
        for i, char in enumerate(sentence[:self.n_neurons]):
            if char in self.vocabulary:
                input_vec[i] = self.vocabulary[char] / self.vocab_size
        
        self.experience(input_vec, skill="grammar", target=np.ones(self.n_neurons) * grammar_score)
    
    def generate(self, prompt: str = "", length: int = 20) -> str:
        """テキスト生成"""
        result = prompt
        
        # プロンプトをエンコード
        if prompt:
            input_vec = np.zeros(self.n_neurons)
            for i, char in enumerate(prompt[:self.n_neurons]):
                if char in self.vocabulary:
                    input_vec[i] = self.vocabulary[char] / max(1, self.vocab_size)
        else:
            input_vec = np.random.randn(self.n_neurons) * 0.3
        
        # 生成
        for _ in range(length):
            output = self.step(input_vec)
            
            # 最も活性化したニューロンから文字を選択
            if self.vocab_size > 0:
                idx = int(np.argmax(output[:self.vocab_size]) % self.vocab_size)
                char = self.id_to_word.get(idx, "")
                result += char
                
                # 入力を更新
                input_vec = np.roll(input_vec, -1)
                input_vec[-1] = idx / self.vocab_size
        
        # 文法フィルタ
        result = self._filter_grammar(result)
        
        return result
    
    def _check_grammar(self, text: str) -> float:
        """文法スコアを計算"""
        score = 1.0
        
        for pattern in self.forbidden_patterns:
            if pattern in text:
                score -= 0.1
        
        return max(0, score)
    
    def _filter_grammar(self, text: str) -> str:
        """禁止パターンをフィルタ"""
        for pattern in self.forbidden_patterns:
            text = text.replace(pattern, pattern[0])
        return text
    
    def evolve_vocabulary(self):
        """語彙を進化的に拡張"""
        # 好奇心が高い場合、新しい文字を探索
        if self.evolution.motivation.state.curiosity > 0.5:
            # ランダムな文字を追加（デモ用）
            new_chars = "あいうえお"
            for char in new_chars:
                self.add_word(char)
    
    def auto_learn_cycle(self, texts: List[str]):
        """自動学習サイクル"""
        print(f"\n📚 {len(texts)}個のテキストから学習中...")
        
        for text in texts:
            self.learn_sentence(text)
        
        print(f"  語彙サイズ: {self.vocab_size}")
        
        # 進化
        self.evolve(verbose=True)


def test_language_snn():
    """テスト"""
    print("\n" + "=" * 70)
    print("📝 自律進化 言語SNN テスト")
    print("=" * 70)
    
    snn = EvolvingLanguageSNN(n_neurons=100)
    
    # 学習
    training_texts = [
        "今日は天気がいい",
        "明日も晴れるといいな",
        "SNNで言語モデルを作る",
        "自律進化する人工知能",
    ]
    
    snn.auto_learn_cycle(training_texts)
    
    # 生成
    print("\n--- テキスト生成 ---")
    for prompt in ["今日", "SNN", ""]:
        generated = snn.generate(prompt, length=15)
        print(f"  プロンプト「{prompt}」→「{generated}」")
    
    # 自律運転
    print("\n--- 自律進化 ---")
    snn.run_autonomous(cycles=3, experience_per_cycle=10)
    
    print("\n" + "=" * 70)
    print("✅ テスト完了")
    print("=" * 70)


if __name__ == "__main__":
    test_language_snn()
