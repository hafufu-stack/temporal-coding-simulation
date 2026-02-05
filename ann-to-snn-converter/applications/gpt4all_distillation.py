"""
GPT4All + 10進数ニューロン 蒸留統合
===================================

GPT4Allから10進数ニューロンLLMへの蒸留

特徴:
- GPT4Allの応答を教師データとして収集
- 10進数ニューロンに知識を蒸留
- ローカル完結（ネット不要）

Author: ろーる (cell_activation)
Date: 2026-01-31
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# GPT4Allが利用可能かチェック
try:
    from gpt4all import GPT4All
    GPT4ALL_AVAILABLE = True
except ImportError:
    GPT4ALL_AVAILABLE = False
    print("警告: GPT4Allがインストールされていません")
    print("  pip install gpt4all でインストールしてください")

from applications.llm_distillation import DecimalLLM, LLMDistiller


# =============================================================================
# GPT4All教師モデル
# =============================================================================

class GPT4AllTeacher:
    """
    GPT4Allを教師モデルとして使用
    """
    
    # 利用可能な日本語対応モデル
    JAPANESE_MODELS = [
        "Phi-3-mini-4k-instruct.Q4_0.gguf",  # 多言語対応
        "mistral-7b-instruct-v0.1.Q4_0.gguf",  # 多言語
        "orca-mini-3b-gguf2-q4_0.gguf",  # 軽量
    ]
    
    def __init__(self, model_name: str = None):
        if not GPT4ALL_AVAILABLE:
            raise RuntimeError("GPT4Allがインストールされていません")
        
        self.model_name = model_name
        self.model = None
        self.responses_cache: Dict[str, str] = {}
    
    def load_model(self, model_name: str = None):
        """モデルをロード"""
        if model_name:
            self.model_name = model_name
        
        if not self.model_name:
            # デフォルトで軽量モデルを使用
            self.model_name = "orca-mini-3b-gguf2-q4_0.gguf"
        
        print(f"  モデルをロード中: {self.model_name}")
        print("  (初回は自動ダウンロードされます)")
        
        try:
            self.model = GPT4All(self.model_name)
            print(f"  ✅ モデルロード完了")
            return True
        except Exception as e:
            print(f"  ❌ モデルロード失敗: {e}")
            return False
    
    def generate(self, prompt: str, max_tokens: int = 100, 
                 temperature: float = 0.7) -> str:
        """応答を生成"""
        if not self.model:
            return ""
        
        # キャッシュをチェック
        cache_key = f"{prompt}_{max_tokens}_{temperature}"
        if cache_key in self.responses_cache:
            return self.responses_cache[cache_key]
        
        try:
            response = self.model.generate(
                prompt,
                max_tokens=max_tokens,
                temp=temperature,
                top_p=0.9,
                repeat_penalty=1.1
            )
            
            # キャッシュに保存
            self.responses_cache[cache_key] = response
            return response
            
        except Exception as e:
            print(f"  生成エラー: {e}")
            return ""
    
    def generate_training_pairs(self, prompts: List[str]) -> List[Tuple[str, str]]:
        """学習用ペアを生成"""
        pairs = []
        
        for prompt in prompts:
            response = self.generate(prompt, max_tokens=50)
            if response:
                pairs.append((prompt, response))
        
        return pairs


# =============================================================================
# 蒸留パイプライン
# =============================================================================

class GPT4AllDistillationPipeline:
    """
    GPT4All → 10進数ニューロン 蒸留パイプライン
    """
    
    def __init__(self, student_hidden_size: int = 32, student_layers: int = 4):
        # 教師モデル
        self.teacher = None
        
        # 生徒モデル（10進数ニューロン）
        self.student = DecimalLLM(
            hidden_size=student_hidden_size,
            n_layers=student_layers,
            context_length=64
        )
        
        # 蒸留器
        self.distiller = LLMDistiller(self.student)
        
        # 学習データ
        self.training_data: List[Tuple[str, str]] = []
    
    def load_teacher(self, model_name: str = None) -> bool:
        """教師モデルをロード"""
        if not GPT4ALL_AVAILABLE:
            print("  GPT4Allが利用できません。シミュレーションモードで実行します。")
            return False
        
        self.teacher = GPT4AllTeacher(model_name)
        return self.teacher.load_model()
    
    def collect_training_data(self, prompts: List[str] = None):
        """教師から学習データを収集"""
        if prompts is None:
            # デフォルトの日本語プロンプト
            prompts = [
                "こんにちは",
                "おはようございます",
                "ありがとう",
                "今日の天気は？",
                "1+1は何？",
                "AIとは何ですか？",
                "日本の首都は？",
                "プログラミングとは？",
            ]
        
        print(f"\n  {len(prompts)}個のプロンプトから学習データを収集中...")
        
        if self.teacher and GPT4ALL_AVAILABLE:
            # GPT4Allから応答を収集
            self.training_data = self.teacher.generate_training_pairs(prompts)
        else:
            # シミュレーションデータ
            self.training_data = self._create_simulation_data(prompts)
        
        print(f"  ✅ {len(self.training_data)}個の学習ペアを収集")
        
        for inp, out in self.training_data[:3]:
            print(f"    '{inp}' → '{out[:30]}...'")
    
    def _create_simulation_data(self, prompts: List[str]) -> List[Tuple[str, str]]:
        """シミュレーション用データ（GPT4Allなしでもテスト可能）"""
        simulation_responses = {
            "こんにちは": "こんにちは！何かお手伝いできることはありますか？",
            "おはようございます": "おはようございます！良い一日を！",
            "ありがとう": "どういたしまして！",
            "今日の天気は？": "今日は晴れの予報です。",
            "1+1は何？": "1+1は2です。",
            "AIとは何ですか？": "AIは人工知能の略で、機械が知的なタスクを行う技術です。",
            "日本の首都は？": "日本の首都は東京です。",
            "プログラミングとは？": "プログラミングはコンピュータに命令を与える方法です。",
        }
        
        pairs = []
        for prompt in prompts:
            response = simulation_responses.get(prompt, f"{prompt}への応答")
            pairs.append((prompt, response))
        
        return pairs
    
    def distill(self, epochs: int = 20):
        """蒸留を実行"""
        print("\n" + "=" * 50)
        print("🔬 蒸留開始")
        print("=" * 50)
        
        # 学習データを追加
        for inp, out in self.training_data:
            self.distiller.add_training_pair(inp, out)
        
        # 蒸留実行
        final_acc = self.distiller.distill(epochs=epochs)
        
        return final_acc
    
    def evaluate(self):
        """評価"""
        print("\n【評価】")
        results = self.distiller.evaluate()
        print(f"  精度: {results['accuracy']:.2%}")
        
        for ex in results["examples"][:3]:
            status = "✓" if ex["correct"] else "✗"
            print(f"  {status} '{ex['input']}' → '{ex['output'][:25]}...'")
        
        return results
    
    def compare_sizes(self):
        """サイズ比較"""
        print("\n" + "=" * 50)
        print("📊 サイズ比較")
        print("=" * 50)
        
        teacher_params = 3_000_000_000  # 3B（推定）
        student_stats = self.student.get_stats()
        student_params = student_stats["total_neurons"]
        
        compression = teacher_params / max(1, student_params)
        
        print(f"""
| モデル | パラメータ | 推定サイズ |
|--------|-----------|-----------|
| GPT4All (3B) | {teacher_params:,} | ~6GB |
| 10進数LLM | {student_params:,} | ~{student_params * 4 // 1000}KB |

圧縮率: {compression:,.0f}倍 小さい！
""")


# =============================================================================
# テスト
# =============================================================================

def test_gpt4all_distillation():
    """GPT4All蒸留テスト"""
    
    print("\n" + "=" * 70)
    print("🧪 GPT4All → 10進数ニューロン 蒸留テスト")
    print("=" * 70)
    
    # パイプライン作成
    pipeline = GPT4AllDistillationPipeline(
        student_hidden_size=32,
        student_layers=4
    )
    
    # 教師モデルロード（利用可能な場合）
    print("\n【教師モデル】")
    if GPT4ALL_AVAILABLE:
        # 軽量モデルを使用（初回はダウンロードに時間がかかる）
        # pipeline.load_teacher("orca-mini-3b-gguf2-q4_0.gguf")
        print("  GPT4All利用可能！")
        print("  ※今回はシミュレーションモードでテスト")
    else:
        print("  GPT4All未インストール - シミュレーションモード")
    
    # 学習データ収集
    print("\n【学習データ収集】")
    pipeline.collect_training_data()
    
    # 蒸留
    pipeline.distill(epochs=15)
    
    # 評価
    pipeline.evaluate()
    
    # サイズ比較
    pipeline.compare_sizes()
    
    print("\n" + "=" * 70)
    print("✅ テスト完了")
    print("=" * 70)
    
    return pipeline


def demo_with_gpt4all():
    """GPT4Allを実際に使うデモ"""
    
    if not GPT4ALL_AVAILABLE:
        print("GPT4Allがインストールされていません")
        return
    
    print("\n" + "=" * 70)
    print("🤖 GPT4All 実動デモ")
    print("=" * 70)
    
    # パイプライン
    pipeline = GPT4AllDistillationPipeline()
    
    # 教師をロード（モデルがダウンロードされる）
    print("\n【教師モデルロード】")
    print("  ※初回はモデルのダウンロードに数分かかります")
    
    if pipeline.load_teacher("orca-mini-3b-gguf2-q4_0.gguf"):
        # テストプロンプト
        prompts = ["Hello", "What is AI?", "1+1=?"]
        
        print("\n【GPT4All応答テスト】")
        for prompt in prompts:
            response = pipeline.teacher.generate(prompt, max_tokens=30)
            print(f"  Q: {prompt}")
            print(f"  A: {response[:50]}...")
            print()
        
        # 蒸留
        pipeline.collect_training_data(prompts)
        pipeline.distill(epochs=10)
        pipeline.evaluate()


if __name__ == "__main__":
    # シミュレーションテスト
    test_gpt4all_distillation()
    
    # GPT4Allを実際に使う場合はこちら
    # demo_with_gpt4all()
