"""
GPT4All実モデル蒸留スクリプト
=============================

実際のGPT4Allモデルをダウンロードして蒸留

Author: ろーる (cell_activation)
Date: 2026-01-31
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt4all import GPT4All
from applications.llm_distillation import DecimalLLM, LLMDistiller


def main():
    print("\n" + "=" * 70)
    print("🤖 GPT4All 実モデル蒸留")
    print("=" * 70)
    
    # 軽量モデルを使用（約2GBダウンロード）
    model_name = "orca-mini-3b-gguf2-q4_0.gguf"
    
    print(f"\n【モデルダウンロード】")
    print(f"  モデル: {model_name}")
    print(f"  ※初回は2-3GBのダウンロードが発生します...")
    
    try:
        model = GPT4All(model_name)
        print("  ✅ モデルロード完了！")
    except Exception as e:
        print(f"  ❌ エラー: {e}")
        return
    
    # テストプロンプト
    print("\n【GPT4All応答テスト】")
    test_prompts = [
        "Hello, how are you?",
        "What is 2+2?",
        "What is artificial intelligence?",
    ]
    
    responses = []
    for prompt in test_prompts:
        print(f"\n  Q: {prompt}")
        response = model.generate(
            prompt,
            max_tokens=50,
            temp=0.7,
            top_p=0.9
        )
        print(f"  A: {response[:100]}...")
        responses.append((prompt, response))
    
    # 10進数ニューロンへ蒸留
    print("\n" + "=" * 50)
    print("🔬 10進数ニューロンへ蒸留")
    print("=" * 50)
    
    # 生徒モデル
    student = DecimalLLM(hidden_size=32, n_layers=4, context_length=64)
    distiller = LLMDistiller(student)
    
    # 学習データを追加
    print("\n【学習データ作成】")
    
    # GPT4Allから応答を収集
    training_prompts = [
        "Hello",
        "Hi there",
        "What is AI?",
        "How are you?",
        "Tell me a joke",
        "What is the capital of Japan?",
        "What is 1+1?",
        "What is programming?",
    ]
    
    for prompt in training_prompts:
        response = model.generate(prompt, max_tokens=30, temp=0.5)
        distiller.add_training_pair(prompt, response)
        print(f"  '{prompt}' → '{response[:40]}...'")
    
    # 蒸留
    print("\n【蒸留実行】")
    distiller.distill(epochs=20)
    
    # 評価
    print("\n【評価】")
    results = distiller.evaluate()
    print(f"  精度: {results['accuracy']:.2%}")
    
    # サイズ比較
    print("\n" + "=" * 50)
    print("📊 サイズ比較")
    print("=" * 50)
    
    student_stats = student.get_stats()
    print(f"""
| モデル | パラメータ | サイズ |
|--------|-----------|--------|
| GPT4All Orca 3B | ~3,000,000,000 | ~2GB |
| 10進数LLM | {student_stats['total_neurons']} | ~{student_stats['total_neurons'] * 4 // 1000}KB |

圧縮率: {3_000_000_000 // student_stats['total_neurons']:,}倍!
""")
    
    print("\n" + "=" * 70)
    print("✅ 蒸留完了！")
    print("=" * 70)
    
    return student


if __name__ == "__main__":
    main()
