"""
ELYZA日本語モデル蒸留スクリプト
================================

ELYZA-japanese-Llama-2-7bを使用して
10進数ニューロンへ日本語能力を蒸留

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
    print("🇯🇵 ELYZA日本語モデル蒸留")
    print("=" * 70)
    
    # ELYZA日本語モデル（約4GBダウンロード）
    # GPT4Allで利用可能なELYZAモデル
    model_name = "elyza-japanese-llama-2-7b-fast-instruct.Q4_K_M.gguf"
    
    print(f"\n【モデルダウンロード】")
    print(f"  モデル: {model_name}")
    print(f"  サイズ: 約4GB")
    print(f"  ※初回ダウンロードに数分かかります...")
    
    try:
        # GPT4Allでモデルをロード
        # allow_download=Trueで自動ダウンロード
        model = GPT4All(model_name, allow_download=True)
        print("  ✅ モデルロード完了！")
    except Exception as e:
        print(f"  モデルが見つかりません。代替モデルを試します...")
        # 代替: 軽量な日本語対応モデル
        try:
            model = GPT4All("mistral-7b-instruct-v0.1.Q4_0.gguf")
            print("  ✅ Mistral（多言語対応）をロード")
        except Exception as e2:
            print(f"  ❌ エラー: {e2}")
            print("  既存のOrcaモデルを使用します")
            model = GPT4All("orca-mini-3b-gguf2-q4_0.gguf")
    
    # 日本語テスト
    print("\n【日本語応答テスト】")
    japanese_prompts = [
        "こんにちは、元気ですか？",
        "日本の首都はどこですか？",
        "プログラミングとは何ですか？",
        "1+1は何ですか？",
        "AIについて簡単に説明してください。",
    ]
    
    responses = []
    for prompt in japanese_prompts:
        print(f"\n  Q: {prompt}")
        response = model.generate(
            prompt,
            max_tokens=100,
            temp=0.7,
            top_p=0.9
        )
        print(f"  A: {response[:150]}...")
        responses.append((prompt, response))
    
    # 10進数ニューロンへ蒸留
    print("\n" + "=" * 50)
    print("🔬 10進数ニューロンへ日本語蒸留")
    print("=" * 50)
    
    # 生徒モデル
    student = DecimalLLM(hidden_size=64, n_layers=6, context_length=128)
    distiller = LLMDistiller(student)
    
    # 日本語学習データを追加
    print("\n【日本語学習データ作成】")
    
    japanese_training = [
        # 挨拶
        "こんにちは",
        "おはようございます", 
        "こんばんは",
        "さようなら",
        "ありがとうございます",
        "すみません",
        "お願いします",
        "わかりました",
        # 質問
        "今何時ですか？",
        "お名前は何ですか？",
        "どこから来ましたか？",
        "天気はどうですか？",
        # 数学
        "1足す1は？",
        "2かける3は？",
        "10割る2は？",
    ]
    
    for prompt in japanese_training:
        response = model.generate(prompt, max_tokens=50, temp=0.5)
        distiller.add_training_pair(prompt, response)
        print(f"  '{prompt}' → '{response[:35]}...'")
    
    # 蒸留
    print("\n【蒸留実行】")
    distiller.distill(epochs=30)
    
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
| ELYZA 7B | ~7,000,000,000 | ~4GB |
| 10進数LLM | {student_stats['total_neurons']} | ~{student_stats['total_neurons'] * 4 // 1000}KB |

圧縮率: {7_000_000_000 // max(1, student_stats['total_neurons']):,}倍!
""")
    
    # 日本語生成テスト
    print("\n【生成テスト】")
    test_prompts = ["こんにちは", "ありがとう", "天気"]
    for prompt in test_prompts:
        student.clear_context()
        output = student.generate(prompt, max_length=15)
        print(f"  '{prompt}' → '{output}'")
    
    print("\n" + "=" * 70)
    print("✅ 日本語蒸留完了！")
    print("=" * 70)
    
    return student


if __name__ == "__main__":
    main()
