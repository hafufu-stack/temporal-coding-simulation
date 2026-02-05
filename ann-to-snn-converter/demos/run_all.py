"""
全システム統合デモ
==================

すべての自律進化SNNを一気に動かす

Author: ろーる (cell_activation)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from applications.crypto_snn import EvolvingCryptoSNN
from applications.language_snn import EvolvingLanguageSNN
from applications.vision_snn import EvolvingVisionSNN
from applications.video_snn import EvolvingVideoSNN
from applications.research_snn import EvolvingResearchSNN


def run_all_demos():
    """全システムのデモを実行"""
    
    print("=" * 70)
    print("🚀 自律進化SNNフレームワーク - 全システムデモ")
    print("=" * 70)
    
    results = {}
    
    # 1. 暗号SNN
    print("\n" + "=" * 70)
    print("🔐 1. 暗号・圧縮SNN")
    print("=" * 70)
    
    crypto = EvolvingCryptoSNN(n_neurons=50)
    test_data = b"Hello, Autonomous SNN Framework!"
    encrypted = crypto.encrypt(test_data)
    decrypted = crypto.decrypt(encrypted)
    
    print(f"暗号化テスト: {test_data == decrypted}")
    
    for i in range(2):
        result = crypto.evolve_for_security()
        print(f"進化{i+1}: セキュリティ={result['security']:.2f}")
    
    results["crypto"] = {
        "security": result["security"],
        "evolution_drive": crypto.evolution.motivation.state.evolution_drive()
    }
    
    # 2. 言語SNN
    print("\n" + "=" * 70)
    print("📝 2. 言語モデルSNN")
    print("=" * 70)
    
    language = EvolvingLanguageSNN(n_neurons=80)
    texts = ["自律進化", "人工知能", "SNN"]
    language.auto_learn_cycle(texts)
    
    generated = language.generate("自律", length=10)
    print(f"生成: {generated}")
    
    results["language"] = {
        "vocab_size": language.vocab_size,
        "evolution_drive": language.evolution.motivation.state.evolution_drive()
    }
    
    # 3. 画像生成SNN
    print("\n" + "=" * 70)
    print("🎨 3. 画像生成SNN")
    print("=" * 70)
    
    vision = EvolvingVisionSNN(n_neurons=100, image_size=(16, 16))
    image = vision.generate()
    beauty = vision.evaluate_beauty(image)
    
    print(f"生成画像: {image.shape}, 美しさ: {beauty:.2f}")
    
    for i in range(2):
        result = vision.evolve_style()
        print(f"進化{i+1}: 美しさ={result['beauty']:.2f}")
    
    results["vision"] = {
        "beauty": result["beauty"],
        "evolution_drive": vision.evolution.motivation.state.evolution_drive()
    }
    
    # 4. 動画SNN
    print("\n" + "=" * 70)
    print("🎬 4. 動画処理SNN")
    print("=" * 70)
    
    video = EvolvingVideoSNN(n_neurons=80)
    
    import numpy as np
    frame1 = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)
    frame2 = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)
    interpolated = video.interpolate_frames(frame1, frame2, n_frames=3)
    
    print(f"補間: 2 → {len(interpolated)}フレーム")
    
    for i in range(2):
        result = video.evolve_quality()
        print(f"進化{i+1}: 品質={result['quality']:.2f}")
    
    results["video"] = {
        "quality": result["quality"],
        "evolution_drive": video.evolution.motivation.state.evolution_drive()
    }
    
    # 5. 研究SNN
    print("\n" + "=" * 70)
    print("🔬 5. 研究AI SNN")
    print("=" * 70)
    
    research = EvolvingResearchSNN(n_neurons=100)
    
    for i in range(2):
        result = research.research_cycle("SNN知性")
    
    theory = research.synthesize_theory()
    print(f"\n理論:\n{theory}")
    
    results["research"] = {
        "hypotheses": len(research.hypotheses),
        "discoveries": len(research.discoveries),
        "evolution_drive": research.evolution.motivation.state.evolution_drive()
    }
    
    # サマリー
    print("\n" + "=" * 70)
    print("📊 全システムサマリー")
    print("=" * 70)
    
    print("\n| システム | 主要指標 | 進化欲 |")
    print("|----------|----------|--------|")
    print(f"| 暗号SNN | セキュリティ={results['crypto']['security']:.2f} | {results['crypto']['evolution_drive']:.2f} |")
    print(f"| 言語SNN | 語彙={results['language']['vocab_size']} | {results['language']['evolution_drive']:.2f} |")
    print(f"| 画像SNN | 美しさ={results['vision']['beauty']:.2f} | {results['vision']['evolution_drive']:.2f} |")
    print(f"| 動画SNN | 品質={results['video']['quality']:.2f} | {results['video']['evolution_drive']:.2f} |")
    print(f"| 研究SNN | 発見={results['research']['discoveries']} | {results['research']['evolution_drive']:.2f} |")
    
    avg_drive = sum(r["evolution_drive"] for r in results.values()) / len(results)
    print(f"\n平均進化欲: {avg_drive:.2f}")
    
    if avg_drive > 0.5:
        print("→ 全システムが進化を求めている！")
    else:
        print("→ システムは現状に満足している")
    
    print("\n" + "=" * 70)
    print("✅ 全デモ完了！")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    run_all_demos()
