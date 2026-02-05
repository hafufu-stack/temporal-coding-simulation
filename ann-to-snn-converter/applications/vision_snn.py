"""
自律進化 画像生成SNN (Evolving Vision SNN)
==========================================

スタイルを自動進化し、より美しい画像を追求する自律進化SNN-VAE

Author: ろーる (cell_activation)
"""

import numpy as np
from typing import Dict, List, Tuple
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.evolution_engine import EvolvingSNN


class EvolvingVisionSNN(EvolvingSNN):
    """
    自律進化する画像生成SNN
    
    自動で:
    - スタイルを進化
    - 美的品質を向上
    - 新しいパターンを探索
    """
    
    def __init__(self, n_neurons: int = 256, image_size: Tuple[int, int] = (32, 32)):
        super().__init__(n_neurons)
        
        self.image_size = image_size
        self.channels = 3  # RGB
        
        # スタイルパラメータ
        self.style = {
            "contrast": 0.5,
            "saturation": 0.5,
            "complexity": 0.5,
            "symmetry": 0.3
        }
        
        # 学習したパターン
        self.learned_patterns: List[np.ndarray] = []
        
        # スキル
        self.skills = {
            "beauty": 0.5,
            "novelty": 0.5,
            "coherence": 0.5
        }
    
    def generate(self, seed: np.ndarray = None, 
                 style_override: Dict = None) -> np.ndarray:
        """画像を生成"""
        h, w = self.image_size
        
        if seed is None:
            seed = np.random.randn(self.n_neurons) * 0.5
        
        # SNNで特徴を生成
        features = []
        state = seed.copy()
        
        for _ in range(h):
            state = self.step(state)
            features.append(state.copy())
        
        features = np.array(features)
        
        # 特徴を画像に変換
        image = np.zeros((h, w, 3))
        
        for c in range(3):
            channel = features[:, c * (w // 3):(c + 1) * (w // 3)]
            if channel.shape[1] < w:
                channel = np.pad(channel, ((0, 0), (0, w - channel.shape[1])))
            image[:, :, c] = channel[:, :w]
        
        # スタイルを適用
        style = style_override or self.style
        image = self._apply_style(image, style)
        
        # 0-255に正規化
        image = np.clip((image - image.min()) / (image.max() - image.min() + 0.001) * 255, 0, 255)
        
        return image.astype(np.uint8)
    
    def _apply_style(self, image: np.ndarray, style: Dict) -> np.ndarray:
        """スタイルを適用"""
        # コントラスト
        contrast = style.get("contrast", 0.5)
        mean = np.mean(image)
        image = (image - mean) * (0.5 + contrast) + mean
        
        # 彩度
        saturation = style.get("saturation", 0.5)
        gray = np.mean(image, axis=2, keepdims=True)
        image = gray + saturation * (image - gray)
        
        # 複雑さ（ノイズ追加）
        complexity = style.get("complexity", 0.5)
        image += np.random.randn(*image.shape) * complexity * 0.1
        
        # 対称性
        symmetry = style.get("symmetry", 0)
        if symmetry > 0.3:
            h, w, _ = image.shape
            left = image[:, :w//2, :]
            image[:, w//2:, :] = left[:, ::-1, :] * symmetry + image[:, w//2:, :] * (1 - symmetry)
        
        return image
    
    def evaluate_beauty(self, image: np.ndarray) -> float:
        """画像の美しさを評価"""
        # 対称性
        h, w, _ = image.shape
        left = image[:, :w//2, :]
        right = image[:, w//2:, :][:, ::-1, :]
        if left.shape == right.shape:
            symmetry = 1 - np.mean(np.abs(left - right)) / 255
        else:
            symmetry = 0.5
        
        # コントラスト（適度な範囲）
        std = np.std(image)
        contrast_score = 1 - abs(std - 50) / 100
        
        # 色の調和
        color_variance = np.var([np.mean(image[:,:,c]) for c in range(3)])
        harmony = 1 - min(1, color_variance / 1000)
        
        beauty = 0.4 * symmetry + 0.3 * contrast_score + 0.3 * harmony
        
        return beauty
    
    def evolve_style(self):
        """スタイルを進化させる"""
        # 現在のスタイルで画像を生成
        image = self.generate()
        beauty = self.evaluate_beauty(image)
        
        # 経験として記録
        self.experience(
            image.flatten()[:self.n_neurons].astype(float) / 255,
            skill="beauty",
            target=np.ones(self.n_neurons) * beauty
        )
        
        self.skills["beauty"] = beauty
        
        # 進化サイクル
        result = self.evolve(verbose=True)
        
        # 進化に応じてスタイルを調整
        if result.get("action") == "explore":
            # 新しいスタイルを探索
            for key in self.style:
                self.style[key] = np.clip(
                    self.style[key] + np.random.randn() * 0.1,
                    0, 1
                )
        
        return {"beauty": beauty, "style": self.style, "evolution": result}


def test_vision_snn():
    """テスト"""
    print("\n" + "=" * 70)
    print("🎨 自律進化 画像生成SNN テスト")
    print("=" * 70)
    
    snn = EvolvingVisionSNN(n_neurons=100, image_size=(16, 16))
    
    # 画像生成
    print("\n--- 画像生成 ---")
    image = snn.generate()
    print(f"  生成画像: {image.shape}")
    print(f"  美しさスコア: {snn.evaluate_beauty(image):.2f}")
    
    # スタイル進化
    print("\n--- スタイル進化 ---")
    for i in range(3):
        result = snn.evolve_style()
        print(f"サイクル{i+1}: 美しさ={result['beauty']:.2f}")
    
    snn.report()
    
    print("\n" + "=" * 70)
    print("✅ テスト完了")
    print("=" * 70)


if __name__ == "__main__":
    test_vision_snn()
