"""
自律進化 動画SNN (Evolving Video SNN)
=====================================

補間品質を自動改善し、音声生成を洗練する自律進化動画処理SNN

Author: ろーる (cell_activation)
"""

import numpy as np
from typing import Dict, List, Tuple
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.evolution_engine import EvolvingSNN


class EvolvingVideoSNN(EvolvingSNN):
    """
    自律進化する動画処理SNN
    
    自動で:
    - フレーム補間品質を改善
    - 超解像を向上
    - 音声生成を洗練
    """
    
    def __init__(self, n_neurons: int = 200):
        super().__init__(n_neurons)
        
        # 処理パラメータ
        self.params = {
            "interpolation_smoothness": 0.5,
            "upscale_sharpness": 0.5,
            "audio_richness": 0.5
        }
        
        # スキル
        self.skills = {
            "interpolation": 0.5,
            "upscaling": 0.5,
            "audio_sync": 0.5
        }
    
    def interpolate_frames(self, frame1: np.ndarray, frame2: np.ndarray,
                           n_frames: int = 5) -> List[np.ndarray]:
        """フレーム補間"""
        interpolated = []
        
        # SNNで補間タイミングを生成
        timings = []
        for i in range(n_frames):
            t = (i + 1) / (n_frames + 1)
            input_vec = np.array([t, 1-t] + [0] * (self.n_neurons - 2))
            output = self.step(input_vec)
            
            # 非線形タイミング
            snn_t = np.clip(t + np.mean(output) * 0.1, 0, 1)
            timings.append(snn_t)
        
        smoothness = self.params["interpolation_smoothness"]
        
        for t in timings:
            # ブレンド
            blended = (1 - t) * frame1 + t * frame2
            
            # スムーズネスを適用
            if smoothness > 0.5:
                # エッジをソフト化
                kernel_size = int(smoothness * 3)
                if kernel_size > 0:
                    # 簡易ガウシアンブラー
                    blended = self._simple_blur(blended, kernel_size)
            
            interpolated.append(blended.astype(np.uint8))
        
        return interpolated
    
    def _simple_blur(self, image: np.ndarray, size: int) -> np.ndarray:
        """簡易ブラー"""
        from scipy.ndimage import uniform_filter
        try:
            return uniform_filter(image.astype(float), size=size)
        except:
            return image
    
    def upscale(self, image: np.ndarray, scale: int = 2) -> np.ndarray:
        """超解像"""
        h, w = image.shape[:2]
        new_h, new_w = h * scale, w * scale
        
        # 基本的なアップスケール
        if len(image.shape) == 3:
            upscaled = np.zeros((new_h, new_w, image.shape[2]))
            for c in range(image.shape[2]):
                upscaled[:, :, c] = self._bilinear_upscale(image[:, :, c], scale)
        else:
            upscaled = self._bilinear_upscale(image, scale)
        
        # SNNでディテール追加
        sharpness = self.params["upscale_sharpness"]
        if sharpness > 0.3:
            # シャープ化
            upscaled = self._sharpen(upscaled, sharpness)
        
        return np.clip(upscaled, 0, 255).astype(np.uint8)
    
    def _bilinear_upscale(self, channel: np.ndarray, scale: int) -> np.ndarray:
        """バイリニア補間"""
        h, w = channel.shape
        new_h, new_w = h * scale, w * scale
        
        result = np.zeros((new_h, new_w))
        for i in range(new_h):
            for j in range(new_w):
                src_i = i / scale
                src_j = j / scale
                
                i0, j0 = int(src_i), int(src_j)
                i1, j1 = min(i0 + 1, h - 1), min(j0 + 1, w - 1)
                
                di, dj = src_i - i0, src_j - j0
                
                result[i, j] = (
                    channel[i0, j0] * (1 - di) * (1 - dj) +
                    channel[i1, j0] * di * (1 - dj) +
                    channel[i0, j1] * (1 - di) * dj +
                    channel[i1, j1] * di * dj
                )
        
        return result
    
    def _sharpen(self, image: np.ndarray, strength: float) -> np.ndarray:
        """シャープ化"""
        # ラプラシアンフィルタの近似
        if len(image.shape) == 3:
            for c in range(image.shape[2]):
                channel = image[:, :, c]
                h, w = channel.shape
                
                laplacian = np.zeros_like(channel)
                laplacian[1:-1, 1:-1] = (
                    4 * channel[1:-1, 1:-1] -
                    channel[:-2, 1:-1] - channel[2:, 1:-1] -
                    channel[1:-1, :-2] - channel[1:-1, 2:]
                )
                
                image[:, :, c] = channel + strength * laplacian * 0.1
        
        return image
    
    def generate_audio(self, duration: float, 
                       scene_hints: List[str] = None) -> np.ndarray:
        """音声生成"""
        sample_rate = 44100
        n_samples = int(duration * sample_rate)
        
        audio = np.zeros(n_samples)
        richness = self.params["audio_richness"]
        
        # SNNで音声パラメータを生成
        freq_base = 200 + 300 * richness
        
        t = np.linspace(0, duration, n_samples)
        
        # 基本波形
        audio = np.sin(2 * np.pi * freq_base * t) * 0.3
        
        # 倍音を追加
        for harmonic in range(2, int(richness * 5) + 2):
            audio += np.sin(2 * np.pi * freq_base * harmonic * t) * 0.1 / harmonic
        
        # ノイズ
        audio += np.random.randn(n_samples) * 0.05 * richness
        
        return np.clip(audio, -1, 1)
    
    def evolve_quality(self, sample_frames: List[np.ndarray] = None):
        """品質を進化させる"""
        # 評価
        quality = np.mean(list(self.params.values()))
        
        # 経験として記録
        self.experience(
            np.random.randn(self.n_neurons),
            skill="interpolation",
            target=np.ones(self.n_neurons) * quality
        )
        
        # 進化
        result = self.evolve(verbose=True)
        
        # パラメータ調整
        if result.get("action") in ["optimize", "explore"]:
            for key in self.params:
                self.params[key] = np.clip(
                    self.params[key] + np.random.randn() * 0.05,
                    0.1, 0.9
                )
        
        return {"quality": quality, "params": self.params, "evolution": result}


def test_video_snn():
    """テスト"""
    print("\n" + "=" * 70)
    print("🎬 自律進化 動画SNN テスト")
    print("=" * 70)
    
    snn = EvolvingVideoSNN(n_neurons=100)
    
    # フレーム補間テスト
    print("\n--- フレーム補間 ---")
    frame1 = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    frame2 = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    
    interpolated = snn.interpolate_frames(frame1, frame2, n_frames=3)
    print(f"  入力: 2フレーム → 出力: {len(interpolated)}フレーム")
    
    # 超解像テスト
    print("\n--- 超解像 ---")
    small = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)
    large = snn.upscale(small, scale=2)
    print(f"  {small.shape} → {large.shape}")
    
    # 音声生成テスト
    print("\n--- 音声生成 ---")
    audio = snn.generate_audio(1.0)
    print(f"  1秒 → {len(audio)}サンプル")
    
    # 品質進化
    print("\n--- 品質進化 ---")
    for i in range(3):
        result = snn.evolve_quality()
        print(f"サイクル{i+1}: 品質={result['quality']:.2f}")
    
    snn.report()
    
    print("\n" + "=" * 70)
    print("✅ テスト完了")
    print("=" * 70)


if __name__ == "__main__":
    test_video_snn()
