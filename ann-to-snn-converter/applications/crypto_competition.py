"""
圧縮・暗号合戦 (Compression/Encryption Competition)
===================================================

SNNが競い合って最高の圧縮・暗号化方法を発見！

ルール:
- ヒントなし！自分で発見する
- 元データを圧縮して復元、誰が一番正確か
- 暗号化して複合、誰が一番安全か
- 長時間サイクルで進化

Author: ろーる (cell_activation)
Date: 2026-01-31
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import random
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from applications.friendly_competition import CompetitiveSNNAgent, CompetitiveNetwork


# =============================================================================
# データ構造
# =============================================================================

@dataclass
class CompressionResult:
    """圧縮結果"""
    agent_id: str
    original_size: int
    compressed_size: int
    compression_ratio: float
    reconstruction_error: float
    success: bool


@dataclass
class EncryptionResult:
    """暗号化結果"""
    agent_id: str
    security_score: float  # 0-1
    decryption_success: bool
    entropy: float
    pattern_randomness: float


# =============================================================================
# 圧縮・暗号競争エージェント
# =============================================================================

class CryptoCompressorAgent(CompetitiveSNNAgent):
    """
    圧縮と暗号化を競争するエージェント
    
    ヒントなしで自分で方法を発見する！
    """
    
    def __init__(self, agent_id: str, n_neurons: int = 100, specialty: str = "general"):
        super().__init__(agent_id, n_neurons, specialty)
        
        # 圧縮パラメータ（自動で進化）
        self.compression_threshold = np.random.uniform(0.1, 0.9)
        self.compression_layers = np.random.randint(1, 5)
        self.sparsity_target = np.random.uniform(0.3, 0.8)
        
        # 暗号パラメータ（自動で進化）
        self.encryption_key_size = np.random.randint(16, 64)
        self.encryption_rounds = np.random.randint(1, 10)
        self.noise_level = np.random.uniform(0.01, 0.5)
        
        # 発見した方法
        self.discovered_methods: List[str] = []
        
        # 統計
        self.best_compression_ratio = 0.0
        self.best_security_score = 0.0
        self.compression_history: List[float] = []
        self.security_history: List[float] = []
    
    def compress(self, data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """データを圧縮"""
        original_size = data.size
        
        # 1. SNNを通してパターン抽出
        pattern = self.step(data.flatten()[:self.n_neurons])
        
        # 2. スパース化（閾値以下をゼロに）
        sparse = pattern.copy()
        sparse[np.abs(sparse) < self.compression_threshold] = 0
        
        # 3. 重要な値のみ保持
        important_indices = np.where(np.abs(sparse) > 0)[0]
        important_values = sparse[important_indices]
        
        # 圧縮データ
        compressed = {
            "indices": important_indices,
            "values": important_values,
            "shape": data.shape,
            "layers": self.compression_layers
        }
        
        compressed_size = len(important_indices) + len(important_values)
        
        return sparse, {
            "original_size": original_size,
            "compressed_size": compressed_size,
            "ratio": 1 - compressed_size / max(1, original_size)
        }
    
    def decompress(self, compressed: np.ndarray, original: np.ndarray) -> np.ndarray:
        """データを復元"""
        # 逆変換を試みる
        reconstructed = np.zeros(self.n_neurons)
        
        # SNNの逆向き処理（近似）
        for _ in range(self.compression_layers):
            reconstructed = self.step(compressed)
        
        # 元のサイズに合わせる
        if len(reconstructed) < original.size:
            reconstructed = np.pad(reconstructed, (0, original.size - len(reconstructed)))
        else:
            reconstructed = reconstructed[:original.size]
        
        return reconstructed.reshape(original.shape)
    
    def encrypt(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """データを暗号化"""
        # 1. SNNベースの変換
        transformed = self.step(data.flatten()[:self.n_neurons])
        
        # 鍵を生成（変換後のサイズに合わせる）
        key = np.random.randn(len(transformed))
        
        # 2. ノイズを追加
        noisy = transformed + np.random.randn(len(transformed)) * self.noise_level
        
        # 3. 複数ラウンドの変換
        encrypted = noisy.copy()
        for _ in range(self.encryption_rounds):
            encrypted = np.tanh(encrypted + key)
        
        return encrypted, key
    
    def decrypt(self, encrypted: np.ndarray, key: np.ndarray) -> np.ndarray:
        """データを復号"""
        decrypted = encrypted.copy()
        
        # 逆変換
        for _ in range(self.encryption_rounds):
            decrypted = np.arctanh(np.clip(decrypted, -0.999, 0.999)) - key[:len(decrypted)]
        
        # SNNで再構成
        reconstructed = self.step(decrypted)
        
        return reconstructed
    
    def compete_compression(self, data: np.ndarray) -> CompressionResult:
        """圧縮競争に参加"""
        # 圧縮
        compressed, info = self.compress(data)
        
        # 復元
        reconstructed = self.decompress(compressed, data)
        
        # 復元誤差
        flat_data = data.flatten()[:len(reconstructed.flatten())]
        flat_recon = reconstructed.flatten()[:len(flat_data)]
        
        if len(flat_data) > 0 and len(flat_recon) > 0:
            error = np.mean((flat_data - flat_recon) ** 2)
        else:
            error = 1.0
        
        ratio = info["ratio"]
        success = error < 0.5 and ratio > 0.1
        
        # 記録
        self.compression_history.append(ratio)
        if ratio > self.best_compression_ratio and success:
            self.best_compression_ratio = ratio
            self.discovered_methods.append(f"圧縮率{ratio:.2f}達成")
        
        return CompressionResult(
            agent_id=self.agent_id,
            original_size=info["original_size"],
            compressed_size=info["compressed_size"],
            compression_ratio=ratio,
            reconstruction_error=error,
            success=success
        )
    
    def compete_encryption(self, data: np.ndarray) -> EncryptionResult:
        """暗号化競争に参加"""
        # 暗号化
        encrypted, key = self.encrypt(data)
        
        # 復号
        decrypted = self.decrypt(encrypted, key)
        
        # 復号成功？
        flat_data = data.flatten()[:len(decrypted)]
        if len(flat_data) > 0:
            similarity = np.corrcoef(flat_data, decrypted[:len(flat_data)])[0, 1]
            if np.isnan(similarity):
                similarity = 0
            decryption_success = similarity > 0.3
        else:
            decryption_success = False
            similarity = 0
        
        # セキュリティスコア
        entropy = self._compute_entropy(encrypted)
        randomness = self._compute_randomness(encrypted)
        security = 0.5 * entropy + 0.5 * randomness
        
        # 記録
        self.security_history.append(security)
        if security > self.best_security_score and decryption_success:
            self.best_security_score = security
            self.discovered_methods.append(f"セキュリティ{security:.2f}達成")
        
        return EncryptionResult(
            agent_id=self.agent_id,
            security_score=security,
            decryption_success=decryption_success,
            entropy=entropy,
            pattern_randomness=randomness
        )
    
    def _compute_entropy(self, data: np.ndarray) -> float:
        """エントロピーを計算"""
        # ビンに分割
        hist, _ = np.histogram(data, bins=20, density=True)
        hist = hist[hist > 0]
        if len(hist) == 0:
            return 0
        return -np.sum(hist * np.log2(hist + 1e-10)) / np.log2(20)
    
    def _compute_randomness(self, data: np.ndarray) -> float:
        """ランダム性を計算"""
        if len(data) < 2:
            return 0
        # 自己相関が低いほどランダム
        autocorr = np.correlate(data, data, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        if len(autocorr) > 1:
            return 1 - np.abs(autocorr[1] / (autocorr[0] + 1e-10))
        return 0.5
    
    def evolve_parameters(self, success: bool, score: float):
        """パラメータを進化"""
        if success and score > 0.5:
            # 成功した方向に進化
            self.compression_threshold *= np.random.uniform(0.9, 1.1)
            self.noise_level *= np.random.uniform(0.9, 1.1)
            self.encryption_rounds = max(1, self.encryption_rounds + np.random.randint(-1, 2))
        else:
            # 探索
            self.compression_threshold = np.random.uniform(0.1, 0.9)
            self.noise_level = np.random.uniform(0.01, 0.5)
            self.encryption_rounds = np.random.randint(1, 10)
        
        # クリップ
        self.compression_threshold = np.clip(self.compression_threshold, 0.1, 0.9)
        self.noise_level = np.clip(self.noise_level, 0.01, 0.5)
    
    def learn_from_winner(self, winner: 'CryptoCompressorAgent', aspect: str = "both"):
        """勝者から学ぶ"""
        blend = 0.2
        
        if aspect in ["compression", "both"]:
            self.compression_threshold = (
                (1 - blend) * self.compression_threshold + 
                blend * winner.compression_threshold
            )
            self.compression_layers = winner.compression_layers
        
        if aspect in ["encryption", "both"]:
            self.noise_level = (
                (1 - blend) * self.noise_level + 
                blend * winner.noise_level
            )
            self.encryption_rounds = winner.encryption_rounds


# =============================================================================
# 圧縮・暗号大会
# =============================================================================

class CryptoCompressionCompetition:
    """
    圧縮・暗号合戦
    """
    
    def __init__(self):
        self.agents: Dict[str, CryptoCompressorAgent] = {}
        self.compression_leaderboard: Dict[str, float] = {}
        self.encryption_leaderboard: Dict[str, float] = {}
        self.round_count = 0
        
        # 進化履歴
        self.evolution_history: List[Dict] = []
    
    def add_agent(self, agent_id: str, specialty: str = "general"):
        """エージェントを追加"""
        agent = CryptoCompressorAgent(agent_id, n_neurons=100, specialty=specialty)
        self.agents[agent_id] = agent
        self.compression_leaderboard[agent_id] = 0
        self.encryption_leaderboard[agent_id] = 0
        print(f"  🤖 {agent_id} ({specialty}) が参戦")
        return agent
    
    def generate_challenge_data(self, difficulty: float = 0.5) -> np.ndarray:
        """競争用データを生成"""
        size = int(50 + 50 * difficulty)
        
        # 様々なパターン
        patterns = [
            np.random.randn(size),  # ランダム
            np.sin(np.linspace(0, 10, size)),  # 周期的
            np.cumsum(np.random.randn(size)),  # ランダムウォーク
            np.eye(int(np.sqrt(size)) + 1).flatten()[:size],  # 構造的
        ]
        
        return random.choice(patterns)
    
    def run_compression_round(self, data: np.ndarray) -> Dict[str, CompressionResult]:
        """圧縮ラウンド"""
        results = {}
        
        for agent_id, agent in self.agents.items():
            result = agent.compete_compression(data)
            results[agent_id] = result
            
            if result.success:
                self.compression_leaderboard[agent_id] += result.compression_ratio
        
        return results
    
    def run_encryption_round(self, data: np.ndarray) -> Dict[str, EncryptionResult]:
        """暗号化ラウンド"""
        results = {}
        
        for agent_id, agent in self.agents.items():
            result = agent.compete_encryption(data)
            results[agent_id] = result
            
            if result.decryption_success:
                self.encryption_leaderboard[agent_id] += result.security_score
        
        return results
    
    def run_round(self, verbose: bool = True):
        """1ラウンド実行"""
        self.round_count += 1
        difficulty = min(1.0, 0.3 + 0.01 * self.round_count)
        
        data = self.generate_challenge_data(difficulty)
        
        # 圧縮競争
        comp_results = self.run_compression_round(data)
        
        # 暗号競争
        enc_results = self.run_encryption_round(data)
        
        # 勝者から学ぶ
        if comp_results:
            comp_winner_id = max(comp_results, key=lambda x: comp_results[x].compression_ratio if comp_results[x].success else 0)
            comp_winner = self.agents[comp_winner_id]
            
            enc_winner_id = max(enc_results, key=lambda x: enc_results[x].security_score if enc_results[x].decryption_success else 0)
            enc_winner = self.agents[enc_winner_id]
            
            for agent in self.agents.values():
                if agent.agent_id != comp_winner_id:
                    agent.learn_from_winner(comp_winner, "compression")
                if agent.agent_id != enc_winner_id:
                    agent.learn_from_winner(enc_winner, "encryption")
                
                # パラメータ進化
                avg_score = (comp_results[agent.agent_id].compression_ratio + 
                            enc_results[agent.agent_id].security_score) / 2
                agent.evolve_parameters(
                    comp_results[agent.agent_id].success or enc_results[agent.agent_id].decryption_success,
                    avg_score
                )
        
        if verbose and self.round_count % 10 == 0:
            print(f"\n--- ラウンド {self.round_count} ---")
            
            # 圧縮トップ
            best_comp = max(comp_results.values(), key=lambda x: x.compression_ratio if x.success else 0)
            print(f"  📦 圧縮: {best_comp.agent_id} (圧縮率={best_comp.compression_ratio:.2f})")
            
            # 暗号トップ
            best_enc = max(enc_results.values(), key=lambda x: x.security_score if x.decryption_success else 0)
            print(f"  🔐 暗号: {best_enc.agent_id} (セキュリティ={best_enc.security_score:.2f})")
        
        # 履歴に記録
        self.evolution_history.append({
            "round": self.round_count,
            "best_compression": max(r.compression_ratio for r in comp_results.values()),
            "best_security": max(r.security_score for r in enc_results.values())
        })
    
    def run_competition(self, rounds: int = 100, verbose: bool = True):
        """大会を実行"""
        print("\n" + "=" * 70)
        print("🏆 圧縮・暗号合戦 開始！")
        print("=" * 70)
        print(f"参加者: {', '.join(self.agents.keys())}")
        print(f"ラウンド数: {rounds}")
        print("ヒント: なし！自分で発見せよ！")
        
        for _ in range(rounds):
            self.run_round(verbose)
        
        self.show_final_results()
    
    def show_final_results(self):
        """最終結果を表示"""
        print("\n" + "=" * 70)
        print("📊 最終結果")
        print("=" * 70)
        
        # 圧縮ランキング
        print("\n【圧縮ランキング】")
        comp_ranking = sorted(self.compression_leaderboard.items(), key=lambda x: x[1], reverse=True)
        medals = ["🥇", "🥈", "🥉", "4️⃣"]
        for i, (agent_id, score) in enumerate(comp_ranking):
            medal = medals[i] if i < len(medals) else f"{i+1}."
            agent = self.agents[agent_id]
            print(f"  {medal} {agent_id}: 累積圧縮率={score:.2f}")
            print(f"      最高記録: {agent.best_compression_ratio:.2f}")
            print(f"      閾値: {agent.compression_threshold:.2f}, レイヤー: {agent.compression_layers}")
        
        # 暗号ランキング
        print("\n【暗号ランキング】")
        enc_ranking = sorted(self.encryption_leaderboard.items(), key=lambda x: x[1], reverse=True)
        for i, (agent_id, score) in enumerate(enc_ranking):
            medal = medals[i] if i < len(medals) else f"{i+1}."
            agent = self.agents[agent_id]
            print(f"  {medal} {agent_id}: 累積セキュリティ={score:.2f}")
            print(f"      最高記録: {agent.best_security_score:.2f}")
            print(f"      ノイズ: {agent.noise_level:.2f}, ラウンド: {agent.encryption_rounds}")
        
        # 発見した方法
        print("\n【発見した方法】")
        for agent_id, agent in self.agents.items():
            if agent.discovered_methods:
                print(f"  {agent_id}:")
                for method in agent.discovered_methods[-3:]:
                    print(f"    • {method}")
        
        # 進化曲線
        if self.evolution_history:
            print("\n【進化の軌跡】")
            checkpoints = [0, len(self.evolution_history)//4, len(self.evolution_history)//2, 
                          3*len(self.evolution_history)//4, len(self.evolution_history)-1]
            for i in checkpoints:
                if i < len(self.evolution_history):
                    h = self.evolution_history[i]
                    print(f"  ラウンド{h['round']:3d}: 圧縮={h['best_compression']:.2f}, セキュリティ={h['best_security']:.2f}")


# =============================================================================
# テスト
# =============================================================================

def test_crypto_compression_competition(rounds: int = 100):
    """圧縮・暗号合戦テスト"""
    
    print("\n" + "=" * 70)
    print("🧪 圧縮・暗号合戦 テスト")
    print("=" * 70)
    
    # 大会作成
    competition = CryptoCompressionCompetition()
    
    # エージェント追加
    competition.add_agent("CompressionMaster", specialty="圧縮")
    competition.add_agent("CryptoKing", specialty="暗号")
    competition.add_agent("AllRounder", specialty="汎用")
    competition.add_agent("Explorer", specialty="探索")
    
    # 大会実行
    competition.run_competition(rounds=rounds, verbose=True)
    
    print("\n" + "=" * 70)
    print("✅ テスト完了")
    print("=" * 70)
    
    return competition


if __name__ == "__main__":
    # 100ラウンドで実行
    test_crypto_compression_competition(rounds=100)
