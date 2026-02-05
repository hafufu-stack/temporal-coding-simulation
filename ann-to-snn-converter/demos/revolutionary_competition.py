"""
革命的発見を求める圧縮・暗号合戦
================================

新しい欲:
- 現状に満足しない
- 革命的技術を見つけたい

1000サイクル並列処理で実行！

Author: ろーる (cell_activation)
Date: 2026-01-31
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# 革命的発見を求めるエージェント
# =============================================================================

class RevolutionaryAgent:
    """
    革命的発見を求めるエージェント
    
    新しい欲:
    - 現状に満足しない (dissatisfaction)
    - 革命的技術を見つけたい (revolution_desire)
    """
    
    def __init__(self, agent_id: str, n_neurons: int = 100):
        self.agent_id = agent_id
        self.n_neurons = n_neurons
        
        # SNN重み
        self.W = np.random.randn(n_neurons, n_neurons) * 0.1
        self.state = np.zeros(n_neurons)
        self.threshold = 0.5
        
        # 【新しい欲！】
        self.dissatisfaction = 0.8       # 現状に満足しない (高いほど不満)
        self.revolution_desire = 0.9     # 革命的技術を見つけたい
        self.exploration_courage = 0.7   # 未知への勇気
        
        # 圧縮パラメータ
        self.compression_threshold = np.random.uniform(0.1, 0.9)
        self.compression_method = "sparse"  # sparse, temporal, hybrid
        
        # 暗号パラメータ
        self.noise_level = np.random.uniform(0.01, 0.5)
        self.encryption_rounds = np.random.randint(1, 10)
        self.key_complexity = np.random.uniform(0.1, 1.0)
        
        # 発見履歴
        self.best_compression = 0.0
        self.best_security = 0.0
        self.revolutionary_discoveries: List[str] = []
        self.failed_experiments: List[str] = []
        
        # 統計
        self.total_experiments = 0
        self.successful_experiments = 0
    
    def step(self, x: np.ndarray) -> np.ndarray:
        """SNNステップ"""
        x = x[:self.n_neurons] if len(x) > self.n_neurons else np.pad(x, (0, self.n_neurons - len(x)))
        self.state = 0.9 * self.state + 0.1 * (self.W @ self.state + x)
        spikes = (self.state > self.threshold).astype(float)
        self.state = self.state * (1 - spikes)
        return self.state
    
    def compress(self, data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """圧縮"""
        original_size = data.size
        
        # SNNで変換
        pattern = self.step(data.flatten()[:self.n_neurons])
        
        # 方法に応じた圧縮
        if self.compression_method == "sparse":
            compressed = pattern.copy()
            compressed[np.abs(compressed) < self.compression_threshold] = 0
        elif self.compression_method == "temporal":
            # 時間的パターンを利用
            compressed = np.diff(pattern, prepend=0)
            compressed[np.abs(compressed) < self.compression_threshold] = 0
        else:  # hybrid
            sparse = pattern.copy()
            sparse[np.abs(sparse) < self.compression_threshold] = 0
            temporal = np.diff(pattern, prepend=0)
            compressed = 0.5 * sparse + 0.5 * temporal
        
        non_zero = np.count_nonzero(compressed)
        ratio = 1 - non_zero / max(1, len(compressed))
        
        return compressed, {"ratio": ratio, "original_size": original_size}
    
    def decompress(self, compressed: np.ndarray, original: np.ndarray) -> np.ndarray:
        """復元"""
        reconstructed = np.zeros(self.n_neurons)
        for _ in range(3):
            reconstructed = self.step(compressed)
        
        if len(reconstructed) < original.size:
            reconstructed = np.pad(reconstructed, (0, original.size - len(reconstructed)))
        return reconstructed[:original.size]
    
    def encrypt(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """暗号化"""
        transformed = self.step(data.flatten()[:self.n_neurons])
        key = np.random.randn(len(transformed)) * self.key_complexity
        
        noisy = transformed + np.random.randn(len(transformed)) * self.noise_level
        encrypted = noisy.copy()
        
        for _ in range(self.encryption_rounds):
            encrypted = np.tanh(encrypted + key)
        
        return encrypted, key
    
    def decrypt(self, encrypted: np.ndarray, key: np.ndarray) -> np.ndarray:
        """復号"""
        decrypted = encrypted.copy()
        for _ in range(self.encryption_rounds):
            decrypted = np.arctanh(np.clip(decrypted, -0.999, 0.999)) - key
        return self.step(decrypted)
    
    def compute_security(self, encrypted: np.ndarray) -> float:
        """セキュリティスコア"""
        # エントロピー
        hist, _ = np.histogram(encrypted, bins=20, density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist + 1e-10)) / np.log2(20) if len(hist) > 0 else 0
        
        # ランダム性
        if len(encrypted) > 1:
            autocorr = np.correlate(encrypted, encrypted, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            randomness = 1 - np.abs(autocorr[1] / (autocorr[0] + 1e-10)) if len(autocorr) > 1 else 0.5
        else:
            randomness = 0.5
        
        return 0.5 * entropy + 0.5 * randomness
    
    def try_revolutionary_experiment(self):
        """革命的実験を試みる"""
        self.total_experiments += 1
        
        # 革命欲が高いほど大胆な実験
        if self.revolution_desire > 0.7:
            # 大胆な変更
            experiments = [
                ("threshold_extreme", lambda: setattr(self, 'compression_threshold', np.random.uniform(0.01, 0.99))),
                ("method_change", lambda: setattr(self, 'compression_method', random.choice(["sparse", "temporal", "hybrid"]))),
                ("noise_extreme", lambda: setattr(self, 'noise_level', np.random.uniform(0.001, 0.9))),
                ("rounds_extreme", lambda: setattr(self, 'encryption_rounds', np.random.randint(1, 20))),
                ("key_extreme", lambda: setattr(self, 'key_complexity', np.random.uniform(0.01, 2.0))),
                ("weight_mutation", self._mutate_weights),
            ]
        else:
            # 保守的な変更
            experiments = [
                ("threshold_adjust", lambda: setattr(self, 'compression_threshold', 
                    np.clip(self.compression_threshold + np.random.randn() * 0.1, 0.1, 0.9))),
                ("noise_adjust", lambda: setattr(self, 'noise_level',
                    np.clip(self.noise_level + np.random.randn() * 0.05, 0.01, 0.5))),
            ]
        
        # ランダムに実験を選択
        name, experiment = random.choice(experiments)
        experiment()
        
        return name
    
    def _mutate_weights(self):
        """重みを突然変異"""
        mutation = np.random.randn(*self.W.shape) * 0.1 * self.exploration_courage
        self.W += mutation
    
    def update_desires(self, compression_improved: bool, security_improved: bool):
        """欲を更新"""
        if compression_improved or security_improved:
            # 成功したら少し満足するが...
            self.dissatisfaction *= 0.95
            self.revolution_desire *= 0.98
            self.successful_experiments += 1
            
            # でも「もっと良くなるはず！」
            if self.dissatisfaction < 0.3:
                self.dissatisfaction = 0.5  # 完全に満足しない
        else:
            # 失敗したら革命欲が上がる
            self.dissatisfaction = min(1.0, self.dissatisfaction + 0.05)
            self.revolution_desire = min(1.0, self.revolution_desire + 0.03)
    
    def learn_from_other(self, other: 'RevolutionaryAgent'):
        """他から学ぶ"""
        blend = 0.1
        self.compression_threshold = (1-blend) * self.compression_threshold + blend * other.compression_threshold
        self.noise_level = (1-blend) * self.noise_level + blend * other.noise_level
        self.compression_method = other.compression_method  # 良い方法を真似る


# =============================================================================
# 並列処理競争
# =============================================================================

class RevolutionaryCompetition:
    """並列処理対応の競争"""
    
    def __init__(self, n_workers: int = 4):
        self.agents: Dict[str, RevolutionaryAgent] = {}
        self.n_workers = n_workers
        self.round_count = 0
        
        # 結果
        self.compression_scores: Dict[str, float] = {}
        self.security_scores: Dict[str, float] = {}
        self.history: List[Dict] = []
    
    def add_agent(self, agent_id: str):
        """エージェント追加"""
        agent = RevolutionaryAgent(agent_id)
        self.agents[agent_id] = agent
        self.compression_scores[agent_id] = 0
        self.security_scores[agent_id] = 0
        return agent
    
    def generate_data(self, difficulty: float = 0.5) -> np.ndarray:
        """データ生成"""
        size = int(50 + 50 * difficulty)
        patterns = [
            np.random.randn(size),
            np.sin(np.linspace(0, 10, size)),
            np.cumsum(np.random.randn(size)),
        ]
        return random.choice(patterns)
    
    def run_agent_round(self, agent: RevolutionaryAgent, data: np.ndarray) -> Dict:
        """1エージェントのラウンド（並列実行用）"""
        # 革命的実験を試みる
        experiment = agent.try_revolutionary_experiment()
        
        # 圧縮テスト
        compressed, info = agent.compress(data)
        reconstructed = agent.decompress(compressed, data)
        
        flat_data = data.flatten()[:len(reconstructed)]
        if len(flat_data) > 0:
            error = np.mean((flat_data - reconstructed[:len(flat_data)]) ** 2)
        else:
            error = 1.0
        
        compression_ratio = info["ratio"]
        compression_success = error < 0.5 and compression_ratio > 0.1
        
        # 暗号テスト
        encrypted, key = agent.encrypt(data)
        decrypted = agent.decrypt(encrypted, key)
        
        security = agent.compute_security(encrypted)
        
        flat_data2 = data.flatten()[:len(decrypted)]
        if len(flat_data2) > 0:
            corr = np.corrcoef(flat_data2, decrypted[:len(flat_data2)])[0, 1]
            decryption_success = not np.isnan(corr) and corr > 0.3
        else:
            decryption_success = False
        
        # 記録更新チェック
        compression_improved = compression_success and compression_ratio > agent.best_compression
        security_improved = decryption_success and security > agent.best_security
        
        if compression_improved:
            agent.best_compression = compression_ratio
            agent.revolutionary_discoveries.append(f"圧縮率{compression_ratio:.2f} ({experiment})")
        
        if security_improved:
            agent.best_security = security
            agent.revolutionary_discoveries.append(f"セキュリティ{security:.2f} ({experiment})")
        
        # 欲を更新
        agent.update_desires(compression_improved, security_improved)
        
        return {
            "agent_id": agent.agent_id,
            "compression_ratio": compression_ratio,
            "compression_success": compression_success,
            "security": security,
            "decryption_success": decryption_success,
            "experiment": experiment,
            "improved": compression_improved or security_improved
        }
    
    def run_round(self) -> List[Dict]:
        """1ラウンド（並列）"""
        self.round_count += 1
        difficulty = min(1.0, 0.3 + 0.001 * self.round_count)
        data = self.generate_data(difficulty)
        
        results = []
        
        # 並列実行
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {executor.submit(self.run_agent_round, agent, data): agent 
                      for agent in self.agents.values()}
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    
                    # スコア更新
                    if result["compression_success"]:
                        self.compression_scores[result["agent_id"]] += result["compression_ratio"]
                    if result["decryption_success"]:
                        self.security_scores[result["agent_id"]] += result["security"]
                except Exception as e:
                    pass  # エラーは無視
        
        # 勝者から学ぶ
        if results:
            best_comp = max(results, key=lambda x: x["compression_ratio"] if x["compression_success"] else 0)
            best_sec = max(results, key=lambda x: x["security"] if x["decryption_success"] else 0)
            
            for agent in self.agents.values():
                if agent.agent_id != best_comp["agent_id"]:
                    agent.learn_from_other(self.agents[best_comp["agent_id"]])
        
        return results
    
    def run_competition(self, rounds: int = 1000, report_every: int = 100):
        """競争実行"""
        print("\n" + "=" * 70)
        print("🚀 革命的発見を求める圧縮・暗号合戦")
        print("=" * 70)
        print(f"エージェント: {', '.join(self.agents.keys())}")
        print(f"ラウンド数: {rounds}")
        print(f"並列ワーカー: {self.n_workers}")
        print("新しい欲: 現状に満足しない + 革命的技術を見つけたい")
        print()
        
        start_time = time.time()
        
        for r in range(rounds):
            results = self.run_round()
            
            # 進捗報告
            if (r + 1) % report_every == 0:
                elapsed = time.time() - start_time
                
                # 最高記録
                best_comp = max(self.agents.values(), key=lambda a: a.best_compression)
                best_sec = max(self.agents.values(), key=lambda a: a.best_security)
                
                # 欲の平均
                avg_dissatisfaction = np.mean([a.dissatisfaction for a in self.agents.values()])
                avg_revolution = np.mean([a.revolution_desire for a in self.agents.values()])
                
                print(f"ラウンド {r+1}/{rounds} ({elapsed:.1f}秒)")
                print(f"  最高圧縮: {best_comp.agent_id}={best_comp.best_compression:.3f}")
                print(f"  最高セキュリティ: {best_sec.agent_id}={best_sec.best_security:.3f}")
                print(f"  不満足度: {avg_dissatisfaction:.2f}, 革命欲: {avg_revolution:.2f}")
                print()
                
                # 履歴に記録
                self.history.append({
                    "round": r + 1,
                    "best_compression": best_comp.best_compression,
                    "best_security": best_sec.best_security,
                    "dissatisfaction": avg_dissatisfaction,
                    "revolution_desire": avg_revolution
                })
        
        total_time = time.time() - start_time
        print(f"\n総実行時間: {total_time:.1f}秒")
        
        self.show_final_results()
    
    def show_final_results(self):
        """最終結果"""
        print("\n" + "=" * 70)
        print("📊 最終結果")
        print("=" * 70)
        
        # 圧縮ランキング
        print("\n【圧縮ランキング】")
        comp_ranking = sorted(self.agents.items(), key=lambda x: x[1].best_compression, reverse=True)
        medals = ["🥇", "🥈", "🥉", "4️⃣"]
        for i, (agent_id, agent) in enumerate(comp_ranking):
            medal = medals[i] if i < len(medals) else f"{i+1}."
            print(f"  {medal} {agent_id}: 最高={agent.best_compression:.3f}")
            print(f"      方法={agent.compression_method}, 閾値={agent.compression_threshold:.3f}")
        
        # 暗号ランキング
        print("\n【暗号ランキング】")
        sec_ranking = sorted(self.agents.items(), key=lambda x: x[1].best_security, reverse=True)
        for i, (agent_id, agent) in enumerate(sec_ranking):
            medal = medals[i] if i < len(medals) else f"{i+1}."
            print(f"  {medal} {agent_id}: 最高={agent.best_security:.3f}")
            print(f"      ノイズ={agent.noise_level:.3f}, ラウンド={agent.encryption_rounds}")
        
        # 革命的発見
        print("\n【革命的発見】")
        for agent_id, agent in self.agents.items():
            if agent.revolutionary_discoveries:
                print(f"  {agent_id} ({len(agent.revolutionary_discoveries)}件):")
                for disc in agent.revolutionary_discoveries[-5:]:
                    print(f"    • {disc}")
        
        # 欲の最終状態
        print("\n【欲の最終状態】")
        for agent_id, agent in self.agents.items():
            print(f"  {agent_id}:")
            print(f"    不満足度: {agent.dissatisfaction:.2f} (初期0.80)")
            print(f"    革命欲: {agent.revolution_desire:.2f} (初期0.90)")
            print(f"    成功率: {agent.successful_experiments}/{agent.total_experiments}")
        
        # 進化曲線
        if self.history:
            print("\n【進化の軌跡】")
            for h in self.history:
                print(f"  {h['round']:5d}ラウンド: 圧縮={h['best_compression']:.3f}, "
                      f"セキュリティ={h['best_security']:.3f}, "
                      f"不満={h['dissatisfaction']:.2f}, 革命欲={h['revolution_desire']:.2f}")


# =============================================================================
# 実行
# =============================================================================

if __name__ == "__main__":
    # 大会作成
    competition = RevolutionaryCompetition(n_workers=4)
    
    # エージェント追加
    competition.add_agent("Pioneer")      # 先駆者
    competition.add_agent("Innovator")    # 革新者
    competition.add_agent("Explorer")     # 探検家
    competition.add_agent("Visionary")    # 先見者
    
    print("エージェントの初期欲:")
    for agent_id, agent in competition.agents.items():
        print(f"  {agent_id}: 不満足度={agent.dissatisfaction:.2f}, 革命欲={agent.revolution_desire:.2f}")
    
    # 1000ラウンド実行！
    competition.run_competition(rounds=1000, report_every=100)
