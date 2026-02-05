"""
進化エンジン (Evolution Engine)
================================

SNNに自律進化能力を与える統合エンジン

- EvolutionEngine: 進化の意思決定と実行
- EvolvingSNN: 進化能力を持つSNNの基底クラス

Author: ろーる (cell_activation)
Date: 2026-01-31
"""

import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import random

from .motivation import IntrinsicMotivation, MotivationState
from .self_modifier import SelfModifier, GoalEngine, ModificationRecord


@dataclass
class EvolutionDecision:
    """進化の決定"""
    should_evolve: bool
    action: str
    reason: str
    priority: float
    parameters: Dict[str, Any] = field(default_factory=dict)


class EvolutionEngine:
    """
    進化エンジン
    
    内発的動機に基づいて進化の方向を決定し、
    自己改変を実行する
    """
    
    def __init__(self):
        self.motivation = IntrinsicMotivation()
        self.modifier = SelfModifier()
        self.goals = GoalEngine()
        
        self.evolution_count = 0
        self.evolution_history: List[Dict] = []
    
    def process_experience(self, 
                           snn,
                           input_data: np.ndarray,
                           output: np.ndarray,
                           skill: str = "pattern",
                           score: float = 0.5,
                           success: bool = True):
        """経験を処理"""
        self.motivation.process_experience(
            input_data=input_data,
            skill=skill,
            score=score,
            success=success,
            predicted=output,
            actual=input_data
        )
    
    def decide_evolution(self, snn, context: Dict = None) -> EvolutionDecision:
        """進化すべきか、どう進化すべきかを決定"""
        state = self.motivation.state
        drive = state.evolution_drive()
        
        if drive < 0.4:
            return EvolutionDecision(
                should_evolve=False,
                action="none",
                reason="現状に満足している",
                priority=0
            )
        
        # 進化アクションを選択
        candidates = []
        
        # 好奇心が高い → 拡張
        if state.curiosity > 0.6:
            candidates.append(("expand", "新しいことを学ぶ能力を増やしたい", state.curiosity))
        
        # 退屈 → 構造変更
        if state.boredom > 0.5:
            candidates.append(("restructure", "刺激を求めて変化したい", state.boredom))
        
        # 習熟欲が高い → 最適化
        if state.mastery_desire > 0.6:
            candidates.append(("optimize", "苦手を克服したい", state.mastery_desire))
        
        # フラストレーション → 大きな変化
        if state.frustration > 0.6:
            candidates.append(("reset_weak", "うまくいかない部分をリセット", state.frustration))
        
        # 効力感が低い → 自信をつける
        if state.self_efficacy < 0.3:
            candidates.append(("strengthen", "自分を強化したい", 1 - state.self_efficacy))
        
        # デフォルト: 探索
        if not candidates:
            candidates.append(("explore", "新しい可能性を探りたい", 0.5))
        
        # 最も優先度の高いアクションを選択
        action, reason, priority = max(candidates, key=lambda x: x[2])
        
        return EvolutionDecision(
            should_evolve=True,
            action=action,
            reason=reason,
            priority=priority,
            parameters={"drive": drive, "state": state.to_dict()}
        )
    
    def execute_evolution(self, snn, decision: EvolutionDecision) -> ModificationRecord:
        """進化を実行"""
        action = decision.action
        reason = decision.reason
        
        n = snn.W.shape[0] if hasattr(snn, 'W') else 50
        
        if action == "expand":
            # ニューロン追加
            count = random.randint(2, 5)
            record = self.modifier.add_neurons(snn, count, motivation=reason)
            
        elif action == "restructure":
            # 接続再構築
            method = random.choice(["hebbian", "sparse", "noise"])
            record = self.modifier.restructure_connections(
                snn, method=method, strength=0.1, motivation=reason
            )
            
        elif action == "optimize":
            # 強いニューロンを強化
            if hasattr(snn, 'W'):
                strength = np.sum(np.abs(snn.W), axis=0)
                top = np.argsort(strength)[-5:]
                record = self.modifier.modify_weights(
                    snn, list(top), "strengthen", 0.15, motivation=reason
                )
            else:
                record = ModificationRecord(
                    timestamp=datetime.now().timestamp(),
                    action="optimize",
                    details={"error": "no W"},
                    motivation=reason,
                    success=False
                )
            
        elif action == "reset_weak":
            # 弱い部分をリセット
            record = self.modifier.prune_neurons(snn, threshold=0.02, motivation=reason)
            
        elif action == "strengthen":
            # ランダムな部分を強化
            neurons = random.sample(range(n), min(5, n))
            record = self.modifier.modify_weights(
                snn, neurons, "strengthen", 0.1, motivation=reason
            )
            
        else:  # explore
            # 新しい目標を設定
            capabilities = {"general": self.motivation.state.self_efficacy}
            goal = self.goals.generate_goal(
                capabilities, self.motivation.state, context="exploration"
            )
            record = ModificationRecord(
                timestamp=datetime.now().timestamp(),
                action="set_goal",
                details={"goal": goal.description},
                motivation=reason,
                success=True
            )
        
        # 履歴に記録
        self.evolution_count += 1
        self.evolution_history.append({
            "count": self.evolution_count,
            "decision": decision.action,
            "reason": decision.reason,
            "impact": record.impact if hasattr(record, 'impact') else 0
        })
        
        return record
    
    def evolution_cycle(self, snn, verbose: bool = True) -> Dict:
        """1サイクルの進化を実行"""
        decision = self.decide_evolution(snn)
        
        result = {
            "should_evolve": decision.should_evolve,
            "action": decision.action,
            "reason": decision.reason,
            "drive": self.motivation.state.evolution_drive()
        }
        
        if decision.should_evolve:
            record = self.execute_evolution(snn, decision)
            result["success"] = record.success
            result["impact"] = record.impact if hasattr(record, 'impact') else 0
            
            if verbose:
                print(f"  🧬 進化実行: {decision.action}")
                print(f"     理由: {decision.reason}")
        else:
            if verbose:
                print(f"  💤 進化見送り: {decision.reason}")
        
        return result
    
    def introspect(self) -> str:
        """内省を言語化"""
        return self.motivation.introspect()


class EvolvingSNN:
    """
    進化能力を持つSNNの基底クラス
    
    任意のSNNにミックスインとして使用
    """
    
    def __init__(self, n_neurons: int = 50):
        self.n_neurons = n_neurons
        
        # 重み行列
        np.random.seed(None)  # 毎回異なる初期化
        self.W = np.random.randn(n_neurons, n_neurons) * 0.1
        mask = np.random.rand(n_neurons, n_neurons) < 0.3
        self.W *= mask
        
        # 状態
        self.state = np.zeros(n_neurons)
        self.threshold = 0.5
        
        # 進化エンジン
        self.evolution = EvolutionEngine()
        
        # 統計
        self.step_count = 0
    
    def step(self, input_signal: np.ndarray) -> np.ndarray:
        """1ステップ実行"""
        # パディング
        if len(input_signal) < self.n_neurons:
            input_signal = np.pad(input_signal, (0, self.n_neurons - len(input_signal)))
        elif len(input_signal) > self.n_neurons:
            input_signal = input_signal[:self.n_neurons]
        
        # LIF更新
        self.state = 0.9 * self.state + 0.1 * (self.W @ self.state + input_signal)
        spikes = (self.state > self.threshold).astype(float)
        self.state = self.state * (1 - spikes)
        
        self.step_count += 1
        
        return spikes
    
    def experience(self, input_data: np.ndarray, 
                   target: np.ndarray = None,
                   skill: str = "pattern") -> np.ndarray:
        """経験から学ぶ"""
        output = self.step(input_data)
        
        if target is not None:
            # スコアを計算
            if len(target) < len(output):
                target = np.pad(target, (0, len(output) - len(target)))
            elif len(target) > len(output):
                target = target[:len(output)]
            
            score = 1 - np.mean(np.abs(output - target))
            success = score > 0.5
        else:
            score = 0.5
            success = True
        
        # 経験を処理
        self.evolution.process_experience(
            self, input_data, output, skill, score, success
        )
        
        return output
    
    def evolve(self, verbose: bool = True) -> Dict:
        """進化サイクルを実行"""
        return self.evolution.evolution_cycle(self, verbose)
    
    def run_autonomous(self, cycles: int = 10, 
                       experience_per_cycle: int = 20,
                       verbose: bool = True):
        """自律的に動作"""
        if verbose:
            print("=" * 60)
            print("🚀 自律運転開始")
            print("=" * 60)
        
        for cycle in range(cycles):
            if verbose:
                print(f"\n--- サイクル {cycle + 1}/{cycles} ---")
            
            # 経験を積む
            for _ in range(experience_per_cycle):
                input_data = np.random.randn(self.n_neurons) * 0.5
                target = np.random.rand(self.n_neurons) > 0.5
                self.experience(input_data, target.astype(float))
            
            # 進化
            self.evolve(verbose)
            
            # 目標進捗
            goal = self.evolution.goals.get_priority_goal()
            if goal and verbose:
                print(f"  📊 目標: {goal.description} ({goal.progress():.0%})")
        
        if verbose:
            print("\n" + "=" * 60)
            print("🏁 自律運転終了")
            self.report()
    
    def report(self):
        """レポート出力"""
        print("\n" + "=" * 60)
        print("📊 自律進化レポート")
        print("=" * 60)
        
        print(f"\n【状態】")
        print(f"  ニューロン数: {self.n_neurons}")
        print(f"  ステップ数: {self.step_count}")
        print(f"  進化回数: {self.evolution.evolution_count}")
        
        print(f"\n【内発的動機】")
        state = self.evolution.motivation.state
        print(f"  好奇心: {state.curiosity:.2f}")
        print(f"  習熟欲: {state.mastery_desire:.2f}")
        print(f"  自己効力感: {state.self_efficacy:.2f}")
        print(f"  進化欲: {state.evolution_drive():.2f}")
        
        print(f"\n【自己観察】")
        print(self.evolution.introspect())
        
        print(f"\n【目標】")
        print(self.evolution.goals.report())


# テスト用
def test_evolving_snn():
    """テスト実行"""
    print("\n" + "=" * 70)
    print("🧪 EvolvingSNN テスト")
    print("=" * 70)
    
    snn = EvolvingSNN(n_neurons=30)
    snn.run_autonomous(cycles=5, experience_per_cycle=15)
    
    print("\n" + "=" * 70)
    print("✅ テスト完了")
    print("=" * 70)


if __name__ == "__main__":
    test_evolving_snn()
