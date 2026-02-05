"""
自己改変モジュール (Self Modification)
======================================

SNNが自分自身を改変する能力

- SelfModifier: パラメータ・構造を変更
- GoalEngine: 目標を自己設定

Author: ろーる (cell_activation)
Date: 2026-01-31
"""

import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import random


@dataclass
class Goal:
    """自己設定された目標"""
    id: str
    description: str
    metric: str
    target: float
    current: float = 0.0
    priority: float = 0.5
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    achieved: bool = False
    deadline_steps: Optional[int] = None
    
    def progress(self) -> float:
        """進捗率"""
        if self.target == 0:
            return 1.0 if self.current >= self.target else 0.0
        return min(1.0, self.current / self.target)


@dataclass
class ModificationRecord:
    """改変の記録"""
    timestamp: float
    action: str
    details: Dict[str, Any]
    motivation: str
    success: bool = True
    impact: float = 0.0  # 改変の影響度


class SelfModifier:
    """
    自己改変エンジン
    
    SNNのパラメータと構造を動的に変更
    """
    
    def __init__(self):
        self.history: List[ModificationRecord] = []
        self.modification_count = 0
    
    def modify_weights(self, 
                       snn,
                       target_neurons: List[int],
                       mode: str = "strengthen",
                       strength: float = 0.1,
                       motivation: str = "") -> ModificationRecord:
        """重みを修正"""
        n = snn.W.shape[0]
        neurons = [i for i in target_neurons if i < n]
        
        if not neurons:
            return self._record("modify_weights", {"error": "no valid neurons"}, 
                               motivation, success=False)
        
        original_sum = np.sum(np.abs(snn.W[:, neurons]))
        
        if mode == "strengthen":
            snn.W[:, neurons] *= (1 + strength)
        elif mode == "weaken":
            snn.W[:, neurons] *= (1 - strength)
        elif mode == "randomize":
            snn.W[:, neurons] += np.random.randn(n, len(neurons)) * strength
        elif mode == "normalize":
            for i in neurons:
                norm = np.linalg.norm(snn.W[:, i])
                if norm > 0:
                    snn.W[:, i] = snn.W[:, i] / norm * strength
        
        new_sum = np.sum(np.abs(snn.W[:, neurons]))
        impact = abs(new_sum - original_sum) / (original_sum + 0.001)
        
        return self._record(
            "modify_weights",
            {"neurons": neurons, "mode": mode, "strength": strength},
            motivation or f"重みを{mode}",
            impact=impact
        )
    
    def add_neurons(self, 
                    snn,
                    count: int = 1,
                    connection_prob: float = 0.3,
                    motivation: str = "") -> ModificationRecord:
        """ニューロンを追加（神経新生）"""
        old_n = snn.W.shape[0]
        new_n = old_n + count
        
        # 新しい重み行列
        new_W = np.zeros((new_n, new_n))
        new_W[:old_n, :old_n] = snn.W
        
        # 新しいニューロンの接続
        for i in range(old_n, new_n):
            # 入力接続
            mask = np.random.rand(old_n) < connection_prob
            new_W[i, :old_n] = np.random.randn(old_n) * 0.1 * mask
            
            # 出力接続
            mask = np.random.rand(old_n) < connection_prob
            new_W[:old_n, i] = np.random.randn(old_n) * 0.1 * mask
        
        snn.W = new_W
        
        # 状態も拡張
        if hasattr(snn, 'state'):
            snn.state = np.pad(snn.state, (0, count))
        if hasattr(snn, 'n_neurons'):
            snn.n_neurons = new_n
        
        return self._record(
            "add_neurons",
            {"count": count, "new_total": new_n},
            motivation or f"{count}個のニューロンを追加して能力を拡張",
            impact=count / old_n
        )
    
    def prune_neurons(self,
                      snn,
                      threshold: float = 0.01,
                      max_prune: float = 0.1,
                      motivation: str = "") -> ModificationRecord:
        """不要なニューロンを削除（プルーニング）"""
        n = snn.W.shape[0]
        
        # 接続強度を計算
        strength = np.sum(np.abs(snn.W), axis=0) + np.sum(np.abs(snn.W), axis=1)
        
        # 弱いニューロンを検出
        weak = np.where(strength < threshold)[0]
        
        # 最大削除数を制限
        max_count = int(n * max_prune)
        to_prune = weak[:max_count] if len(weak) > max_count else weak
        
        if len(to_prune) > 0:
            # 接続を0に（実際の削除は複雑なので簡略化）
            snn.W[to_prune, :] = 0
            snn.W[:, to_prune] = 0
        
        return self._record(
            "prune_neurons",
            {"pruned": len(to_prune), "threshold": threshold},
            motivation or f"不要な{len(to_prune)}個の接続を削除して効率化",
            impact=len(to_prune) / n if n > 0 else 0
        )
    
    def adjust_threshold(self,
                         snn,
                         delta: float = 0.01,
                         motivation: str = "") -> ModificationRecord:
        """発火閾値を調整"""
        if hasattr(snn, 'threshold'):
            old = snn.threshold
            snn.threshold = np.clip(snn.threshold + delta, 0.1, 0.9)
            new = snn.threshold
            
            return self._record(
                "adjust_threshold",
                {"old": old, "new": new, "delta": delta},
                motivation or f"発火閾値を{delta:+.3f}調整",
                impact=abs(delta)
            )
        
        return self._record("adjust_threshold", {"error": "no threshold"}, 
                           motivation, success=False)
    
    def restructure_connections(self,
                                snn,
                                method: str = "hebbian",
                                strength: float = 0.1,
                                motivation: str = "") -> ModificationRecord:
        """接続パターンを再構築"""
        n = snn.W.shape[0]
        
        if method == "hebbian" and hasattr(snn, 'state'):
            # ヘブ則: 同時に活性化するニューロン間の接続を強化
            activity = snn.state.reshape(-1, 1)
            hebbian = activity @ activity.T
            snn.W += strength * hebbian * 0.01
            
        elif method == "anti_hebbian" and hasattr(snn, 'state'):
            # 反ヘブ則: 競合学習
            activity = snn.state.reshape(-1, 1)
            anti = -activity @ activity.T
            snn.W += strength * anti * 0.01
            
        elif method == "sparse":
            # スパース化
            mask = np.abs(snn.W) > np.percentile(np.abs(snn.W), 70)
            snn.W *= mask
            
        elif method == "noise":
            # ノイズ注入
            snn.W += np.random.randn(n, n) * strength * 0.01
        
        return self._record(
            "restructure",
            {"method": method, "strength": strength},
            motivation or f"{method}法で接続を再構築",
            impact=strength
        )
    
    def _record(self, action: str, details: Dict, motivation: str,
                success: bool = True, impact: float = 0.0) -> ModificationRecord:
        """記録を作成"""
        record = ModificationRecord(
            timestamp=datetime.now().timestamp(),
            action=action,
            details=details,
            motivation=motivation,
            success=success,
            impact=impact
        )
        self.history.append(record)
        self.modification_count += 1
        return record


class GoalEngine:
    """
    目標設定エンジン
    
    SNNが自分で目標を設定し、追跡する
    """
    
    def __init__(self):
        self.goals: List[Goal] = []
        self.achieved: List[Goal] = []
        self.goal_counter = 0
    
    def generate_goal(self,
                      capabilities: Dict[str, float],
                      motivation_state,
                      context: str = "") -> Goal:
        """内発的動機に基づいて目標を生成"""
        self.goal_counter += 1
        
        # 最も弱い能力を改善する目標
        if capabilities:
            weakest = min(capabilities.items(), key=lambda x: x[1])
            metric = weakest[0]
            current = weakest[1]
            target = min(1.0, current + 0.15)
            desc = f"{metric}を{current:.0%}から{target:.0%}に向上"
        else:
            metric = "general"
            current = 0.5
            target = 0.7
            desc = "全体的なパフォーマンスを向上"
        
        # 好奇心が高い場合は探索的な目標
        if motivation_state.curiosity > 0.7:
            desc = "新しいパターンを5個発見する"
            metric = "novelty_count"
            target = 5
            current = 0
        
        # 退屈な場合は挑戦的な目標
        if motivation_state.boredom > 0.6:
            target = min(1.0, target + 0.1)
            desc = f"【挑戦】{desc}"
        
        goal = Goal(
            id=f"goal_{self.goal_counter}",
            description=desc,
            metric=metric,
            target=target,
            current=current,
            priority=motivation_state.evolution_drive()
        )
        
        self.goals.append(goal)
        return goal
    
    def update_progress(self, metric: str, value: float):
        """目標の進捗を更新"""
        for goal in self.goals:
            if not goal.achieved and goal.metric == metric:
                goal.current = value
                
                if goal.current >= goal.target:
                    goal.achieved = True
                    self.achieved.append(goal)
    
    def get_active_goals(self) -> List[Goal]:
        """アクティブな目標を取得"""
        return [g for g in self.goals if not g.achieved]
    
    def get_priority_goal(self) -> Optional[Goal]:
        """最優先の目標を取得"""
        active = self.get_active_goals()
        if not active:
            return None
        return max(active, key=lambda g: g.priority)
    
    def report(self) -> str:
        """目標のレポート"""
        lines = ["=== 目標レポート ==="]
        
        active = self.get_active_goals()
        if active:
            lines.append(f"\n【進行中】{len(active)}個")
            for g in active[:3]:
                lines.append(f"  • {g.description} ({g.progress():.0%})")
        
        if self.achieved:
            lines.append(f"\n【達成済み】{len(self.achieved)}個")
            for g in self.achieved[-3:]:
                lines.append(f"  ✅ {g.description}")
        
        return "\n".join(lines)
