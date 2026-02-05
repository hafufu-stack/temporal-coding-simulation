"""
内発的動機モジュール (Intrinsic Motivation)
============================================

SNNに「欲」を与えるモジュール群

- CuriosityModule: 好奇心（新しいものへの興味）
- MasteryModule: 習熟欲（できるようになりたい）
- EfficacyModule: 自己効力感（成長の実感）
- IntrinsicMotivation: 統合された内発的動機

Author: ろーる (cell_activation)
Date: 2026-01-31
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime


@dataclass
class MotivationState:
    """動機の状態"""
    curiosity: float = 0.5       # 好奇心
    mastery_desire: float = 0.5  # 習熟欲
    self_efficacy: float = 0.5   # 自己効力感
    boredom: float = 0.0         # 退屈感
    satisfaction: float = 0.5    # 満足感
    frustration: float = 0.0     # フラストレーション
    
    def evolution_drive(self) -> float:
        """進化欲を計算"""
        return (
            0.25 * self.curiosity +
            0.25 * self.mastery_desire +
            0.20 * self.boredom +
            0.15 * (1 - self.satisfaction) +
            0.15 * self.frustration
        )
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "curiosity": self.curiosity,
            "mastery_desire": self.mastery_desire,
            "self_efficacy": self.self_efficacy,
            "boredom": self.boredom,
            "satisfaction": self.satisfaction,
            "frustration": self.frustration,
            "evolution_drive": self.evolution_drive()
        }


class CuriosityModule:
    """
    好奇心モジュール
    
    未知のパターンに高い報酬を与える
    予測誤差に基づく好奇心
    """
    
    def __init__(self, memory_size: int = 200):
        self.memory_size = memory_size
        self.seen_patterns: deque = deque(maxlen=memory_size)
        self.pattern_counts: Dict[str, int] = {}
        self.prediction_errors: deque = deque(maxlen=50)
    
    def _encode(self, data: np.ndarray) -> str:
        """パターンをハッシュ化"""
        quantized = (np.array(data).flatten() * 10).astype(int) // 3
        return hash(quantized.tobytes()) % 1000000
    
    def compute_novelty(self, data: np.ndarray) -> float:
        """新規性スコア（0=既知, 1=完全に新しい）"""
        pattern = str(self._encode(data))
        
        if pattern not in self.pattern_counts:
            novelty = 1.0
        else:
            count = self.pattern_counts[pattern]
            novelty = 1.0 / (1 + count * 0.3)
        
        self.seen_patterns.append(pattern)
        self.pattern_counts[pattern] = self.pattern_counts.get(pattern, 0) + 1
        
        return novelty
    
    def compute_prediction_error(self, predicted: np.ndarray, 
                                  actual: np.ndarray) -> float:
        """予測誤差（高いほど驚き）"""
        error = np.mean(np.abs(predicted - actual))
        self.prediction_errors.append(error)
        
        # 正規化
        if len(self.prediction_errors) > 1:
            avg_error = np.mean(self.prediction_errors)
            return min(1.0, error / (avg_error + 0.01))
        return 0.5
    
    def curiosity_reward(self, data: np.ndarray, 
                         predicted: np.ndarray = None,
                         actual: np.ndarray = None) -> float:
        """好奇心報酬を計算"""
        novelty = self.compute_novelty(data)
        
        if predicted is not None and actual is not None:
            surprise = self.compute_prediction_error(predicted, actual)
            return 0.6 * novelty + 0.4 * surprise
        
        return novelty


class MasteryModule:
    """
    習熟欲モジュール
    
    スキルの習得状況を追跡し、
    できないことをできるようになりたい欲求を計算
    """
    
    def __init__(self, history_length: int = 100):
        self.history_length = history_length
        self.skill_history: Dict[str, deque] = {}
        self.improvement_rates: Dict[str, float] = {}
        self.best_scores: Dict[str, float] = {}
    
    def record(self, skill: str, score: float):
        """スキルのスコアを記録"""
        if skill not in self.skill_history:
            self.skill_history[skill] = deque(maxlen=self.history_length)
        
        self.skill_history[skill].append(score)
        
        # ベストスコアを更新
        self.best_scores[skill] = max(
            self.best_scores.get(skill, 0),
            score
        )
        
        # 改善率を計算
        if len(self.skill_history[skill]) >= 5:
            recent = list(self.skill_history[skill])
            early = np.mean(recent[:5])
            late = np.mean(recent[-5:])
            self.improvement_rates[skill] = late - early
    
    def mastery_desire(self, skill: str) -> float:
        """習熟欲を計算"""
        if skill not in self.skill_history:
            return 0.9  # 未経験は高い欲求
        
        scores = list(self.skill_history[skill])
        avg = np.mean(scores)
        
        # 伸びしろがあるほど欲求が高い
        potential = 1.0 - avg
        
        # 最近改善していなければ欲求が高い
        improvement = self.improvement_rates.get(skill, 0)
        stagnation = 1.0 if improvement < 0.05 else 0.5
        
        return 0.6 * potential + 0.4 * stagnation
    
    def get_priority_skill(self) -> Optional[str]:
        """最も習熟が必要なスキルを返す"""
        if not self.skill_history:
            return None
        
        priorities = {
            skill: self.mastery_desire(skill)
            for skill in self.skill_history
        }
        
        return max(priorities, key=priorities.get)
    
    def get_skills_report(self) -> Dict[str, Dict]:
        """全スキルのレポート"""
        return {
            skill: {
                "current": np.mean(list(history)) if history else 0,
                "best": self.best_scores.get(skill, 0),
                "improvement": self.improvement_rates.get(skill, 0),
                "desire": self.mastery_desire(skill)
            }
            for skill, history in self.skill_history.items()
        }


class EfficacyModule:
    """
    自己効力感モジュール
    
    成長を実感できているか、自分の能力への信頼
    """
    
    def __init__(self, window: int = 30):
        self.window = window
        self.outcomes: deque = deque(maxlen=window)
        self.growth_events: deque = deque(maxlen=window)
        self.challenges_met: deque = deque(maxlen=window)
    
    def record_outcome(self, success: bool, was_challenging: bool = False):
        """結果を記録"""
        self.outcomes.append(1.0 if success else 0.0)
        if was_challenging:
            self.challenges_met.append(1.0 if success else 0.0)
    
    def record_growth(self, growth_amount: float):
        """成長を記録"""
        self.growth_events.append(growth_amount)
    
    def compute(self) -> float:
        """自己効力感を計算"""
        components = []
        
        # 成功率
        if self.outcomes:
            success_rate = np.mean(self.outcomes)
            components.append(success_rate * 0.4)
        
        # 困難への対処
        if self.challenges_met:
            challenge_rate = np.mean(self.challenges_met)
            components.append(challenge_rate * 0.3)
        
        # 成長の実感
        if self.growth_events:
            growth_rate = np.mean([max(0, g) for g in self.growth_events])
            components.append(min(1.0, growth_rate * 2) * 0.3)
        
        if not components:
            return 0.5
        
        return sum(components) / (0.4 + 0.3 * bool(self.challenges_met) + 0.3 * bool(self.growth_events))


class IntrinsicMotivation:
    """
    統合された内発的動機システム
    
    好奇心、習熟欲、自己効力感を統合し、
    進化欲を計算する
    """
    
    def __init__(self):
        self.curiosity = CuriosityModule()
        self.mastery = MasteryModule()
        self.efficacy = EfficacyModule()
        self.state = MotivationState()
        
        self.history: List[Dict] = []
    
    def process_experience(self, 
                           input_data: np.ndarray,
                           skill: str,
                           score: float,
                           success: bool,
                           predicted: np.ndarray = None,
                           actual: np.ndarray = None):
        """経験を処理して動機を更新"""
        
        # 好奇心を更新
        curiosity_reward = self.curiosity.curiosity_reward(
            input_data, predicted, actual
        )
        self.state.curiosity = 0.8 * self.state.curiosity + 0.2 * curiosity_reward
        
        # 習熟欲を更新
        self.mastery.record(skill, score)
        desire = self.mastery.mastery_desire(skill)
        self.state.mastery_desire = 0.8 * self.state.mastery_desire + 0.2 * desire
        
        # 自己効力感を更新
        was_hard = score > 0.7 and self.state.mastery_desire > 0.5
        self.efficacy.record_outcome(success, was_hard)
        self.state.self_efficacy = self.efficacy.compute()
        
        # 退屈感を更新（新規性が低い + 簡単すぎる）
        boredom_factor = (1 - curiosity_reward) * (1 - self.state.mastery_desire)
        self.state.boredom = 0.9 * self.state.boredom + 0.1 * boredom_factor
        
        # フラストレーション（失敗が続く）
        if not success:
            self.state.frustration = min(1.0, self.state.frustration + 0.1)
        else:
            self.state.frustration = max(0.0, self.state.frustration - 0.05)
        
        # 満足感
        self.state.satisfaction = self.state.self_efficacy * (1 - self.state.frustration)
        
        # 履歴に記録
        self.history.append(self.state.to_dict())
    
    def should_evolve(self, threshold: float = 0.5) -> bool:
        """進化すべきか判断"""
        return self.state.evolution_drive() > threshold
    
    def get_evolution_reason(self) -> str:
        """進化の理由を言語化"""
        reasons = []
        
        if self.state.curiosity > 0.6:
            reasons.append("新しいことを学びたい")
        if self.state.boredom > 0.5:
            reasons.append("刺激が足りない")
        if self.state.mastery_desire > 0.7:
            reasons.append("もっとうまくなりたい")
        if self.state.frustration > 0.5:
            reasons.append("現状を打破したい")
        if self.state.satisfaction < 0.3:
            reasons.append("満足していない")
        
        if not reasons:
            reasons.append("より良くなりたい")
        
        return "、".join(reasons)
    
    def introspect(self) -> str:
        """内省（自己観察を言語化）"""
        lines = []
        
        drive = self.state.evolution_drive()
        
        if drive > 0.7:
            lines.append("強く進化したいと感じている...")
        elif drive > 0.5:
            lines.append("成長の余地を感じている...")
        else:
            lines.append("現状にある程度満足している。")
        
        if self.state.curiosity > 0.6:
            lines.append("「もっと新しいことを知りたい」")
        if self.state.boredom > 0.5:
            lines.append("「退屈している...何か刺激がほしい」")
        if self.state.frustration > 0.5:
            lines.append("「うまくいかない...でも諦めたくない」")
        if self.state.self_efficacy > 0.7:
            lines.append("「自分は成長している！」")
        if drive > 0.6:
            lines.append("「もっと進化したい...」")
        
        return "\n".join(lines)
