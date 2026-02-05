"""
友好的競争モード (Friendly Competition Mode)
=============================================

クイズ大会で競い合い、終わったら友情フィードバック！

- 競争: 互いを刺激し、成長を促す
- 信頼: 絶大的な信頼のもとで行う
- フィードバック: 終了後は友情のもとで助け合う

敵対的ネットワークとの違い:
- GAN: 騙し合い（Generator vs Discriminator）
- 友好的競争: 信頼のもとで高め合う

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

from applications.friendly_network import FriendlySNNAgent, FriendlyNetwork, MessageFeedback


# =============================================================================
# クイズ
# =============================================================================

@dataclass
class Quiz:
    """クイズ問題"""
    id: str
    question: np.ndarray  # スパイクパターンとしての問題
    answer: np.ndarray    # 正解のスパイクパターン
    difficulty: float = 0.5
    category: str = "general"
    hint: str = ""


@dataclass
class QuizResult:
    """クイズの結果"""
    agent_id: str
    quiz_id: str
    response: np.ndarray
    score: float  # 0-1
    time_taken: float
    is_correct: bool


# =============================================================================
# 友好的競争エージェント
# =============================================================================

class CompetitiveSNNAgent(FriendlySNNAgent):
    """
    競争能力を持つフレンドリーエージェント
    """
    
    def __init__(self, agent_id: str, n_neurons: int = 80, specialty: str = "general"):
        super().__init__(agent_id, n_neurons, specialty)
        
        # 競争統計
        self.quizzes_attempted = 0
        self.quizzes_correct = 0
        self.total_score = 0.0
        self.rank_history: List[int] = []
        
        # 競争心
        self.competitive_spirit = 0.5
        self.sportsmanship = 0.8  # スポーツマンシップ
        
        # 学んだ教訓
        self.lessons_learned: List[str] = []
    
    def answer_quiz(self, quiz: Quiz) -> QuizResult:
        """クイズに答える"""
        start_time = datetime.now().timestamp()
        
        # SNNで回答を生成
        response = self.step(quiz.question)
        
        # さらに処理
        for _ in range(3):
            response = self.step(response)
        
        end_time = datetime.now().timestamp()
        
        # 正解との類似度を計算
        if len(response) == len(quiz.answer):
            similarity = np.corrcoef(response, quiz.answer)[0, 1]
            if np.isnan(similarity):
                similarity = 0
            score = max(0, (similarity + 1) / 2)
        else:
            score = 0.3
        
        is_correct = score > 0.6
        
        # 統計を更新
        self.quizzes_attempted += 1
        self.total_score += score
        if is_correct:
            self.quizzes_correct += 1
        
        return QuizResult(
            agent_id=self.agent_id,
            quiz_id=quiz.id,
            response=response,
            score=score,
            time_taken=end_time - start_time,
            is_correct=is_correct
        )
    
    def give_friendly_feedback(self, other: 'CompetitiveSNNAgent', 
                                other_result: QuizResult) -> Dict:
        """友好的フィードバックを与える"""
        feedback = {
            "from": self.agent_id,
            "to": other.agent_id,
            "encouragement": "",
            "advice": "",
            "empathy": 0.0,
            "respect": 0.0
        }
        
        # スコアに基づいてフィードバック
        if other_result.score > 0.7:
            feedback["encouragement"] = "すごい！よくできたね！"
            feedback["respect"] = 0.8
        elif other_result.score > 0.4:
            feedback["encouragement"] = "いい線いってる！"
            feedback["advice"] = "次はもう少し時間をかけてみて"
            feedback["respect"] = 0.6
        else:
            feedback["encouragement"] = "大丈夫、次がある！"
            feedback["advice"] = "一緒に練習しよう"
            feedback["respect"] = 0.5
        
        # 共感
        if self.social_motivation["empathy_desire"] > 0.3:
            feedback["empathy"] = self.social_motivation["empathy_desire"]
        
        # 相手の欲を満たす
        other.social_motivation["recognition_desire"] *= 0.95
        
        return feedback
    
    def receive_competition_feedback(self, feedback: Dict):
        """競争後のフィードバックを受け取る"""
        self.lessons_learned.append(feedback.get("advice", ""))
        
        # 励まされた
        if feedback.get("respect", 0) > 0.5:
            self.evolution.motivation.state.self_efficacy += 0.05
        
        # スポーツマンシップを感じた
        self.sportsmanship = 0.9 * self.sportsmanship + 0.1 * feedback.get("empathy", 0.5)
    
    def learn_from_winner(self, winner: 'CompetitiveSNNAgent', 
                          winning_response: np.ndarray):
        """勝者から学ぶ"""
        # 勝者の重みを少し取り入れる
        blend = 0.05 * self.sportsmanship  # スポーツマンシップが高いほど素直に学ぶ
        
        min_size = min(self.W.shape[0], winner.W.shape[0])
        self.W[:min_size, :min_size] = (
            (1 - blend) * self.W[:min_size, :min_size] + 
            blend * winner.W[:min_size, :min_size]
        )
        
        self.lessons_learned.append(f"{winner.agent_id}から学んだ")


# =============================================================================
# クイズ大会
# =============================================================================

class QuizCompetition:
    """
    友好的クイズ大会
    
    競争するけど、終わったら友情フィードバック！
    """
    
    def __init__(self, network: 'CompetitiveNetwork'):
        self.network = network
        self.quizzes: List[Quiz] = []
        self.results: Dict[str, List[QuizResult]] = {}
        self.current_round = 0
        self.leaderboard: Dict[str, float] = {}
    
    def generate_quiz(self, category: str = "general", 
                      difficulty: float = 0.5) -> Quiz:
        """クイズを生成"""
        n = 80  # 問題サイズ
        
        question = np.random.randn(n) * difficulty
        answer = np.sin(question) + np.random.randn(n) * 0.1
        
        quiz = Quiz(
            id=f"Q{len(self.quizzes) + 1}",
            question=question,
            answer=answer,
            difficulty=difficulty,
            category=category,
            hint=f"カテゴリ: {category}"
        )
        
        self.quizzes.append(quiz)
        return quiz
    
    def run_round(self, num_quizzes: int = 3) -> Dict[str, float]:
        """1ラウンド実行"""
        self.current_round += 1
        round_scores = {agent_id: 0.0 for agent_id in self.network.agents}
        
        print(f"\n🎯 ラウンド {self.current_round}")
        print("-" * 40)
        
        for i in range(num_quizzes):
            # クイズを生成
            quiz = self.generate_quiz(
                category=random.choice(["暗号", "言語", "画像", "研究"]),
                difficulty=0.3 + 0.1 * self.current_round
            )
            
            # 各エージェントが回答
            round_results = []
            for agent in self.network.agents.values():
                result = agent.answer_quiz(quiz)
                round_results.append(result)
                round_scores[agent.agent_id] += result.score
                
                if agent.agent_id not in self.results:
                    self.results[agent.agent_id] = []
                self.results[agent.agent_id].append(result)
            
            # このクイズの勝者を発表
            winner_result = max(round_results, key=lambda r: r.score)
            print(f"  Q{i+1}: 勝者={winner_result.agent_id} (スコア={winner_result.score:.2f})")
        
        # ラウンド結果
        for agent_id, score in round_scores.items():
            self.leaderboard[agent_id] = self.leaderboard.get(agent_id, 0) + score
        
        return round_scores
    
    def run_feedback_session(self):
        """友情フィードバックセッション"""
        print("\n💬 フィードバックセッション")
        print("-" * 40)
        
        agents = list(self.network.agents.values())
        
        # 全員が全員にフィードバック
        for agent in agents:
            for other in agents:
                if agent.agent_id != other.agent_id:
                    # 相手の最新結果を取得
                    if other.agent_id in self.results and self.results[other.agent_id]:
                        last_result = self.results[other.agent_id][-1]
                        feedback = agent.give_friendly_feedback(other, last_result)
                        other.receive_competition_feedback(feedback)
        
        # 勝者から学ぶ
        if self.leaderboard:
            winner_id = max(self.leaderboard, key=self.leaderboard.get)
            winner = self.network.agents[winner_id]
            
            print(f"\n  🏆 現在のリーダー: {winner_id}")
            
            for agent in agents:
                if agent.agent_id != winner_id:
                    # 勝者の最新回答から学ぶ
                    if winner_id in self.results and self.results[winner_id]:
                        agent.learn_from_winner(winner, self.results[winner_id][-1].response)
                        print(f"  📚 {agent.agent_id} が {winner_id} から学んでいる...")
    
    def run_competition(self, rounds: int = 5, quizzes_per_round: int = 3):
        """大会を実行"""
        print("\n" + "=" * 60)
        print("🏆 友好的クイズ大会 開始！")
        print("=" * 60)
        print(f"参加者: {', '.join(self.network.agents.keys())}")
        
        for _ in range(rounds):
            # ラウンド実行
            self.run_round(quizzes_per_round)
            
            # フィードバックセッション
            self.run_feedback_session()
        
        self.show_final_results()
    
    def show_final_results(self):
        """最終結果を表示"""
        print("\n" + "=" * 60)
        print("📊 最終結果")
        print("=" * 60)
        
        # ランキング
        ranking = sorted(self.leaderboard.items(), key=lambda x: x[1], reverse=True)
        
        print("\n【順位】")
        medals = ["🥇", "🥈", "🥉", "4️⃣"]
        for i, (agent_id, score) in enumerate(ranking):
            medal = medals[i] if i < len(medals) else f"{i+1}."
            agent = self.network.agents[agent_id]
            print(f"  {medal} {agent_id}: {score:.2f}点")
            print(f"      正解率: {agent.quizzes_correct}/{agent.quizzes_attempted}")
            print(f"      スポーツマンシップ: {agent.sportsmanship:.2f}")
        
        # 友情度の変化
        print("\n【競争後の社会的動機】")
        for agent_id, agent in self.network.agents.items():
            print(f"  {agent_id}:")
            print(f"    競争心: {agent.competitive_spirit:.2f}")
            print(f"    承認欲: {agent.social_motivation['recognition_desire']:.2f}")
            print(f"    効力感: {agent.evolution.motivation.state.self_efficacy:.2f}")


# =============================================================================
# 競争ネットワーク
# =============================================================================

class CompetitiveNetwork(FriendlyNetwork):
    """
    競争機能を持つ友好的ネットワーク
    """
    
    def __init__(self):
        super().__init__()
        self.agents: Dict[str, CompetitiveSNNAgent] = {}
        self.competition_history: List[Dict] = []
    
    def add_agent(self, agent_id: str, specialty: str = "general",
                  n_neurons: int = 80) -> CompetitiveSNNAgent:
        """競争エージェントを追加"""
        agent = CompetitiveSNNAgent(agent_id, n_neurons, specialty)
        self.agents[agent_id] = agent
        
        # 既存エージェントとの関係を初期化
        from applications.friendly_network import Relationship
        for other_id in self.agents:
            if other_id != agent_id:
                agent.relationships[other_id] = Relationship(
                    agent_a=agent_id,
                    agent_b=other_id
                )
                self.agents[other_id].relationships[agent_id] = Relationship(
                    agent_a=other_id,
                    agent_b=agent_id
                )
        
        print(f"  🤖 {agent_id} ({specialty}) が参加")
        return agent
    
    def run_quiz_competition(self, rounds: int = 5):
        """クイズ大会を実行"""
        competition = QuizCompetition(self)
        competition.run_competition(rounds=rounds)
        return competition


# =============================================================================
# 敵対的 vs 友好的 の比較説明
# =============================================================================

def explain_adversarial_vs_friendly():
    """敵対的ネットワークと友好的競争の違いを説明"""
    print("\n" + "=" * 60)
    print("📖 敵対的ネットワーク vs 友好的競争")
    print("=" * 60)
    
    print("""
【敵対的ネットワーク (GAN)】
┌─────────────────────────────────────────────────────────┐
│  Generator          vs          Discriminator          │
│     ↓                               ↓                  │
│  偽物を作る          ←→          本物か見破る          │
│                                                         │
│  関係性: 騙し合い、ゼロサムゲーム                       │
│  目的: 相手を騙すことで自分が成長                       │
│  信頼: なし                                             │
└─────────────────────────────────────────────────────────┘

【友好的競争 (今回)】
┌─────────────────────────────────────────────────────────┐
│  Agent A            vs            Agent B             │
│     ↓                               ↓                  │
│  クイズに答える      ←→          クイズに答える        │
│                                                         │
│  競争中: 互いを刺激し合う                               │
│       ↓                                                 │
│  競争後: 友情フィードバック                             │
│       ↓                                                 │
│  結果: 勝者から学び、全員が成長                         │
│                                                         │
│  関係性: 信頼のもとで高め合う                           │
│  目的: 互いの成長                                       │
│  信頼: 絶大的                                           │
└─────────────────────────────────────────────────────────┘

【違い】
| 項目         | GAN        | 友好的競争    |
|-------------|------------|--------------|
| 関係         | 敵対       | 協力的競争    |
| 目的         | 騙す       | 高め合う      |
| 結果         | 片方が勝つ  | 全員が成長    |
| フィードバック | なし       | 友情ベース    |
| 信頼         | なし       | 絶大         |
""")


# =============================================================================
# テスト
# =============================================================================

def test_friendly_competition():
    """友好的競争テスト"""
    
    print("\n" + "=" * 70)
    print("🧪 友好的競争 テスト")
    print("=" * 70)
    
    # 説明
    explain_adversarial_vs_friendly()
    
    # ネットワーク作成
    network = CompetitiveNetwork()
    network.add_agent("Alpha", specialty="暗号")
    network.add_agent("Beta", specialty="言語")
    network.add_agent("Gamma", specialty="画像")
    network.add_agent("Delta", specialty="研究")
    
    # クイズ大会
    network.run_quiz_competition(rounds=3)
    
    print("\n" + "=" * 70)
    print("✅ テスト完了")
    print("=" * 70)
    
    return network


if __name__ == "__main__":
    test_friendly_competition()
