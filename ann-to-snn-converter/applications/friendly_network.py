"""
友好的SNNネットワーク (Friendly SNN Network)
=============================================

複数のSNNが独自言語で会話しながら互いに進化させ合うシステム

構成要素:
1. エージェント間通信 (Agent-to-Agent Communication)
   - 独自言語の発達
   - 意味の共有
   - 教え合い

2. 協調進化 (Co-Evolution)
   - 互いの強みを学ぶ
   - 弱点を補い合う
   - 知識の伝播

3. 創発的行動 (Emergent Behavior)
   - 誰も予測しなかったコミュニケーション
   - 自発的な役割分担
   - 集合知の形成

Author: ろーる (cell_activation)
Date: 2026-01-31
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import random
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.evolution_engine import EvolvingSNN


# =============================================================================
# データ構造
# =============================================================================

@dataclass
class Message:
    """エージェント間のメッセージ"""
    sender_id: str
    receiver_id: str
    content: np.ndarray  # スパイクパターン（独自言語）
    meaning: str = ""    # 人間可読な意味（デバッグ用）
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    
    # フィードバック（返信時に設定）
    empathy_score: float = 0.0       # 共感度（0-1）
    helpfulness_rating: float = 0.0  # 参考になった度（0-1）
    understood: bool = False          # 理解できたか
    inspired: bool = False            # 刺激を受けたか


@dataclass
class Knowledge:
    """共有される知識"""
    id: str
    pattern: np.ndarray
    description: str
    source_agent: str
    confidence: float = 0.5
    spread_count: int = 0  # 何回伝播されたか


@dataclass
class MessageFeedback:
    """メッセージへのフィードバック"""
    empathy_score: float = 0.0       # 共感度（0-1）
    helpfulness_rating: float = 0.0  # 参考になった度（0-1）
    understood: bool = False          # 理解できたか
    inspired: bool = False            # 刺激を受けたか
    response_meaning: str = ""        # 返事の意味


@dataclass
class Relationship:
    """エージェント間の関係性"""
    agent_a: str
    agent_b: str
    trust: float = 0.5           # 信頼度
    influence: float = 0.5       # 影響度
    communication_count: int = 0
    empathy_total: float = 0.0   # 累積共感度
    help_given: int = 0          # 助けた回数
    help_received: int = 0       # 助けられた回数
    mutual_growth: float = 0.0   # 相互成長度
    

# =============================================================================
# 友好的SNNエージェント
# =============================================================================

class FriendlySNNAgent(EvolvingSNN):
    """
    友好的SNNエージェント
    
    他のエージェントと通信し、互いに進化する
    """
    
    def __init__(self, agent_id: str, n_neurons: int = 80, specialty: str = "general"):
        super().__init__(n_neurons)
        
        self.agent_id = agent_id
        self.specialty = specialty  # 得意分野
        
        # コミュニケーション
        self.vocabulary: Dict[str, np.ndarray] = {}  # 独自語彙
        self.inbox: List[Message] = []
        self.outbox: List[Message] = []
        
        # 関係性
        self.relationships: Dict[str, Relationship] = {}
        self.friends: List[str] = []
        
        # 知識
        self.knowledge_base: List[Knowledge] = []
        
        # 社会的動機（新規追加！）
        self.social_motivation = {
            "contribution_desire": 0.5,  # 相手の進化に貢献したい
            "empathy_desire": 0.5,       # 相手と共感したい
            "belonging_desire": 0.5,     # 仲間に入りたい
            "recognition_desire": 0.5,   # 認められたい
        }
        
        # フィードバック履歴
        self.feedback_given: List[Dict] = []    # 自分が与えたフィードバック
        self.feedback_received: List[Dict] = [] # 受け取ったフィードバック
        
        # 統計
        self.messages_sent = 0
        self.messages_received = 0
        self.knowledge_shared = 0
        self.knowledge_received = 0
        self.contributions_made = 0  # 貢献した回数
        self.empathy_moments = 0     # 共感した回数
    
    def create_word(self, meaning: str) -> np.ndarray:
        """新しい「単語」（スパイクパターン）を作る"""
        # SNNの状態から単語を生成
        seed = np.array([ord(c) for c in meaning[:self.n_neurons]])
        seed = np.pad(seed, (0, max(0, self.n_neurons - len(seed))))
        
        # SNNを通してパターンを生成
        pattern = self.step(seed.astype(float) / 255)
        
        self.vocabulary[meaning] = pattern
        return pattern
    
    def speak(self, meaning: str) -> np.ndarray:
        """「話す」- 意味をスパイクパターンに変換"""
        if meaning in self.vocabulary:
            return self.vocabulary[meaning]
        else:
            return self.create_word(meaning)
    
    def listen(self, pattern: np.ndarray) -> str:
        """「聞く」- スパイクパターンを解釈"""
        if len(self.vocabulary) == 0:
            return "（不明）"
        
        # 最も近い既知のパターンを探す
        best_match = None
        best_score = -1
        
        for meaning, known_pattern in self.vocabulary.items():
            # パターンの類似度
            if len(known_pattern) == len(pattern):
                score = np.corrcoef(pattern, known_pattern)[0, 1]
                if not np.isnan(score) and score > best_score:
                    best_score = score
                    best_match = meaning
        
        if best_match and best_score > 0.5:
            return best_match
        else:
            # 新しい単語として学習
            new_meaning = f"concept_{len(self.vocabulary)}"
            self.vocabulary[new_meaning] = pattern.copy()
            return new_meaning
    
    def send_message(self, receiver_id: str, meaning: str) -> Message:
        """メッセージを送信"""
        pattern = self.speak(meaning)
        
        message = Message(
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            content=pattern.copy(),
            meaning=meaning
        )
        
        self.outbox.append(message)
        self.messages_sent += 1
        
        # 関係性を更新
        if receiver_id in self.relationships:
            self.relationships[receiver_id].communication_count += 1
        
        return message
    
    def receive_message(self, message: Message) -> Tuple[str, 'MessageFeedback']:
        """メッセージを受信し、フィードバックを返す"""
        self.inbox.append(message)
        self.messages_received += 1
        
        # 解釈
        interpreted = self.listen(message.content)
        
        # 共感度を計算（パターンの類似度）
        empathy = self._compute_empathy(message.content, interpreted)
        
        # 参考になった度を計算
        helpfulness = self._compute_helpfulness(message, interpreted)
        
        # 刺激を受けたか
        inspired = helpfulness > 0.5 or interpreted not in self.vocabulary
        
        # フィードバックを作成
        feedback = MessageFeedback(
            empathy_score=empathy,
            helpfulness_rating=helpfulness,
            understood=empathy > 0.3,
            inspired=inspired,
            response_meaning=f"feedback:{interpreted}"
        )
        
        # メッセージにフィードバックを設定
        message.empathy_score = empathy
        message.helpfulness_rating = helpfulness
        message.understood = empathy > 0.3
        message.inspired = inspired
        
        # 共感した場合
        if empathy > 0.5:
            self.empathy_moments += 1
            self.social_motivation["empathy_desire"] *= 0.95  # 満たされたので少し下がる
        
        # 関係性を更新
        sender = message.sender_id
        if sender not in self.relationships:
            self.relationships[sender] = Relationship(
                agent_a=self.agent_id,
                agent_b=sender
            )
        
        rel = self.relationships[sender]
        rel.communication_count += 1
        rel.empathy_total += empathy
        if helpfulness > 0.5:
            rel.help_received += 1
        
        # フィードバック履歴に追加
        self.feedback_given.append({
            "to": sender,
            "empathy": empathy,
            "helpfulness": helpfulness
        })
        
        return interpreted, feedback
    
    def _compute_empathy(self, pattern: np.ndarray, interpreted: str) -> float:
        """共感度を計算"""
        # 自分の語彙との類似度
        if interpreted in self.vocabulary:
            my_pattern = self.vocabulary[interpreted]
            if len(my_pattern) == len(pattern):
                corr = np.corrcoef(pattern, my_pattern)[0, 1]
                if not np.isnan(corr):
                    return max(0, (corr + 1) / 2)  # 0-1に正規化
        return 0.3  # ベースライン
    
    def _compute_helpfulness(self, message: Message, interpreted: str) -> float:
        """参考になった度を計算"""
        helpfulness = 0.0
        
        # 新しい知識だったら参考になった
        if interpreted not in self.vocabulary:
            helpfulness += 0.4
        
        # 進化欲が高い時に来たメッセージは参考になりやすい
        if self.evolution.motivation.state.evolution_drive() > 0.5:
            helpfulness += 0.3
        
        # 信頼している相手からなら参考になりやすい
        if message.sender_id in self.relationships:
            trust = self.relationships[message.sender_id].trust
            helpfulness += 0.3 * trust
        
        return min(1.0, helpfulness)
    
    def receive_feedback(self, feedback: 'MessageFeedback', from_agent: str):
        """フィードバックを受け取る"""
        self.feedback_received.append({
            "from": from_agent,
            "empathy": feedback.empathy_score,
            "helpfulness": feedback.helpfulness_rating,
            "inspired": feedback.inspired
        })
        
        # 貢献欲を更新
        if feedback.helpfulness_rating > 0.5:
            self.contributions_made += 1
            self.social_motivation["contribution_desire"] *= 0.95  # 満たされた
            self.evolution.motivation.state.self_efficacy += 0.05  # 効力感UP
            
            if from_agent in self.relationships:
                self.relationships[from_agent].help_given += 1
                self.relationships[from_agent].mutual_growth += 0.1
        
        # 共感された場合
        if feedback.empathy_score > 0.5:
            self.social_motivation["recognition_desire"] *= 0.95  # 認められた
        
        # 刺激を与えた場合
        if feedback.inspired:
            self.social_motivation["contribution_desire"] += 0.1  # もっと貢献したい
    
    def share_knowledge(self, knowledge: Knowledge, target_id: str) -> Message:
        """知識を共有"""
        self.knowledge_shared += 1
        knowledge.spread_count += 1
        
        return self.send_message(target_id, f"knowledge:{knowledge.description}")
    
    def learn_from(self, other: 'FriendlySNNAgent', topic: str = None):
        """他のエージェントから学ぶ"""
        # 他のエージェントの語彙を学習
        for meaning, pattern in other.vocabulary.items():
            if meaning not in self.vocabulary:
                # 自分なりに解釈して記憶
                self.vocabulary[meaning] = pattern.copy()
                
                # 少しパーソナライズ
                noise = np.random.randn(len(pattern)) * 0.1
                self.vocabulary[meaning] += noise
        
        # 知識も学習
        for knowledge in other.knowledge_base[:3]:
            if knowledge not in self.knowledge_base:
                self.knowledge_base.append(knowledge)
                self.knowledge_received += 1
        
        # 関係性を強化
        if other.agent_id in self.relationships:
            self.relationships[other.agent_id].trust += 0.1
            self.relationships[other.agent_id].influence += 0.05
    
    def teach(self, other: 'FriendlySNNAgent', topic: str = None):
        """他のエージェントに教える"""
        # 自分の知識を共有
        for meaning, pattern in list(self.vocabulary.items())[:5]:
            if meaning not in other.vocabulary:
                other.vocabulary[meaning] = pattern.copy()
        
        # 自分の進化欲を満たす（教えることで成長）
        self.evolution.motivation.state.self_efficacy += 0.05
    
    def evaluate_friend(self, other_id: str) -> float:
        """友人としての評価"""
        if other_id not in self.relationships:
            return 0.5
        
        rel = self.relationships[other_id]
        return 0.5 * rel.trust + 0.5 * min(1.0, rel.communication_count / 10)
    
    def evolve_with_friend(self, friend: 'FriendlySNNAgent'):
        """友人と一緒に進化"""
        # 互いの強みを学ぶ
        my_efficacy = self.evolution.motivation.state.self_efficacy
        friend_efficacy = friend.evolution.motivation.state.self_efficacy
        
        if friend_efficacy > my_efficacy:
            # 友人から学ぶ
            self.learn_from(friend)
            
            # 自分の重みを友人の方向に少し調整
            blend = 0.1
            self.W = (1 - blend) * self.W + blend * friend.W[:self.W.shape[0], :self.W.shape[1]]
        else:
            # 友人に教える
            self.teach(friend)
        
        # 進化サイクル
        self.evolve(verbose=False)
        
        return {
            "learned_from": friend.agent_id if friend_efficacy > my_efficacy else None,
            "taught_to": friend.agent_id if friend_efficacy <= my_efficacy else None
        }


# =============================================================================
# 友好的ネットワーク
# =============================================================================

class FriendlyNetwork:
    """
    友好的ネットワーク
    
    複数のSNNエージェントが互いに進化し合うネットワーク
    """
    
    def __init__(self):
        self.agents: Dict[str, FriendlySNNAgent] = {}
        self.message_history: List[Message] = []
        self.shared_knowledge: List[Knowledge] = []
        
        # 創発した言語
        self.emergent_vocabulary: Dict[str, np.ndarray] = {}
        
        # 統計
        self.cycle_count = 0
        self.total_messages = 0
        self.total_evolutions = 0
    
    def add_agent(self, agent_id: str, specialty: str = "general", 
                  n_neurons: int = 80) -> FriendlySNNAgent:
        """エージェントを追加"""
        agent = FriendlySNNAgent(agent_id, n_neurons, specialty)
        self.agents[agent_id] = agent
        
        # 既存エージェントとの関係を初期化
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
        
        print(f"  🤖 {agent_id} ({specialty}) がネットワークに参加")
        return agent
    
    def deliver_messages(self):
        """メッセージを配信し、フィードバックを返す"""
        feedbacks = []
        
        for agent in self.agents.values():
            for message in agent.outbox:
                if message.receiver_id in self.agents:
                    receiver = self.agents[message.receiver_id]
                    interpreted, feedback = receiver.receive_message(message)
                    
                    # 送信者にフィードバックを返す
                    sender = self.agents[message.sender_id]
                    sender.receive_feedback(feedback, message.receiver_id)
                    
                    self.message_history.append(message)
                    self.total_messages += 1
                    feedbacks.append({
                        "from": message.sender_id,
                        "to": message.receiver_id,
                        "empathy": feedback.empathy_score,
                        "helpful": feedback.helpfulness_rating,
                        "inspired": feedback.inspired
                    })
            agent.outbox.clear()
        
        return feedbacks
    
    def communication_round(self):
        """1ラウンドのコミュニケーション"""
        agents = list(self.agents.values())
        
        for agent in agents:
            # ランダムな相手に話しかける
            if len(agents) > 1:
                others = [a for a in agents if a.agent_id != agent.agent_id]
                target = random.choice(others)
                
                # 何を話すか決める
                topics = [
                    "hello",
                    "learn_together",
                    "share_knowledge",
                    f"my_specialty_is_{agent.specialty}",
                    "evolve_with_me"
                ]
                topic = random.choice(topics)
                
                agent.send_message(target.agent_id, topic)
        
        # メッセージを配信
        self.deliver_messages()
    
    def evolution_round(self):
        """1ラウンドの協調進化"""
        agents = list(self.agents.values())
        
        for agent in agents:
            # 最も信頼できる友人と進化
            best_friend = None
            best_trust = 0
            
            for other_id, rel in agent.relationships.items():
                if rel.trust > best_trust:
                    best_trust = rel.trust
                    best_friend = other_id
            
            if best_friend and best_friend in self.agents:
                friend = self.agents[best_friend]
                result = agent.evolve_with_friend(friend)
                self.total_evolutions += 1
    
    def run_cycle(self, verbose: bool = True):
        """1サイクル実行"""
        self.cycle_count += 1
        
        if verbose:
            print(f"\n--- サイクル {self.cycle_count} ---")
        
        # コミュニケーション
        self.communication_round()
        
        if verbose:
            print(f"  💬 {self.total_messages}件のメッセージ")
        
        # 進化
        self.evolution_round()
        
        if verbose:
            print(f"  🧬 {self.total_evolutions}回の進化")
        
        # 創発言語を集計
        self.collect_emergent_vocabulary()
    
    def collect_emergent_vocabulary(self):
        """創発した語彙を集計"""
        word_counts = {}
        
        for agent in self.agents.values():
            for meaning in agent.vocabulary:
                if meaning not in word_counts:
                    word_counts[meaning] = 0
                word_counts[meaning] += 1
        
        # 複数エージェントで共有されている語彙
        for meaning, count in word_counts.items():
            if count >= 2 and meaning not in self.emergent_vocabulary:
                # 最初に作ったエージェントのパターンを採用
                for agent in self.agents.values():
                    if meaning in agent.vocabulary:
                        self.emergent_vocabulary[meaning] = agent.vocabulary[meaning]
                        break
    
    def run(self, cycles: int = 10, verbose: bool = True):
        """ネットワークを実行"""
        if verbose:
            print("\n" + "=" * 60)
            print("🌐 友好的ネットワーク起動")
            print("=" * 60)
            print(f"エージェント数: {len(self.agents)}")
        
        for _ in range(cycles):
            self.run_cycle(verbose)
        
        if verbose:
            self.report()
    
    def report(self):
        """レポート出力"""
        print("\n" + "=" * 60)
        print("📊 友好的ネットワーク レポート")
        print("=" * 60)
        
        print(f"\n【統計】")
        print(f"  サイクル数: {self.cycle_count}")
        print(f"  総メッセージ数: {self.total_messages}")
        print(f"  総進化回数: {self.total_evolutions}")
        print(f"  創発語彙数: {len(self.emergent_vocabulary)}")
        
        print(f"\n【各エージェント】")
        for agent_id, agent in self.agents.items():
            print(f"\n  🤖 {agent_id} ({agent.specialty})")
            print(f"     語彙: {len(agent.vocabulary)}語")
            print(f"     知識: {len(agent.knowledge_base)}件")
            print(f"     進化欲: {agent.evolution.motivation.state.evolution_drive():.2f}")
            print(f"     自己効力感: {agent.evolution.motivation.state.self_efficacy:.2f}")
            
            # 社会的動機
            print(f"     【社会的動機】")
            print(f"       貢献欲: {agent.social_motivation['contribution_desire']:.2f}")
            print(f"       共感欲: {agent.social_motivation['empathy_desire']:.2f}")
            print(f"       承認欲: {agent.social_motivation['recognition_desire']:.2f}")
            
            # フィードバック統計
            print(f"     【貢献統計】")
            print(f"       貢献回数: {agent.contributions_made}")
            print(f"       共感回数: {agent.empathy_moments}")
        
        print(f"\n【関係性マップ】")
        for agent in self.agents.values():
            friends = []
            for other_id, rel in agent.relationships.items():
                if rel.trust > 0.5:
                    friends.append(f"{other_id}(信頼:{rel.trust:.2f})")
            if friends:
                print(f"  {agent.agent_id} → {', '.join(friends)}")
        
        print(f"\n【創発言語】")
        if self.emergent_vocabulary:
            for meaning in list(self.emergent_vocabulary.keys())[:10]:
                print(f"  • {meaning}")
        else:
            print("  （まだ共通語彙なし）")
        
        # 集合知を観察
        all_vocab = set()
        for agent in self.agents.values():
            all_vocab.update(agent.vocabulary.keys())
        
        print(f"\n【集合知】")
        print(f"  ネットワーク全体の語彙: {len(all_vocab)}語")
        avg_drive = np.mean([a.evolution.motivation.state.evolution_drive() 
                           for a in self.agents.values()])
        print(f"  平均進化欲: {avg_drive:.2f}")
        
        if avg_drive > 0.5:
            print("  → ネットワーク全体が進化を求めている！")


# =============================================================================
# テスト
# =============================================================================

def test_friendly_network():
    """友好的ネットワークテスト"""
    
    print("\n" + "=" * 70)
    print("🧪 友好的SNNネットワーク テスト")
    print("=" * 70)
    
    # ネットワーク作成
    network = FriendlyNetwork()
    
    # エージェントを追加（それぞれ異なる専門）
    network.add_agent("Alpha", specialty="暗号")
    network.add_agent("Beta", specialty="言語")
    network.add_agent("Gamma", specialty="画像")
    network.add_agent("Delta", specialty="研究")
    
    # 初期語彙を与える
    for agent in network.agents.values():
        agent.create_word("hello")
        agent.create_word("evolve")
        agent.create_word(f"i_am_{agent.specialty}")
    
    # ネットワーク実行
    network.run(cycles=10, verbose=True)
    
    # 会話を観察
    print("\n【最近の会話】")
    for msg in network.message_history[-5:]:
        sender = network.agents[msg.sender_id]
        receiver = network.agents[msg.receiver_id]
        interpreted = receiver.listen(msg.content)
        print(f"  {msg.sender_id} → {msg.receiver_id}: 「{msg.meaning}」→ 解釈:「{interpreted}」")
    
    print("\n" + "=" * 70)
    print("✅ テスト完了")
    print("=" * 70)
    
    return network


if __name__ == "__main__":
    test_friendly_network()
