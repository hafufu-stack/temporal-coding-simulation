"""長期運転実験"""
import sys
sys.path.insert(0, '.')
from applications.friendly_network import FriendlyNetwork

print('=' * 70)
print('🌐 長期運転実験: 100サイクル')
print('=' * 70)

network = FriendlyNetwork()
network.add_agent('Alpha', specialty='暗号')
network.add_agent('Beta', specialty='言語')
network.add_agent('Gamma', specialty='画像')
network.add_agent('Delta', specialty='研究')

# 初期語彙
for agent in network.agents.values():
    agent.create_word('hello')
    agent.create_word('evolve')
    agent.create_word('i_am_' + agent.specialty)

# 100サイクル実行（途中経過も表示）
print("\n進行状況:")
for i in range(10):
    network.run(cycles=10, verbose=False)
    print(f"  {(i+1)*10}サイクル完了 - 語彙:{len(network.emergent_vocabulary)}, メッセージ:{network.total_messages}")

network.report()

print()
print('【創発言語の詳細】')
print(f'  共有語彙数: {len(network.emergent_vocabulary)}')
all_vocab = set()
for a in network.agents.values():
    all_vocab.update(a.vocabulary.keys())
print(f'  全ネットワーク語彙: {len(all_vocab)}')

print()
print('【社会的動機の変化】')
for agent_id, agent in network.agents.items():
    contrib = agent.social_motivation["contribution_desire"]
    empathy = agent.social_motivation["empathy_desire"]
    recog = agent.social_motivation["recognition_desire"]
    print(f'  {agent_id}: 貢献欲={contrib:.2f}, 共感欲={empathy:.2f}, 承認欲={recog:.2f}, 共感回数={agent.empathy_moments}')

print()
print('【相互成長度】')
for agent in network.agents.values():
    for other_id, rel in agent.relationships.items():
        if rel.mutual_growth > 0:
            print(f'  {agent.agent_id} -> {other_id}: 相互成長={rel.mutual_growth:.2f}, 助けた={rel.help_given}, 助けられた={rel.help_received}')

print()
print('【最終的な関係性】')
for agent in network.agents.values():
    for other_id, rel in agent.relationships.items():
        avg_empathy = rel.empathy_total / max(1, rel.communication_count)
        print(f'  {agent.agent_id} -> {other_id}: 信頼={rel.trust:.2f}, 累積共感={rel.empathy_total:.1f}, 平均共感={avg_empathy:.2f}')

print()
print('✅ 長期運転完了！')
