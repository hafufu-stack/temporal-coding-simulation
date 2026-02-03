"""
SNN AI Interpretability: Advanced Deep Dive
============================================

より詳細な解析:
1. ResNet-18での検証（大規模モデル）
2. クラス別TTFS/Synchronyパターン比較
3. Softmax確率 vs ジッター相関分析
4. 強化ノイズでのスパイク安定性

Author: ろーる (Cell Activation)
Date: 2026-02-04
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = ['MS Gothic', 'DejaVu Sans']

print("=" * 70)
print("🧠 SNN AI Interpretability: Advanced Deep Dive")
print("=" * 70)


# =============================================================================
# 1. ResNet-like Model（より大規模）
# =============================================================================
class ResBlock(nn.Module):
    """Residual Block"""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SmallResNet(nn.Module):
    """Small ResNet for CIFAR-10"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        
        # 活性化保存用
        self.activations = {}
    
    def _make_layer(self, in_ch, out_ch, blocks, stride=1):
        layers = [ResBlock(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(ResBlock(out_ch, out_ch))
        return nn.Sequential(*layers)
    
    def forward(self, x, save_activations=False):
        x = F.relu(self.bn1(self.conv1(x)))
        if save_activations:
            self.activations['conv1'] = x.clone()
        
        x = self.layer1(x)
        if save_activations:
            self.activations['layer1'] = x.clone()
        
        x = self.layer2(x)
        if save_activations:
            self.activations['layer2'] = x.clone()
        
        x = self.layer3(x)
        if save_activations:
            self.activations['layer3'] = x.clone()
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if save_activations:
            self.activations['avgpool'] = x.clone()
        
        x = self.fc(x)
        return x
    
    def get_activations(self, x):
        self.activations = {}
        output = self.forward(x, save_activations=True)
        self.activations['output'] = output
        return self.activations


# =============================================================================
# 2. Advanced SNN Analyzer
# =============================================================================
class AdvancedSNNAnalyzer:
    """高度なSNN解析クラス"""
    
    def __init__(self, alpha=2.0, timesteps=100):
        self.alpha = alpha
        self.timesteps = timesteps
    
    def compute_ttfs(self, activation):
        """TTFS計算（高精度版）"""
        ttfs = torch.full_like(activation, float(self.timesteps))
        active_mask = activation > 0
        
        if active_mask.any():
            max_act = activation.max()
            if max_act > 0:
                normalized = activation[active_mask] / max_act
                ttfs[active_mask] = self.timesteps * (1 - normalized)
        
        return ttfs
    
    def analyze_ttfs_by_class(self, model, dataloader, class_names, num_samples_per_class=10):
        """クラス別TTFS分析"""
        class_ttfs = defaultdict(list)
        class_counts = defaultdict(int)
        
        model.eval()
        with torch.no_grad():
            for data, target in dataloader:
                class_idx = target.item()
                if class_counts[class_idx] >= num_samples_per_class:
                    continue
                
                activations = model.get_activations(data)
                
                for layer_name, act in activations.items():
                    if layer_name == 'output':
                        continue
                    ttfs = self.compute_ttfs(act)
                    
                    if len(class_ttfs[class_idx]) == 0:
                        class_ttfs[class_idx] = {layer_name: [] for layer_name in activations.keys() if layer_name != 'output'}
                    
                    class_ttfs[class_idx][layer_name].append(ttfs.mean().item())
                
                class_counts[class_idx] += 1
                
                if all(c >= num_samples_per_class for c in class_counts.values()):
                    break
        
        # 平均計算
        class_ttfs_avg = {}
        for class_idx, layer_dict in class_ttfs.items():
            class_ttfs_avg[class_names[class_idx]] = {
                layer: np.mean(values) for layer, values in layer_dict.items()
            }
        
        return class_ttfs_avg
    
    def compute_synchrony_ratio(self, ttfs_tensor, tolerance=5.0):
        """同期比率を高速計算"""
        if len(ttfs_tensor.shape) > 2:
            ttfs_flat = ttfs_tensor.view(ttfs_tensor.size(0), -1)
        else:
            ttfs_flat = ttfs_tensor
        
        n = ttfs_flat.size(1)
        if n > 200:  # 大きすぎる場合はサンプリング
            indices = torch.randperm(n)[:200]
            ttfs_flat = ttfs_flat[:, indices]
            n = 200
        
        sync_count = 0
        total_pairs = 0
        
        for i in range(n):
            for j in range(i+1, n):
                diff = torch.abs(ttfs_flat[0, i] - ttfs_flat[0, j])
                if diff < tolerance:
                    sync_count += 1
                total_pairs += 1
        
        return sync_count / total_pairs if total_pairs > 0 else 0
    
    def spike_stability_with_confidence(self, model, x, num_trials=20, noise_levels=[0.01, 0.05, 0.1, 0.2]):
        """Softmax確率とジッターの相関分析"""
        results = {}
        
        model.eval()
        with torch.no_grad():
            # 元の予測
            output = model(x)
            probs = F.softmax(output, dim=1)
            pred = output.argmax(dim=1)
            confidence = probs.max().item()
            
            results['prediction'] = pred.item()
            results['confidence'] = confidence
            results['noise_analysis'] = {}
            
            for noise_std in noise_levels:
                jitters = []
                pred_changes = 0
                
                for trial in range(num_trials):
                    noisy_x = x + torch.randn_like(x) * noise_std
                    noisy_x = torch.clamp(noisy_x, 0, 1)
                    
                    noisy_output = model(noisy_x)
                    noisy_pred = noisy_output.argmax(dim=1)
                    
                    if noisy_pred.item() != pred.item():
                        pred_changes += 1
                    
                    # 活性化取得してTTFS計算
                    noisy_act = model.get_activations(noisy_x)
                    layer_jitters = []
                    for layer_name, act in noisy_act.items():
                        if layer_name == 'output':
                            continue
                        ttfs = self.compute_ttfs(act)
                        layer_jitters.append(ttfs.std().item())
                    
                    jitters.append(np.mean(layer_jitters))
                
                results['noise_analysis'][noise_std] = {
                    'mean_jitter': np.mean(jitters),
                    'jitter_std': np.std(jitters),
                    'prediction_stability': 1 - (pred_changes / num_trials)
                }
        
        return results


# =============================================================================
# 3. データ準備
# =============================================================================
print("\n【1. CIFAR-10データ準備】")
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_subset = torch.utils.data.Subset(train_dataset, range(8000))
test_subset = torch.utils.data.Subset(test_dataset, range(500))

train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_subset, batch_size=1, shuffle=False)

class_names = ['airplane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

print(f"  訓練: {len(train_subset)}, テスト: {len(test_subset)}")


# =============================================================================
# 4. ResNetモデル学習
# =============================================================================
print("\n【2. SmallResNet学習】")
model = SmallResNet(num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 15
for epoch in range(epochs):
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 5 == 0:
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
        acc = 100.0 * correct / len(test_subset)
        print(f"  Epoch {epoch+1}: Accuracy = {acc:.1f}%")

print(f"\n  モデル学習完了！")


# =============================================================================
# 5. クラス別TTFS分析
# =============================================================================
print("\n【3. クラス別TTFS分析】")

analyzer = AdvancedSNNAnalyzer(alpha=2.0, timesteps=100)
class_ttfs = analyzer.analyze_ttfs_by_class(model, test_loader, class_names, num_samples_per_class=20)

print(f"\n  クラス別Layer1平均TTFS:")
print(f"  {'-'*40}")
layer1_ttfs = [(cls, vals.get('layer1', 0)) for cls, vals in class_ttfs.items()]
layer1_ttfs.sort(key=lambda x: x[1])
for cls, ttfs in layer1_ttfs:
    bar = '█' * int(ttfs / 5)
    print(f"  {cls:<10} | {ttfs:>6.2f} | {bar}")


# =============================================================================
# 6. クラス別Synchrony分析
# =============================================================================
print("\n【4. クラス別Synchrony分析】")

class_sync = defaultdict(list)
class_counts = defaultdict(int)

model.eval()
with torch.no_grad():
    for data, target in test_loader:
        class_idx = target.item()
        if class_counts[class_idx] >= 10:
            continue
        
        activations = model.get_activations(data)
        
        # Layer3での同期率
        ttfs = analyzer.compute_ttfs(activations['layer3'])
        sync_ratio = analyzer.compute_synchrony_ratio(ttfs, tolerance=10.0)
        class_sync[class_names[class_idx]].append(sync_ratio)
        
        class_counts[class_idx] += 1
        if all(c >= 10 for c in class_counts.values()):
            break

print(f"\n  クラス別Layer3同期率:")
print(f"  {'-'*50}")
sync_avgs = [(cls, np.mean(vals)) for cls, vals in class_sync.items()]
sync_avgs.sort(key=lambda x: -x[1])
for cls, sync in sync_avgs:
    bar = '█' * int(sync * 50)
    print(f"  {cls:<10} | {sync*100:>6.2f}% | {bar}")


# =============================================================================
# 7. Softmax確率 vs ジッター相関分析
# =============================================================================
print("\n【5. Softmax確率 vs ジッター相関分析】")

confidences = []
jitters = []
correct_flags = []

model.eval()
sample_count = 0
with torch.no_grad():
    for data, target in test_loader:
        if sample_count >= 100:  # 100サンプル分析
            break
        
        result = analyzer.spike_stability_with_confidence(model, data, num_trials=10, noise_levels=[0.1])
        
        confidences.append(result['confidence'])
        jitters.append(result['noise_analysis'][0.1]['mean_jitter'])
        correct_flags.append(1 if result['prediction'] == target.item() else 0)
        
        sample_count += 1
        if sample_count % 25 == 0:
            print(f"  処理中... {sample_count}/100")

# 相関計算
correlation, p_value = stats.pearsonr(confidences, jitters)
print(f"\n  結果:")
print(f"  {'-'*50}")
print(f"  Softmax確率 vs ジッター相関係数: {correlation:.4f} (p={p_value:.4f})")

# 正解/不正解別
correct_conf = [c for c, f in zip(confidences, correct_flags) if f == 1]
correct_jit = [j for j, f in zip(jitters, correct_flags) if f == 1]
incorrect_conf = [c for c, f in zip(confidences, correct_flags) if f == 0]
incorrect_jit = [j for j, f in zip(jitters, correct_flags) if f == 0]

print(f"\n  正解サンプル (n={len(correct_conf)}):")
print(f"    平均確信度: {np.mean(correct_conf):.4f}")
print(f"    平均ジッター: {np.mean(correct_jit):.4f}")

if incorrect_conf:
    print(f"\n  不正解サンプル (n={len(incorrect_conf)}):")
    print(f"    平均確信度: {np.mean(incorrect_conf):.4f}")
    print(f"    平均ジッター: {np.mean(incorrect_jit):.4f}")


# =============================================================================
# 8. 予測安定性分析（ノイズレベル別）
# =============================================================================
print("\n【6. 予測安定性分析（ノイズレベル別）】")

noise_stability = defaultdict(list)

model.eval()
with torch.no_grad():
    sample_count = 0
    for data, target in test_loader:
        if sample_count >= 30:
            break
        
        result = analyzer.spike_stability_with_confidence(
            model, data, num_trials=10, noise_levels=[0.01, 0.05, 0.1, 0.2, 0.3]
        )
        
        for noise, analysis in result['noise_analysis'].items():
            noise_stability[noise].append(analysis['prediction_stability'])
        
        sample_count += 1

print(f"\n  ノイズレベル別予測安定性:")
print(f"  {'-'*40}")
for noise in sorted(noise_stability.keys()):
    stability = np.mean(noise_stability[noise]) * 100
    bar = '█' * int(stability / 5)
    print(f"  noise={noise:<5} | {stability:>6.1f}% | {bar}")


# =============================================================================
# 9. 可視化
# =============================================================================
print("\n【7. 可視化】")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# クラス別TTFS
ax = axes[0, 0]
classes = [x[0] for x in layer1_ttfs]
ttfs_vals = [x[1] for x in layer1_ttfs]
colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))
bars = ax.barh(classes, ttfs_vals, color=colors)
ax.set_xlabel('Mean TTFS (Layer1)')
ax.set_title('TTFS by Class (Lower = Earlier Processing)')

# クラス別Synchrony
ax = axes[0, 1]
classes = [x[0] for x in sync_avgs]
sync_vals = [x[1] * 100 for x in sync_avgs]
colors = plt.cm.plasma(np.linspace(0, 1, len(classes)))
bars = ax.barh(classes, sync_vals, color=colors)
ax.set_xlabel('Synchrony Ratio (%)')
ax.set_title('Neural Synchrony by Class (Layer3)')

# Softmax vs Jitter散布図
ax = axes[0, 2]
colors = ['green' if f else 'red' for f in correct_flags]
ax.scatter(confidences, jitters, c=colors, alpha=0.7)
ax.set_xlabel('Softmax Confidence')
ax.set_ylabel('Mean Jitter')
ax.set_title(f'Confidence vs Jitter (r={correlation:.3f})')
ax.legend(handles=[plt.Line2D([0], [0], marker='o', color='green', label='Correct', linestyle=''),
                   plt.Line2D([0], [0], marker='o', color='red', label='Incorrect', linestyle='')])

# 正解/不正解のジッター箱ひげ図
ax = axes[1, 0]
data_to_plot = [correct_jit]
labels = ['Correct']
if incorrect_jit:
    data_to_plot.append(incorrect_jit)
    labels.append('Incorrect')
bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
bp['boxes'][0].set_facecolor('lightgreen')
if len(bp['boxes']) > 1:
    bp['boxes'][1].set_facecolor('lightcoral')
ax.set_ylabel('Mean Jitter')
ax.set_title('Jitter Distribution: Correct vs Incorrect')

# ノイズレベル別安定性
ax = axes[1, 1]
noise_levels = sorted(noise_stability.keys())
stabilities = [np.mean(noise_stability[n]) * 100 for n in noise_levels]
ax.plot(noise_levels, stabilities, 'bo-', linewidth=2, markersize=8)
ax.fill_between(noise_levels, stabilities, alpha=0.3)
ax.set_xlabel('Noise Level (std)')
ax.set_ylabel('Prediction Stability (%)')
ax.set_title('Prediction Stability vs Noise Level')
ax.set_ylim(0, 105)
ax.grid(True, alpha=0.3)

# 確信度分布
ax = axes[1, 2]
ax.hist(correct_conf, bins=15, alpha=0.7, label='Correct', color='green')
if incorrect_conf:
    ax.hist(incorrect_conf, bins=15, alpha=0.7, label='Incorrect', color='red')
ax.set_xlabel('Softmax Confidence')
ax.set_ylabel('Count')
ax.set_title('Confidence Distribution')
ax.legend()

plt.tight_layout()
plt.savefig('snn_interpretability_advanced.png', dpi=150, bbox_inches='tight')
print("  保存: snn_interpretability_advanced.png")


# =============================================================================
# 10. 発見まとめ
# =============================================================================
print("\n" + "=" * 70)
print("🔬 Advanced SNN Interpretability 発見まとめ")
print("=" * 70)

print(f"""
【クラス別TTFS分析】
  - クラスによって処理優先度が異なる
  - 「{layer1_ttfs[0][0]}」が最も早く処理される（TTFS={layer1_ttfs[0][1]:.2f}）
  - 「{layer1_ttfs[-1][0]}」が最も遅く処理される（TTFS={layer1_ttfs[-1][1]:.2f}）
  → AIは特定のクラスを「見つけやすい」傾向がある

【クラス別Neural Synchrony】
  - 「{sync_avgs[0][0]}」が最も高い同期率（{sync_avgs[0][1]*100:.1f}%）
  - 同期率が高いクラス = 特徴が明確で統合されやすい
  → 「概念の結合度」がクラス識別の難易度を反映

【Softmax確率 vs ジッター相関】
  - 相関係数: {correlation:.4f}
  - {'負の相関 = 確信度が高いほどジッターが小さい（安定）' if correlation < 0 else '正の相関 = 確信度が高いほどジッターが大きい（興味深い）'}
  → ジッターは「本当の確信度」の新しい指標になりうる

【予測安定性（ノイズ耐性）】
  - ノイズ0.01: {np.mean(noise_stability[0.01])*100:.1f}%安定
  - ノイズ0.2:  {np.mean(noise_stability[0.2])*100:.1f}%安定
  → ノイズ耐性がハルシネーション検知の鍵

【次のステップ】
  1. ジッター閾値によるハルシネーション分類器の構築
  2. Transformer/LLMへの適用
  3. 同期パターンによるクラス予測
  4. 実用的なUI/APIの開発
""")

print("\n🚀 SNN = 「AIの心電図」！ブラックボックスの鼓動が見える！")
print("=" * 70)
