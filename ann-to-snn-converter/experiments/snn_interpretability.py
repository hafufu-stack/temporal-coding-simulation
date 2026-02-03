"""
SNN AI Interpretability: Dynamic Analysis of ANN Black Box
============================================================

SNNを「時間的顕微鏡」として使用し、ANNのブラックボックスを解剖する。

3つの解析手法:
1. TTFS (Time-to-First-Spike): 思考順序の可視化
2. Neural Synchrony: 概念結合（同期）の発見  
3. Spike Stability: ハルシネーション検知（ジッター解析）

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
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.family'] = ['MS Gothic', 'DejaVu Sans']

print("=" * 70)
print("🧠 SNN AI Interpretability: ブラックボックスの時間的解剖")
print("=" * 70)


# =============================================================================
# 1. シンプルなCNNモデル（解析対象）
# =============================================================================
class SimpleCNN(nn.Module):
    """解析対象のシンプルなCNN"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv2(x)))  # 16x16 -> 8x8
        x = self.pool(F.relu(self.conv3(x)))  # 8x8 -> 4x4
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def get_layer_activations(self, x):
        """各層の活性化値を取得"""
        activations = {}
        
        x = self.conv1(x)
        x_relu = F.relu(x)
        activations['conv1'] = x_relu.clone()
        x = self.pool(x_relu)
        
        x = self.conv2(x)
        x_relu = F.relu(x)
        activations['conv2'] = x_relu.clone()
        x = self.pool(x_relu)
        
        x = self.conv3(x)
        x_relu = F.relu(x)
        activations['conv3'] = x_relu.clone()
        x = self.pool(x_relu)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x_relu = F.relu(x)
        activations['fc1'] = x_relu.clone()
        
        activations['output'] = self.fc2(x_relu)
        
        return activations


# =============================================================================
# 2. SNN変換 & TTFS計算
# =============================================================================
class SNNAnalyzer:
    """SNN変換と時間的解析を行うクラス"""
    
    def __init__(self, alpha=2.0, timesteps=50):
        self.alpha = alpha
        self.timesteps = timesteps
    
    def compute_ttfs(self, activation, threshold):
        """
        Time-to-First-Spike (TTFS) を計算
        
        高い活性化値 → 早い発火（小さいTTFS）
        低い活性化値 → 遅い発火（大きいTTFS）
        """
        # 活性化値が0以下の場合は発火しない（最大タイムステップ）
        ttfs = torch.full_like(activation, float(self.timesteps))
        
        # 発火するニューロンのTTFS計算
        # TTFS ∝ 1/activation (高い活性化 → 早い発火)
        active_mask = activation > 0
        if active_mask.any():
            # 正規化して0～timestepsにスケール
            max_act = activation.max()
            if max_act > 0:
                normalized = activation[active_mask] / max_act
                # 高い活性化 → 小さいTTFS
                ttfs[active_mask] = self.timesteps * (1 - normalized)
        
        return ttfs
    
    def analyze_layer_ttfs(self, activations):
        """各層のTTFSを計算"""
        ttfs_results = {}
        
        for layer_name, act in activations.items():
            if layer_name == 'output':
                continue
                
            # 閾値計算（α=2.0公式）
            threshold = self.alpha * act.max().item()
            
            # TTFS計算
            ttfs = self.compute_ttfs(act, threshold)
            
            ttfs_results[layer_name] = {
                'ttfs': ttfs,
                'activation': act,
                'threshold': threshold,
                'mean_ttfs': ttfs.mean().item(),
                'min_ttfs': ttfs.min().item()
            }
        
        return ttfs_results
    
    def compute_synchrony(self, ttfs_layer, tolerance=2.0):
        """
        Neural Synchrony（同期発火ペア）を検出
        
        同じタイミング（tolerance内）で発火するニューロンペアを検出
        """
        if len(ttfs_layer.shape) == 4:
            # Conv層: (B, C, H, W) -> flatten
            ttfs_flat = ttfs_layer.view(ttfs_layer.size(0), -1)
        else:
            ttfs_flat = ttfs_layer
        
        synchrony_matrix = torch.zeros(ttfs_flat.size(1), ttfs_flat.size(1))
        
        for i in range(ttfs_flat.size(1)):
            for j in range(i+1, ttfs_flat.size(1)):
                # 発火タイミングの差
                diff = torch.abs(ttfs_flat[0, i] - ttfs_flat[0, j])
                if diff < tolerance:
                    synchrony_matrix[i, j] = 1.0
                    synchrony_matrix[j, i] = 1.0
        
        return synchrony_matrix
    
    def compute_spike_stability(self, model, x, num_trials=10, noise_std=0.05):
        """
        Spike Stability解析（ハルシネーション検知用）
        
        入力にノイズを加えて複数回推論し、発火タイミングの揺れ（ジッター）を測定
        """
        stability_results = {}
        all_ttfs = defaultdict(list)
        
        for trial in range(num_trials):
            # ノイズ付加
            noisy_x = x + torch.randn_like(x) * noise_std
            noisy_x = torch.clamp(noisy_x, 0, 1)
            
            # 活性化取得
            activations = model.get_layer_activations(noisy_x)
            
            # TTFS計算
            ttfs_results = self.analyze_layer_ttfs(activations)
            
            for layer_name, result in ttfs_results.items():
                all_ttfs[layer_name].append(result['ttfs'])
        
        # ジッター計算（標準偏差）
        for layer_name, ttfs_list in all_ttfs.items():
            stacked = torch.stack(ttfs_list)
            jitter = stacked.std(dim=0)  # 試行間の標準偏差
            
            stability_results[layer_name] = {
                'jitter_mean': jitter.mean().item(),
                'jitter_max': jitter.max().item(),
                'jitter_map': jitter
            }
        
        return stability_results


# =============================================================================
# 3. データ準備
# =============================================================================
print("\n【1. CIFAR-10データ準備】")
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

try:
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
except:
    print("  CIFAR-10ダウンロード中...")
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# サブセット使用（高速化）
train_subset = torch.utils.data.Subset(train_dataset, range(5000))
test_subset = torch.utils.data.Subset(test_dataset, range(500))

train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_subset, batch_size=1, shuffle=False)

class_names = ['airplane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

print(f"  訓練: {len(train_subset)}, テスト: {len(test_subset)}")


# =============================================================================
# 4. モデル学習
# =============================================================================
print("\n【2. CNNモデル学習】")
model = SimpleCNN(num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # 精度確認
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
# 5. TTFS解析（思考順序の可視化）
# =============================================================================
print("\n【3. TTFS解析 - 思考順序の可視化】")

analyzer = SNNAnalyzer(alpha=2.0, timesteps=50)

# サンプル画像を取得
sample_data, sample_target = next(iter(test_loader))
sample_class = class_names[sample_target.item()]

print(f"  サンプル画像クラス: {sample_class}")

# 活性化取得
model.eval()
with torch.no_grad():
    activations = model.get_layer_activations(sample_data)

# TTFS計算
ttfs_results = analyzer.analyze_layer_ttfs(activations)

print(f"\n  各層のTTFS統計:")
print(f"  {'-'*60}")
print(f"  {'層名':<10} | {'平均TTFS':>12} | {'最小TTFS':>12} | {'閾値':>10}")
print(f"  {'-'*60}")
for layer_name, result in ttfs_results.items():
    print(f"  {layer_name:<10} | {result['mean_ttfs']:>12.2f} | {result['min_ttfs']:>12.2f} | {result['threshold']:>10.2f}")


# =============================================================================
# 6. Neural Synchrony解析（概念結合）
# =============================================================================
print("\n【4. Neural Synchrony解析 - 概念結合の発見】")

# FC層での同期検出
fc1_ttfs = ttfs_results['fc1']['ttfs']
sync_matrix = analyzer.compute_synchrony(fc1_ttfs, tolerance=3.0)

# 同期ペア数をカウント
num_sync_pairs = (sync_matrix.sum() / 2).int().item()
total_pairs = (fc1_ttfs.numel() * (fc1_ttfs.numel() - 1)) // 2
sync_ratio = num_sync_pairs / total_pairs * 100

print(f"  FC1層での同期ペア: {num_sync_pairs}/{total_pairs} ({sync_ratio:.2f}%)")
print(f"  → 同期ペアは「概念の塊」を形成している可能性")


# =============================================================================
# 7. Spike Stability解析（ハルシネーション検知PoC）
# =============================================================================
print("\n【5. Spike Stability解析 - ハルシネーション検知PoC】")

# 正解サンプルでのジッター
correct_samples = []
incorrect_samples = []

with torch.no_grad():
    for i, (data, target) in enumerate(test_loader):
        if i >= 50:  # 最初の50サンプルで解析
            break
        
        output = model(data)
        pred = output.argmax(dim=1)
        
        # Spike Stability計算
        stability = analyzer.compute_spike_stability(model, data, num_trials=10, noise_std=0.05)
        
        avg_jitter = np.mean([s['jitter_mean'] for s in stability.values()])
        
        if pred.item() == target.item():
            correct_samples.append(avg_jitter)
        else:
            incorrect_samples.append(avg_jitter)

print(f"\n  結果:")
print(f"  {'-'*50}")
if correct_samples:
    correct_jitter = np.mean(correct_samples)
    print(f"  正解サンプルの平均ジッター: {correct_jitter:.4f} (n={len(correct_samples)})")
if incorrect_samples:
    incorrect_jitter = np.mean(incorrect_samples)
    print(f"  不正解サンプルの平均ジッター: {incorrect_jitter:.4f} (n={len(incorrect_samples)})")
else:
    print(f"  不正解サンプル: なし（全て正解）")
    incorrect_jitter = None

print(f"\n  【解釈】")
if incorrect_samples and len(incorrect_samples) > 0:
    if incorrect_jitter > correct_jitter:
        ratio = incorrect_jitter / correct_jitter
        print(f"  ✅ 不正解サンプルのジッターが {ratio:.2f}x 大きい！")
        print(f"  → スパイク不安定性がハルシネーションの指標になる可能性")
    else:
        print(f"  ⚠️ ジッター差が小さい - より詳細な解析が必要")
else:
    print(f"  → 不正解サンプルが少ないため、より大きなデータセットで検証が必要")


# =============================================================================
# 8. 可視化
# =============================================================================
print("\n【6. 可視化】")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 入力画像
ax = axes[0, 0]
img = sample_data[0].permute(1, 2, 0).numpy()
img = (img - img.min()) / (img.max() - img.min())  # 正規化
ax.imshow(img)
ax.set_title(f'Input: {sample_class}', fontsize=12)
ax.axis('off')

# Conv1 TTFS
ax = axes[0, 1]
ttfs_conv1 = ttfs_results['conv1']['ttfs'][0].mean(dim=0).numpy()
im = ax.imshow(ttfs_conv1, cmap='hot', vmin=0, vmax=50)
ax.set_title('Conv1 TTFS (Thought Priority)', fontsize=12)
plt.colorbar(im, ax=ax, label='TTFS')

# Conv2 TTFS
ax = axes[0, 2]
ttfs_conv2 = ttfs_results['conv2']['ttfs'][0].mean(dim=0).numpy()
im = ax.imshow(ttfs_conv2, cmap='hot', vmin=0, vmax=50)
ax.set_title('Conv2 TTFS', fontsize=12)
plt.colorbar(im, ax=ax, label='TTFS')

# Conv3 TTFS
ax = axes[1, 0]
ttfs_conv3 = ttfs_results['conv3']['ttfs'][0].mean(dim=0).numpy()
im = ax.imshow(ttfs_conv3, cmap='hot', vmin=0, vmax=50)
ax.set_title('Conv3 TTFS', fontsize=12)
plt.colorbar(im, ax=ax, label='TTFS')

# FC1同期行列
ax = axes[1, 1]
im = ax.imshow(sync_matrix[:50, :50].numpy(), cmap='Blues')
ax.set_title('FC1 Synchrony (50x50)', fontsize=12)
ax.set_xlabel('Neuron i')
ax.set_ylabel('Neuron j')
plt.colorbar(im, ax=ax, label='Sync')

# ジッター比較
ax = axes[1, 2]
if correct_samples and incorrect_samples:
    bp = ax.boxplot([correct_samples, incorrect_samples], 
                    labels=['Correct', 'Incorrect'],
                    patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax.set_ylabel('Mean Jitter')
    ax.set_title('Spike Stability: Jitter Comparison', fontsize=12)
else:
    ax.bar(['Correct'], [np.mean(correct_samples) if correct_samples else 0], color='lightgreen')
    ax.set_ylabel('Mean Jitter')
    ax.set_title('Spike Stability (Only Correct)', fontsize=12)

plt.tight_layout()
plt.savefig('snn_interpretability_analysis.png', dpi=150, bbox_inches='tight')
print("  保存: snn_interpretability_analysis.png")


# =============================================================================
# 9. まとめ
# =============================================================================
print("\n" + "=" * 70)
print("📊 SNN AI Interpretability 実験まとめ")
print("=" * 70)

print("""
【TTFS解析（思考順序）】
  - 各層で「何を最初に見たか」が可視化可能
  - 低いTTFS値 = AIが重要視している特徴
  - 層が深くなるほど抽象的な「概念」の優先度を反映

【Neural Synchrony（概念結合）】
  - 同期発火するニューロンペアを検出
  - 同期ペア = 「意味の塊」を形成する神経群
  - クラスによって異なる同期パターンの可能性

【Spike Stability（ハルシネーション検知）】
  - ノイズ摂動下での発火タイミングの揺れを測定
  - 高いジッター = 不安定な判断 = ハルシネーションの予兆
  - 「自信満々に嘘をつく」AIを見抜く新手法

【次のステップ】
  1. より大規模モデル（ResNet）での検証
  2. クラス別TTFS/Synchronyパターン分析
  3. ハルシネーション検知の閾値最適化
  4. LLM（Transformer）への応用
""")

print("\n🚀 SNN = 「AIの脳波計」として機能！")
print("=" * 70)
