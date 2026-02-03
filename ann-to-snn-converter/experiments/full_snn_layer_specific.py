"""
全層SNN＋層別閾値調整＋層間ハイブリッド読み出し
==============================================

各層ごとに最適なα値を設定し、
層間でもハイブリッド読み出し（スパイク＋膜電位）を使用

Author: ろーる (Cell Activation)
Date: 2026-02-03
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

print("=" * 70)
print("🧠 全層SNN＋層別閾値調整＋層間ハイブリッド読み出し")
print("=" * 70)

# ============================================================
# 1. データ準備
# ============================================================
print("\n【1. CIFAR-10データ準備】")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_subset = torch.utils.data.Subset(trainset, range(10000))
test_subset = torch.utils.data.Subset(testset, range(500))  # 少なめ（全層SNNは遅い）

trainloader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=0)
testloader = DataLoader(test_subset, batch_size=32, shuffle=False, num_workers=0)

print(f"  訓練: {len(train_subset)}, テスト: {len(test_subset)}")

# ============================================================
# 2. シンプルなCNN（ANN）
# ============================================================
print("\n【2. CNN構築＆学習】")

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.pool2 = nn.AvgPool2d(2)
        self.fc = nn.Linear(64 * 8 * 8, 10, bias=False)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = SimpleCNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    for inputs, labels in trainloader:
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()

model.eval()
correct = total = 0
with torch.no_grad():
    for inputs, labels in testloader:
        _, predicted = model(inputs).max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
ann_acc = 100. * correct / total
print(f"  ANN精度: {ann_acc:.1f}%")

# ============================================================
# 3. 各層の最大活性化を計測
# ============================================================
print("\n【3. 各層の最大活性化計測】")

activation_max = {'conv1': 0, 'conv2': 0, 'fc': 0}

with torch.no_grad():
    for inputs, _ in trainloader:
        x = torch.relu(model.conv1(inputs))
        activation_max['conv1'] = max(activation_max['conv1'], x.abs().max().item())
        x = model.pool1(x)
        x = torch.relu(model.conv2(x))
        activation_max['conv2'] = max(activation_max['conv2'], x.abs().max().item())
        x = model.pool2(x)
        x = x.view(x.size(0), -1)
        out = model.fc(x)
        activation_max['fc'] = max(activation_max['fc'], out.abs().max().item())

for name, val in activation_max.items():
    print(f"  {name}: {val:.2f}")

# ============================================================
# 4. 全層SNN推論（層ごとのα、層間ハイブリッド読み出し）
# ============================================================
print("\n【4. 全層SNN推論（層間ハイブリッド読み出し）】")

def full_snn_inference(x_np, alphas, T, hybrid_weight=0.7):
    """
    全層をSNN化、層間でもハイブリッド読み出しを使用
    
    alphas: {'conv1': α1, 'conv2': α2, 'fc': α3}
    hybrid_weight: スパイクの重み（1-hybrid_weight = 膜電位の重み）
    """
    batch_size = x_np.shape[0]
    
    # 閾値設定
    thresholds = {name: alphas[name] * activation_max[name] for name in alphas}
    
    # 重み取得
    w_conv1 = model.conv1.weight.detach().numpy()
    w_conv2 = model.conv2.weight.detach().numpy()
    w_fc = model.fc.weight.detach().numpy()
    
    # === Conv1 SNN ===
    mem_conv1 = np.zeros((batch_size, 32, 32, 32))
    spike_count_conv1 = np.zeros_like(mem_conv1)
    
    input_per_step = x_np / T
    
    for t in range(T):
        with torch.no_grad():
            conv1_out = torch.nn.functional.conv2d(
                torch.tensor(input_per_step, dtype=torch.float32),
                model.conv1.weight, padding=1
            ).numpy()
        
        mem_conv1 += conv1_out
        spikes = (mem_conv1 >= thresholds['conv1']).astype(float)
        mem_conv1 -= spikes * thresholds['conv1']
        spike_count_conv1 += spikes
    
    # 層間ハイブリッド読み出し（スパイク率＋膜電位）
    conv1_output = (hybrid_weight * (spike_count_conv1 / T) + 
                   (1 - hybrid_weight) * (mem_conv1 / thresholds['conv1']))
    
    # AvgPool
    pool1_out = conv1_output.reshape(batch_size, 32, 16, 2, 16, 2).mean(axis=(3, 5))
    
    # === Conv2 SNN ===
    mem_conv2 = np.zeros((batch_size, 64, 16, 16))
    spike_count_conv2 = np.zeros_like(mem_conv2)
    
    input2_per_step = pool1_out / T
    
    for t in range(T):
        with torch.no_grad():
            conv2_out = torch.nn.functional.conv2d(
                torch.tensor(input2_per_step * T, dtype=torch.float32),  # スケール調整
                model.conv2.weight, padding=1
            ).numpy()
        
        mem_conv2 += conv2_out / T
        spikes = (mem_conv2 >= thresholds['conv2']).astype(float)
        mem_conv2 -= spikes * thresholds['conv2']
        spike_count_conv2 += spikes
    
    # 層間ハイブリッド読み出し
    conv2_output = (hybrid_weight * (spike_count_conv2 / T) + 
                   (1 - hybrid_weight) * (mem_conv2 / thresholds['conv2']))
    
    # AvgPool
    pool2_out = conv2_output.reshape(batch_size, 64, 8, 2, 8, 2).mean(axis=(3, 5))
    
    # === FC SNN ===
    flat = pool2_out.reshape(batch_size, -1)
    
    mem_fc = np.zeros((batch_size, 10))
    spike_count_fc = np.zeros_like(mem_fc)
    
    fc_current = flat @ w_fc.T
    fc_per_step = fc_current / T
    
    for t in range(T):
        mem_fc += fc_per_step
        spikes = (mem_fc >= thresholds['fc']).astype(float)
        mem_fc -= spikes * thresholds['fc']
        spike_count_fc += spikes
    
    # 最終ハイブリッド読み出し
    output = (hybrid_weight * (spike_count_fc / T) + 
             (1 - hybrid_weight) * (mem_fc / thresholds['fc']))
    
    return np.argmax(output, axis=1)


# テスト
print("\n  層別α値での精度テスト:")
print("-" * 80)
print(f"  {'Conv1 α':>8} | {'Conv2 α':>8} | {'FC α':>8} | {'SNN精度':>10} | {'ANN差':>8}")
print("-" * 80)

T = 30  # タイムステップ

# 様々なα値の組み合わせをテスト
test_configs = [
    # 全て同じα
    {'conv1': 1.0, 'conv2': 1.0, 'fc': 1.0},
    {'conv1': 2.0, 'conv2': 2.0, 'fc': 2.0},
    {'conv1': 3.0, 'conv2': 3.0, 'fc': 3.0},
    # 層ごとに異なるα（前が低い）
    {'conv1': 1.0, 'conv2': 1.5, 'fc': 2.0},
    {'conv1': 1.5, 'conv2': 2.0, 'fc': 2.5},
    # 層ごとに異なるα（前が高い）
    {'conv1': 3.0, 'conv2': 2.5, 'fc': 2.0},
    {'conv1': 2.5, 'conv2': 2.0, 'fc': 1.5},
    # 特殊パターン
    {'conv1': 0.5, 'conv2': 2.0, 'fc': 2.0},  # 入力層だけ低め
    {'conv1': 2.0, 'conv2': 0.5, 'fc': 2.0},  # 中間層だけ低め
    {'conv1': 2.0, 'conv2': 2.0, 'fc': 0.5},  # 出力層だけ低め
]

results = []

for alphas in test_configs:
    correct = 0
    total = 0
    
    for inputs, labels in testloader:
        x_np = inputs.numpy()
        preds = full_snn_inference(x_np, alphas, T)
        correct += (preds == labels.numpy()).sum()
        total += len(labels)
    
    snn_acc = 100. * correct / total
    diff = snn_acc - ann_acc
    marker = " ✅" if abs(diff) < 5 else ""
    results.append((alphas, snn_acc, diff))
    
    print(f"  {alphas['conv1']:>8.1f} | {alphas['conv2']:>8.1f} | {alphas['fc']:>8.1f} | "
          f"{snn_acc:>9.1f}% | {diff:>+7.1f}%{marker}")

print("-" * 80)

# 最良の結果を表示
best = max(results, key=lambda x: x[1])
print(f"\n  【最良】α={best[0]} → {best[1]:.1f}%")

# ============================================================
# 5. 結果
# ============================================================
print("\n" + "=" * 70)
print("📊 全層SNN＋層間ハイブリッド読み出し 結果")
print("=" * 70)
print(f"""
  【ANN精度】{ann_acc:.1f}%
  
  【手法】
  - 全層をIFニューロンでSNN化
  - 各層で独立したα値（閾値=α×max_activation）
  - 層間でハイブリッド読み出し（70%スパイク + 30%膜電位）
  
  【発見】
  - 層間ハイブリッド読み出しで全層SNNでも精度維持？
  - 最適なα値は層によって異なる？
""")
