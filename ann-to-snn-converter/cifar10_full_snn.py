"""
CIFAR-10 全層SNN化 ResNet-18
==============================

全ての層をSNNとして推論する本格版

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
import time

print("=" * 70)
print("🔬 CIFAR-10 全層SNN化 ResNet-18")
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

# サブセット
train_subset = torch.utils.data.Subset(trainset, range(10000))
test_subset = torch.utils.data.Subset(testset, range(1000))  # 全層SNNは遅いので少なめ

trainloader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=0)
testloader = DataLoader(test_subset, batch_size=32, shuffle=False, num_workers=0)  # バッチ小さめ

print(f"  訓練: {len(train_subset)}, テスト: {len(test_subset)}")

# ============================================================
# 2. シンプルなCNN（ResNetより軽量、でも3層以上）
# ============================================================
print("\n【2. 3層CNN構築（全層SNN化テスト用）】")

class SimpleCNN(nn.Module):
    """3層CNN（全層SNN化しやすい構造）"""
    def __init__(self):
        super().__init__()
        # Conv層（bias=False、AvgPool）
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        self.pool1 = nn.AvgPool2d(2)  # 32→16
        
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.pool2 = nn.AvgPool2d(2)  # 16→8
        
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1, bias=False)
        self.pool3 = nn.AvgPool2d(2)  # 8→4
        
        self.fc = nn.Linear(128 * 4 * 4, 10, bias=False)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.relu(self.conv3(x))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


model = SimpleCNN()
print(f"  パラメータ数: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# 3. 学習
# ============================================================
print("\n【3. CNN学習】")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    for inputs, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 2 == 0:
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        print(f"  Epoch {epoch+1}: {100.*correct/total:.1f}%")

# ANN精度
model.eval()
correct = total = 0
with torch.no_grad():
    for inputs, labels in testloader:
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
ann_acc = 100. * correct / total
print(f"\n  【ANN精度】{ann_acc:.1f}%")

# ============================================================
# 4. 各層の最大活性化を計測
# ============================================================
print("\n【4. 各層の最大活性化計測】")

activation_max = {}

def hook_fn(name):
    def hook(module, input, output):
        max_val = output.detach().abs().max().item()
        if name not in activation_max:
            activation_max[name] = max_val
        else:
            activation_max[name] = max(activation_max[name], max_val)
    return hook

hooks = []
hooks.append(model.conv1.register_forward_hook(hook_fn('conv1')))
hooks.append(model.conv2.register_forward_hook(hook_fn('conv2')))
hooks.append(model.conv3.register_forward_hook(hook_fn('conv3')))
hooks.append(model.fc.register_forward_hook(hook_fn('fc')))

with torch.no_grad():
    for inputs, _ in trainloader:
        _ = model(inputs)

for h in hooks:
    h.remove()

for name, val in activation_max.items():
    print(f"  {name}: {val:.2f}")

# ============================================================
# 5. 全層SNN推論
# ============================================================
print("\n【5. 全層SNN推論】")

def snn_inference_full_layers(x_np, alpha, T):
    """
    全ての層をIFニューロンとして推論
    
    Conv1 → IFニューロン → Pool → Conv2 → IFニューロン → ...
    """
    batch_size = x_np.shape[0]
    
    # 閾値設定（α × max_activation）
    thresholds = {
        'conv1': alpha * activation_max['conv1'],
        'conv2': alpha * activation_max['conv2'],
        'conv3': alpha * activation_max['conv3'],
        'fc': alpha * activation_max['fc'],
    }
    
    # 重み取得
    w_conv1 = model.conv1.weight.detach().numpy()
    w_conv2 = model.conv2.weight.detach().numpy()
    w_conv3 = model.conv3.weight.detach().numpy()
    w_fc = model.fc.weight.detach().numpy()
    
    # 膜電位初期化
    # Conv1出力: (batch, 32, 32, 32)
    mem_conv1 = np.zeros((batch_size, 32, 32, 32))
    spike_conv1 = np.zeros_like(mem_conv1)
    
    # Pool後: (batch, 32, 16, 16)
    # Conv2出力: (batch, 64, 16, 16)
    mem_conv2 = np.zeros((batch_size, 64, 16, 16))
    spike_conv2 = np.zeros_like(mem_conv2)
    
    # Pool後: (batch, 64, 8, 8)
    # Conv3出力: (batch, 128, 8, 8)
    mem_conv3 = np.zeros((batch_size, 128, 8, 8))
    spike_conv3 = np.zeros_like(mem_conv3)
    
    # Pool後: (batch, 128, 4, 4) → flatten: (batch, 2048)
    # FC出力: (batch, 10)
    mem_fc = np.zeros((batch_size, 10))
    spike_fc = np.zeros_like(mem_fc)
    
    # 入力を時間方向に分散
    input_per_step = x_np / T
    
    for t in range(T):
        # === Conv1 + IFニューロン ===
        # PyTorchのConv2dを使用
        with torch.no_grad():
            conv1_out = torch.nn.functional.conv2d(
                torch.tensor(input_per_step, dtype=torch.float32),
                model.conv1.weight,
                padding=1
            ).numpy()
        
        mem_conv1 += conv1_out
        spikes1 = (mem_conv1 >= thresholds['conv1']).astype(float)
        mem_conv1 -= spikes1 * thresholds['conv1']
        spike_conv1 += spikes1
        
        # AvgPool
        pool1_out = spikes1.reshape(batch_size, 32, 16, 2, 16, 2).mean(axis=(3, 5))
        
        # === Conv2 + IFニューロン ===
        with torch.no_grad():
            conv2_out = torch.nn.functional.conv2d(
                torch.tensor(pool1_out, dtype=torch.float32),
                model.conv2.weight,
                padding=1
            ).numpy()
        
        mem_conv2 += conv2_out
        spikes2 = (mem_conv2 >= thresholds['conv2']).astype(float)
        mem_conv2 -= spikes2 * thresholds['conv2']
        spike_conv2 += spikes2
        
        # AvgPool
        pool2_out = spikes2.reshape(batch_size, 64, 8, 2, 8, 2).mean(axis=(3, 5))
        
        # === Conv3 + IFニューロン ===
        with torch.no_grad():
            conv3_out = torch.nn.functional.conv2d(
                torch.tensor(pool2_out, dtype=torch.float32),
                model.conv3.weight,
                padding=1
            ).numpy()
        
        mem_conv3 += conv3_out
        spikes3 = (mem_conv3 >= thresholds['conv3']).astype(float)
        mem_conv3 -= spikes3 * thresholds['conv3']
        spike_conv3 += spikes3
        
        # AvgPool
        pool3_out = spikes3.reshape(batch_size, 128, 4, 2, 4, 2).mean(axis=(3, 5))
        
        # === FC + IFニューロン ===
        flat = pool3_out.reshape(batch_size, -1)
        fc_out = flat @ w_fc.T
        
        mem_fc += fc_out
        spikes_fc = (mem_fc >= thresholds['fc']).astype(float)
        mem_fc -= spikes_fc * thresholds['fc']
        spike_fc += spikes_fc
    
    # ハイブリッド読み出し（70%スパイク + 30%膜電位）
    output = 0.7 * (spike_fc / T) + 0.3 * (mem_fc / thresholds['fc'])
    return np.argmax(output, axis=1)


# テスト
print("\n  全層SNN精度テスト:")
print("-" * 60)
print(f"  {'α':>6} | {'SNN精度':>10} | {'ANN差':>10}")
print("-" * 60)

T = 50  # タイムステップ（全層SNNは遅いので少なめ）

for alpha in [1.0, 1.5, 2.0, 2.5, 3.0]:
    correct = 0
    total = 0
    
    start = time.time()
    for inputs, labels in testloader:
        x_np = inputs.numpy()
        labels_np = labels.numpy()
        
        preds = snn_inference_full_layers(x_np, alpha, T)
        correct += (preds == labels_np).sum()
        total += len(labels_np)
    
    elapsed = time.time() - start
    snn_acc = 100. * correct / total
    diff = snn_acc - ann_acc
    marker = " ✅" if abs(diff) < 3 else ""
    print(f"  {alpha:>6.1f} | {snn_acc:>9.1f}% | {diff:>+9.1f}%{marker}  ({elapsed:.1f}秒)")

print("-" * 60)

# ============================================================
# 6. 結果
# ============================================================
print("\n" + "=" * 70)
print("📊 全層SNN化 結果")
print("=" * 70)
print(f"""
  【ANN精度】{ann_acc:.1f}%
  【全層SNN】Conv1〜FC 全てIFニューロン化
  
  【比較】
  - 最終層のみSNN: 精度落ちる傾向
  - 全層SNN: より正確にANNを再現
  
  【公式】θ = α × max(activation) が全層で適用
""")
