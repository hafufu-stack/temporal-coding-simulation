"""
海馬インスパイア・ハイブリッドSNNアーキテクチャ
================================================

脳の海馬（DG-CA3-CA1）の役割分担にインスパイア：
- DG（入力）: ANN特徴抽出（強い入力を作る）
- CA3（中間）: 再帰SNN（パターン保持）
- CA1（出力）: SNN出力（α=2.0読み出し）

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
print("🧠 海馬インスパイア・ハイブリッドSNN")
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
test_subset = torch.utils.data.Subset(testset, range(1000))

trainloader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=0)
testloader = DataLoader(test_subset, batch_size=64, shuffle=False, num_workers=0)

print(f"  訓練: {len(train_subset)}, テスト: {len(test_subset)}")

# ============================================================
# 2. 海馬インスパイア・ハイブリッドモデル
# ============================================================
print("\n【2. ハイブリッドモデル構築】")

class HippocampalHybridNet(nn.Module):
    """
    海馬インスパイアのハイブリッドアーキテクチャ
    
    DG (Dentate Gyrus) 相当: 
      - Conv特徴抽出（ANN）
      - パターン分離、強い入力生成
    
    CA3 相当:
      - 再帰結合層（SNN風の処理）
      - パターン保持・連想
    
    CA1 相当:
      - 出力層（SNN、α=2.0）
      - 読み出し
    """
    def __init__(self):
        super().__init__()
        
        # === DG（歯状回）: 入力処理・パターン分離 ===
        # 強力な特徴抽出（デトネーター的）
        self.dg_conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.dg_pool = nn.AvgPool2d(2)  # 32→16
        self.dg_conv2 = nn.Conv2d(64, 128, 3, 1, 1, bias=False)
        self.dg_pool2 = nn.AvgPool2d(2)  # 16→8
        
        # === CA3: 再帰結合層 ===
        # 自己再帰的な処理をシミュレート
        self.ca3_hidden = nn.Linear(128 * 8 * 8, 256, bias=False)
        self.ca3_recurrent = nn.Linear(256, 256, bias=False)  # 再帰結合
        
        # === CA1: 出力層 ===
        self.ca1_output = nn.Linear(256, 10, bias=False)
    
    def forward(self, x):
        # DG処理
        x = torch.relu(self.dg_conv1(x))
        x = self.dg_pool(x)
        x = torch.relu(self.dg_conv2(x))
        x = self.dg_pool2(x)
        x = x.view(x.size(0), -1)
        
        # CA3処理（再帰なし、学習時）
        ca3 = torch.relu(self.ca3_hidden(x))
        
        # CA1出力
        out = self.ca1_output(ca3)
        return out
    
    def forward_with_recurrent(self, x, recurrent_steps=3):
        """再帰結合を使った推論"""
        # DG処理
        x = torch.relu(self.dg_conv1(x))
        x = self.dg_pool(x)
        x = torch.relu(self.dg_conv2(x))
        x = self.dg_pool2(x)
        x = x.view(x.size(0), -1)
        
        # CA3処理（再帰あり）
        ca3 = torch.relu(self.ca3_hidden(x))
        
        # 再帰結合を数ステップ回す（CA3の自己活性化）
        for _ in range(recurrent_steps):
            ca3_recur = torch.relu(self.ca3_recurrent(ca3))
            ca3 = 0.5 * ca3 + 0.5 * ca3_recur  # 元の活性化と混合
        
        # CA1出力
        out = self.ca1_output(ca3)
        return out


model = HippocampalHybridNet()
print(f"  パラメータ数: {sum(p.numel() for p in model.parameters()):,}")
print("""
  アーキテクチャ:
  ┌─────────────────────────────────────┐
  │ DG（歯状回）: Conv64 → Conv128      │ ← ANN特徴抽出
  │   ↓                                 │
  │ CA3: FC256 + 再帰結合               │ ← SNN風パターン保持
  │   ↓                                 │
  │ CA1: FC10（α=2.0 SNN出力）          │ ← SNN読み出し
  └─────────────────────────────────────┘
""")

# ============================================================
# 3. 学習
# ============================================================
print("\n【3. ハイブリッドモデル学習】")

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
print(f"\n  【ANN精度（通常推論）】{ann_acc:.1f}%")

# 再帰結合を使った推論
correct_recur = total = 0
with torch.no_grad():
    for inputs, labels in testloader:
        outputs = model.forward_with_recurrent(inputs, recurrent_steps=3)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct_recur += predicted.eq(labels).sum().item()
recur_acc = 100. * correct_recur / total
print(f"  【ANN精度（CA3再帰3step）】{recur_acc:.1f}%")

# ============================================================
# 4. CA1層のSNN化テスト
# ============================================================
print("\n【4. CA1層のSNN化（α=2.0）】")

# CA3出力の最大値を計測
ca3_max = 0
with torch.no_grad():
    for inputs, _ in trainloader:
        x = torch.relu(model.dg_conv1(inputs))
        x = model.dg_pool(x)
        x = torch.relu(model.dg_conv2(x))
        x = model.dg_pool2(x)
        x = x.view(x.size(0), -1)
        ca3 = torch.relu(model.ca3_hidden(x))
        ca3_max = max(ca3_max, ca3.abs().max().item())

print(f"  CA3最大活性化: {ca3_max:.2f}")

# CA1層をSNN化
def snn_ca1_inference(ca3_activation, alpha, T):
    """CA1層のみをSNN化して推論"""
    batch_size = ca3_activation.shape[0]
    w_ca1 = model.ca1_output.weight.detach().numpy()
    
    threshold = alpha * ca3_max
    
    # IFニューロン
    membrane = np.zeros((batch_size, 10))
    spike_count = np.zeros((batch_size, 10))
    
    # 入力を時間分散
    current = ca3_activation.numpy() @ w_ca1.T
    current_per_step = current / T
    
    for t in range(T):
        membrane += current_per_step
        spikes = (membrane >= threshold).astype(float)
        membrane -= spikes * threshold
        spike_count += spikes
    
    # ハイブリッド読み出し
    output = 0.7 * (spike_count / T) + 0.3 * (membrane / threshold)
    return np.argmax(output, axis=1)


print("\n  海馬SNN精度テスト:")
print("-" * 70)
print(f"  {'方式':^25} | {'α':>6} | {'精度':>8} | {'ANN差':>8}")
print("-" * 70)

T = 50

for alpha in [1.0, 1.5, 2.0, 2.5, 3.0]:
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in testloader:
            # DG + CA3処理（ANN）
            x = torch.relu(model.dg_conv1(inputs))
            x = model.dg_pool(x)
            x = torch.relu(model.dg_conv2(x))
            x = model.dg_pool2(x)
            x = x.view(x.size(0), -1)
            ca3 = torch.relu(model.ca3_hidden(x))
            
            # CA1をSNN化
            preds = snn_ca1_inference(ca3, alpha, T)
            correct += (preds == labels.numpy()).sum()
            total += len(labels)
    
    snn_acc = 100. * correct / total
    diff = snn_acc - ann_acc
    marker = " ✅" if abs(diff) < 3 else ""
    print(f"  {'DG(ANN)+CA3(ANN)+CA1(SNN)':<25} | {alpha:>6.1f} | {snn_acc:>7.1f}% | {diff:>+7.1f}%{marker}")

# CA3に再帰を追加した場合
print()
for alpha in [2.0]:
    for recur_steps in [0, 1, 3, 5]:
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in testloader:
                # DG処理
                x = torch.relu(model.dg_conv1(inputs))
                x = model.dg_pool(x)
                x = torch.relu(model.dg_conv2(x))
                x = model.dg_pool2(x)
                x = x.view(x.size(0), -1)
                
                # CA3処理（再帰付き）
                ca3 = torch.relu(model.ca3_hidden(x))
                for _ in range(recur_steps):
                    ca3_recur = torch.relu(model.ca3_recurrent(ca3))
                    ca3 = 0.5 * ca3 + 0.5 * ca3_recur
                
                # CA1をSNN化
                preds = snn_ca1_inference(ca3, alpha, T)
                correct += (preds == labels.numpy()).sum()
                total += len(labels)
        
        snn_acc = 100. * correct / total
        diff = snn_acc - ann_acc
        method = f"DG+CA3(再帰{recur_steps})+CA1(SNN)"
        marker = " ✅" if abs(diff) < 3 else ""
        print(f"  {method:<25} | {alpha:>6.1f} | {snn_acc:>7.1f}% | {diff:>+7.1f}%{marker}")

print("-" * 70)

# ============================================================
# 5. 結果
# ============================================================
print("\n" + "=" * 70)
print("📊 海馬インスパイア・ハイブリッドSNN 結果")
print("=" * 70)
print(f"""
  【ANN精度】{ann_acc:.1f}%
  
  【アーキテクチャ】
    DG（歯状回）: Conv層（ANN） - パターン分離
    CA3: FC層 + 再帰結合（ANN/SNN混合） - パターン保持
    CA1: 出力層（SNN、α=2.0） - 読み出し
  
  【仮説】
    - 脳の各部位は異なる性質を持つ
    - ANN（連続値処理）とSNN（スパイク処理）を
      部位ごとに使い分けることで精度維持できる？
""")
