"""ResNet α検証スクリプト"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

print('=' * 70)
print('ResNet での α公式検証')
print('=' * 70)

# データ
np.random.seed(42)
def digit(l):
    img = np.zeros((28,28), np.float32)
    patterns = {
        0: [(slice(y-1,y+2),slice(x-1,x+2)) for x,y in [(int(14+7*np.cos(a)),int(14+7*np.sin(a))) for a in np.linspace(0,6.28,30)] if 0<=x<27 and 0<=y<27],
        1: [(slice(4,24),slice(13,16))],
        2: [(slice(5,8),slice(8,20)),(slice(12,15),slice(8,20)),(slice(20,23),slice(8,20))],
        3: [(slice(5,8),slice(8,20)),(slice(12,15),slice(10,20)),(slice(20,23),slice(8,20)),(slice(6,22),slice(17,20))],
        4: [(slice(5,14),slice(8,11)),(slice(12,15),slice(8,20)),(slice(5,23),slice(17,20))],
        5: [(slice(5,8),slice(8,20)),(slice(6,14),slice(8,11)),(slice(12,15),slice(8,20)),(slice(14,22),slice(17,20)),(slice(20,23),slice(8,20))],
        6: [(slice(5,22),slice(8,11)),(slice(12,15),slice(8,20)),(slice(14,22),slice(17,20)),(slice(20,23),slice(8,20))],
        7: [(slice(5,8),slice(8,20)),(slice(6,23),slice(17,20))],
        8: [(slice(5,8),slice(8,20)),(slice(12,15),slice(8,20)),(slice(20,23),slice(8,20)),(slice(6,14),slice(8,11)),(slice(6,14),slice(17,20)),(slice(14,22),slice(8,11)),(slice(14,22),slice(17,20))],
        9: [(slice(5,8),slice(8,20)),(slice(6,14),slice(8,11)),(slice(6,22),slice(17,20)),(slice(12,15),slice(8,20))],
    }
    for s in patterns.get(l, []):
        if isinstance(s, tuple) and len(s) == 2: img[s] = 1.0
    return np.clip(img + np.random.randn(28,28)*0.1, 0, 1)

train_x = np.array([digit(i%10) for i in range(1000)])
train_y = np.array([i%10 for i in range(1000)])
test_x = np.array([digit(i%10) for i in range(200)])
test_y = np.array([i%10 for i in range(200)])
train_xt = torch.FloatTensor(train_x).unsqueeze(1)
train_yt = torch.LongTensor(train_y)
test_xt = torch.FloatTensor(test_x).unsqueeze(1)

# ResNet Block
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
    
    def forward(self, x):
        residual = x
        out = torch.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + residual  # Skip connection
        return torch.relu(out)

# Mini ResNet
class MiniResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1, bias=False)
        self.block1 = ResBlock(16)
        self.block2 = ResBlock(16)
        self.pool = nn.AvgPool2d(2, 2)
        self.fc = nn.Linear(16*7*7, 10, bias=False)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # 28->14
        x = self.block1(x)
        x = self.pool(x)  # 14->7
        x = self.block2(x)
        x = x.view(-1, 16*7*7)
        return self.fc(x)

# 学習
print("【ResNet学習】")
model = MiniResNet()
opt = optim.Adam(model.parameters(), lr=0.002)

for epoch in range(15):
    model.train()
    for i in range(0, 1000, 64):
        out = model(train_xt[i:i+64])
        loss = nn.CrossEntropyLoss()(out, train_yt[i:i+64])
        opt.zero_grad()
        loss.backward()
        opt.step()
    
    if (epoch + 1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            acc = (model(test_xt).argmax(1).numpy() == test_y).mean() * 100
        print(f"  Epoch {epoch+1}: {acc:.0f}%")

model.eval()
with torch.no_grad():
    ann_acc = (model(test_xt).argmax(1).numpy() == test_y).mean() * 100
print(f"  ANN最終精度: {ann_acc:.0f}%")

# 活性化統計
print("\n【活性化統計】")
max_fc = []
with torch.no_grad():
    for i in range(100):
        x = test_xt[i:i+1]
        h = model.pool(torch.relu(model.conv1(x)))
        h = model.block1(h)
        h = model.pool(h)
        h = model.block2(h)
        out = model.fc(h.view(-1))
        max_fc.append(out.max().item())

avg_max_fc = np.mean(max_fc)
print(f"  FC output max avg: {avg_max_fc:.2f}")

# SNN推論
def snn_forward(x, threshold, T=100):
    with torch.no_grad():
        xt = torch.FloatTensor(x).reshape(1, 1, 28, 28)
        h = model.pool(torch.relu(model.conv1(xt)))
        h = model.block1(h)
        h = model.pool(h)
        h = model.block2(h)
        fc_in = model.fc(h.view(-1)).numpy()
    
    membrane = np.zeros(10)
    spikes = np.zeros(10)
    
    for t in range(T):
        membrane += fc_in / T
        fired = membrane >= threshold
        membrane[fired] -= threshold
        spikes += fired
    
    output = 0.7 * spikes/T + 0.3 * membrane/max(threshold, 0.01)
    return int(np.argmax(output))

# αテスト
print("\n【異なるα値でのSNN精度】")
print("  α    | 閾値     | SNN精度 | ANN差")
print("-" * 45)

for alpha in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
    threshold = avg_max_fc * alpha
    
    correct = 0
    for i in range(200):
        pred = snn_forward(test_x[i], threshold)
        if pred == test_y[i]:
            correct += 1
    
    snn_acc = correct / 200 * 100
    diff = snn_acc - ann_acc
    mark = "✅" if abs(diff) < 5 else ""
    print(f"  {alpha:.1f}  | {threshold:7.2f} | {snn_acc:5.1f}%  | {diff:+5.1f}% {mark}")

print("-" * 45)
print()
print("=" * 70)
print("📐 ResNet検証結果")
print("=" * 70)
