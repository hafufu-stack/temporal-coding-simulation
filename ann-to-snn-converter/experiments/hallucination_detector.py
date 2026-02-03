"""
SNN Hallucination Detector: ジッターベース分類器
=================================================

Softmax確率とスパイクジッターを組み合わせて
AIの「自信過剰な誤り」（ハルシネーション）を検知する。

発見: 正の相関 r=0.22 を活用
- 高確信度 + 高ジッター = 危険信号（自信過剰）
- 高確信度 + 低ジッター = 安全（本当に確信）

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
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = ['MS Gothic', 'DejaVu Sans']

print("=" * 70)
print("🔍 SNN Hallucination Detector: ジッターベース信頼度補正")
print("=" * 70)


# =============================================================================
# 1. モデル定義
# =============================================================================
class ResBlock(nn.Module):
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
        return F.relu(out)


class SmallResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        self.activations = {}
    
    def _make_layer(self, in_ch, out_ch, blocks, stride=1):
        layers = [ResBlock(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(ResBlock(out_ch, out_ch))
        return nn.Sequential(*layers)
    
    def forward(self, x, save_act=False):
        x = F.relu(self.bn1(self.conv1(x)))
        if save_act: self.activations['conv1'] = x.clone()
        x = self.layer1(x)
        if save_act: self.activations['layer1'] = x.clone()
        x = self.layer2(x)
        if save_act: self.activations['layer2'] = x.clone()
        x = self.layer3(x)
        if save_act: self.activations['layer3'] = x.clone()
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    
    def get_activations(self, x):
        self.activations = {}
        _ = self.forward(x, save_act=True)
        return self.activations


# =============================================================================
# 2. ハルシネーション検知器
# =============================================================================
class HallucinationDetector:
    """SNNジッターによるハルシネーション検知"""
    
    def __init__(self, model, timesteps=100, num_trials=15):
        self.model = model
        self.timesteps = timesteps
        self.num_trials = num_trials
        
        # 検知閾値（後で学習）
        self.jitter_threshold = None
        self.combined_threshold = None
    
    def compute_ttfs(self, activation):
        """TTFS計算"""
        ttfs = torch.full_like(activation, float(self.timesteps))
        active_mask = activation > 0
        if active_mask.any():
            max_act = activation.max()
            if max_act > 0:
                normalized = activation[active_mask] / max_act
                ttfs[active_mask] = self.timesteps * (1 - normalized)
        return ttfs
    
    def compute_jitter(self, x, noise_std=0.1):
        """ジッター計算（複数回ノイズ付加で発火タイミングの揺れを測定）"""
        all_ttfs = []
        
        self.model.eval()
        with torch.no_grad():
            for _ in range(self.num_trials):
                noisy_x = x + torch.randn_like(x) * noise_std
                noisy_x = torch.clamp(noisy_x, 0, 1)
                
                activations = self.model.get_activations(noisy_x)
                
                layer_ttfs = []
                for layer_name, act in activations.items():
                    ttfs = self.compute_ttfs(act)
                    layer_ttfs.append(ttfs.mean().item())
                
                all_ttfs.append(np.mean(layer_ttfs))
        
        return np.std(all_ttfs)  # ジッター = タイムステップ間のばらつき
    
    def get_confidence(self, x):
        """Softmax確信度を取得"""
        self.model.eval()
        with torch.no_grad():
            output = self.model(x)
            probs = F.softmax(output, dim=1)
            confidence = probs.max().item()
            pred = output.argmax(dim=1).item()
        return confidence, pred
    
    def compute_risk_score(self, x, noise_std=0.1):
        """
        リスクスコア計算
        
        低スコア = 安全（高確信度 + 低ジッター）
        高スコア = 危険（高確信度 + 高ジッター = 自信過剰）
        """
        confidence, pred = self.get_confidence(x)
        jitter = self.compute_jitter(x, noise_std)
        
        # リスクスコア = ジッター × 確信度
        # 高確信度で高ジッター = 最も危険
        risk_score = jitter * confidence
        
        return {
            'confidence': confidence,
            'jitter': jitter,
            'risk_score': risk_score,
            'prediction': pred
        }
    
    def calibrate(self, dataloader, num_samples=100):
        """検知閾値をキャリブレーション"""
        print("\n  キャリブレーション中...")
        
        correct_risks = []
        incorrect_risks = []
        
        for i, (data, target) in enumerate(dataloader):
            if i >= num_samples:
                break
            
            result = self.compute_risk_score(data)
            is_correct = result['prediction'] == target.item()
            
            if is_correct:
                correct_risks.append(result['risk_score'])
            else:
                incorrect_risks.append(result['risk_score'])
            
            if (i + 1) % 25 == 0:
                print(f"    処理: {i+1}/{num_samples}")
        
        # 閾値決定: 正解の95パーセンタイルを採用
        if correct_risks:
            self.jitter_threshold = np.percentile(correct_risks, 95)
        
        print(f"\n  正解サンプル: {len(correct_risks)}")
        print(f"  不正解サンプル: {len(incorrect_risks)}")
        print(f"  閾値: {self.jitter_threshold:.4f}")
        
        return correct_risks, incorrect_risks
    
    def is_hallucination(self, x):
        """ハルシネーション判定"""
        if self.jitter_threshold is None:
            raise ValueError("先にcalibrateを実行してください")
        
        result = self.compute_risk_score(x)
        is_risky = result['risk_score'] > self.jitter_threshold
        
        return {
            **result,
            'is_hallucination': is_risky,
            'threshold': self.jitter_threshold
        }


# =============================================================================
# 3. データ準備
# =============================================================================
print("\n【1. データ準備】")
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_subset = torch.utils.data.Subset(train_dataset, range(8000))
test_subset = torch.utils.data.Subset(test_dataset, range(500))
val_subset = torch.utils.data.Subset(test_dataset, range(500, 1000))

train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_subset, batch_size=1, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_subset, batch_size=1, shuffle=False)

class_names = ['airplane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

print(f"  訓練: {len(train_subset)}, テスト: {len(test_subset)}, 検証: {len(val_subset)}")


# =============================================================================
# 4. モデル学習
# =============================================================================
print("\n【2. SmallResNet学習】")
model = SmallResNet(num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(15):
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
# 5. 検知器のキャリブレーション
# =============================================================================
print("\n【3. ハルシネーション検知器キャリブレーション】")

detector = HallucinationDetector(model, timesteps=100, num_trials=10)
correct_risks, incorrect_risks = detector.calibrate(test_loader, num_samples=100)


# =============================================================================
# 6. 検証セットで評価
# =============================================================================
print("\n【4. 検証セットでの評価】")

predictions = []
labels = []  # 1 = 不正解（ハルシネーション扱い）
risk_scores = []
confidences = []

model.eval()
for i, (data, target) in enumerate(val_loader):
    if i >= 150:
        break
    
    result = detector.is_hallucination(data)
    is_correct = result['prediction'] == target.item()
    
    predictions.append(1 if result['is_hallucination'] else 0)
    labels.append(0 if is_correct else 1)  # 不正解を1とする
    risk_scores.append(result['risk_score'])
    confidences.append(result['confidence'])
    
    if (i + 1) % 50 == 0:
        print(f"  処理: {i+1}/150")

# 評価指標
print(f"\n  評価結果:")
print(f"  {'-'*50}")

# 混同行列
tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)
fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"  真陽性 (正しく誤りを検知): {tp}")
print(f"  偽陽性 (正解を誤りと判定): {fp}")
print(f"  真陰性 (正しく正解と判定): {tn}")
print(f"  偽陰性 (誤りを見逃し): {fn}")
print(f"\n  適合率 (Precision): {precision:.4f}")
print(f"  再現率 (Recall): {recall:.4f}")
print(f"  F1スコア: {f1:.4f}")

# AUC-ROC
if len(set(labels)) > 1:
    try:
        auc = roc_auc_score(labels, risk_scores)
        print(f"  AUC-ROC: {auc:.4f}")
    except:
        auc = None
        print(f"  AUC-ROC: 計算不可")
else:
    auc = None
    print(f"  AUC-ROC: 単一クラスのため計算不可")


# =============================================================================
# 7. 可視化
# =============================================================================
print("\n【5. 可視化】")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# リスクスコア分布
ax = axes[0, 0]
correct_risks_val = [r for r, l in zip(risk_scores, labels) if l == 0]
incorrect_risks_val = [r for r, l in zip(risk_scores, labels) if l == 1]
ax.hist(correct_risks_val, bins=20, alpha=0.7, label='Correct', color='green')
ax.hist(incorrect_risks_val, bins=20, alpha=0.7, label='Incorrect', color='red')
ax.axvline(detector.jitter_threshold, color='blue', linestyle='--', label=f'Threshold={detector.jitter_threshold:.3f}')
ax.set_xlabel('Risk Score (Jitter × Confidence)')
ax.set_ylabel('Count')
ax.set_title('Risk Score Distribution')
ax.legend()

# ROC曲線
ax = axes[0, 1]
if len(set(labels)) > 1:
    fpr, tpr, _ = roc_curve(labels, risk_scores)
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC={auc:.3f}' if auc else 'AUC=N/A')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
else:
    ax.text(0.5, 0.5, 'Not enough data', ha='center', va='center')
    ax.set_title('ROC Curve (N/A)')

# 確信度 vs リスクスコア
ax = axes[1, 0]
colors = ['green' if l == 0 else 'red' for l in labels]
ax.scatter(confidences, risk_scores, c=colors, alpha=0.7)
ax.axhline(detector.jitter_threshold, color='blue', linestyle='--', label='Threshold')
ax.set_xlabel('Softmax Confidence')
ax.set_ylabel('Risk Score')
ax.set_title('Confidence vs Risk Score')
ax.legend()

# 混同行列
ax = axes[1, 1]
cm = [[tn, fp], [fn, tp]]
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Predicted\nCorrect', 'Predicted\nHallucination'])
ax.set_yticklabels(['Actual\nCorrect', 'Actual\nWrong'])
ax.set_title('Confusion Matrix')
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i][j]), ha='center', va='center', fontsize=16)
plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('hallucination_detector_results.png', dpi=150, bbox_inches='tight')
print("  保存: hallucination_detector_results.png")


# =============================================================================
# 8. まとめ
# =============================================================================
print("\n" + "=" * 70)
print("🔍 SNN Hallucination Detector まとめ")
print("=" * 70)

print(f"""
【手法】
  リスクスコア = ジッター × Softmax確信度
  
  高確信度 + 高ジッター = 「自信過剰な誤り」= ハルシネーション
  
【評価結果】
  - Precision: {precision:.4f}
  - Recall: {recall:.4f}
  - F1: {f1:.4f}
  - 閾値: {detector.jitter_threshold:.4f}

【解釈】
  - SNNのジッターは「判断の安定性」を表す
  - Softmax確信度だけでは見抜けない誤りを検知可能
  - 「確信満々だが不安定」= AI過信の警告

【今後の課題】
  1. より大規模データでの検証
  2. 最適な閾値の自動調整
  3. Transformer/LLMへの応用
  4. リアルタイム検知API
""")

print("\n🚀 SNN = 「AIの嘘発見器」！過信を見抜く新技術")
print("=" * 70)
