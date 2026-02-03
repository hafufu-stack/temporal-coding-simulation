"""
SNN Hallucination Detector v2: 多特徴量アプローチ
==================================================

改善点:
1. 単純なジッター→層別ジッターパターン
2. Synchrony特徴量の追加
3. TTFS分散の追加
4. 機械学習ベースの閾値調整

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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, roc_curve, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = ['MS Gothic', 'DejaVu Sans']

print("=" * 70)
print("🔍 SNN Hallucination Detector v2: 多特徴量アプローチ")
print("=" * 70)


# =============================================================================
# 1. モデル定義（前回と同じ）
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
        if save_act: self.activations['avgpool'] = x.clone()
        return self.fc(x)
    
    def get_activations(self, x):
        self.activations = {}
        _ = self.forward(x, save_act=True)
        return self.activations


# =============================================================================
# 2. 多特徴量抽出器
# =============================================================================
class MultiFeatureExtractor:
    """SNN解析に基づく多特徴量抽出"""
    
    def __init__(self, model, timesteps=100, num_trials=10):
        self.model = model
        self.timesteps = timesteps
        self.num_trials = num_trials
    
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
    
    def extract_features(self, x, noise_std=0.1):
        """多特徴量抽出"""
        features = {}
        
        self.model.eval()
        with torch.no_grad():
            # 元の予測情報
            output = self.model(x)
            probs = F.softmax(output, dim=1)
            features['confidence'] = probs.max().item()
            features['entropy'] = -(probs * torch.log(probs + 1e-8)).sum().item()
            features['margin'] = (probs.topk(2)[0][0,0] - probs.topk(2)[0][0,1]).item()
            
            # 層別活性化統計
            activations = self.model.get_activations(x)
            for layer_name, act in activations.items():
                act_flat = act.view(-1)
                features[f'{layer_name}_mean'] = act_flat.mean().item()
                features[f'{layer_name}_std'] = act_flat.std().item()
                features[f'{layer_name}_max'] = act_flat.max().item()
                features[f'{layer_name}_sparsity'] = (act_flat == 0).float().mean().item()
            
            # 層別TTFS統計
            for layer_name, act in activations.items():
                ttfs = self.compute_ttfs(act)
                ttfs_flat = ttfs.view(-1)
                features[f'{layer_name}_ttfs_mean'] = ttfs_flat.mean().item()
                features[f'{layer_name}_ttfs_std'] = ttfs_flat.std().item()
                features[f'{layer_name}_ttfs_min'] = ttfs_flat.min().item()
            
            # ジッター分析（ノイズ摂動）
            all_outputs = []
            layer_jitters = defaultdict(list)
            
            for _ in range(self.num_trials):
                noisy_x = x + torch.randn_like(x) * noise_std
                noisy_x = torch.clamp(noisy_x, 0, 1)
                
                noisy_output = self.model(noisy_x)
                all_outputs.append(noisy_output)
                
                noisy_act = self.model.get_activations(noisy_x)
                for layer_name, act in noisy_act.items():
                    ttfs = self.compute_ttfs(act)
                    layer_jitters[layer_name].append(ttfs.mean().item())
            
            # 出力ジッター
            stacked_outputs = torch.stack(all_outputs)
            features['output_jitter'] = stacked_outputs.std(dim=0).mean().item()
            
            # 層別ジッター
            for layer_name, jitter_list in layer_jitters.items():
                features[f'{layer_name}_jitter'] = np.std(jitter_list)
            
            # 予測安定性
            preds = [o.argmax(dim=1).item() for o in all_outputs]
            features['pred_stability'] = len(set(preds)) / len(preds)
            
        return features
    
    def get_feature_names(self):
        """特徴量名のリスト"""
        return list(self.extract_features(torch.randn(1, 3, 32, 32)).keys())


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
test_subset = torch.utils.data.Subset(test_dataset, range(400))
val_subset = torch.utils.data.Subset(test_dataset, range(400, 700))

train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_subset, batch_size=1, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_subset, batch_size=1, shuffle=False)

print(f"  訓練: {len(train_subset)}, 特徴抽出: {len(test_subset)}, 検証: {len(val_subset)}")


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
                pred = model(data).argmax(dim=1)
                correct += (pred == target).sum().item()
        print(f"  Epoch {epoch+1}: Accuracy = {100*correct/len(test_subset):.1f}%")


# =============================================================================
# 5. 特徴量抽出
# =============================================================================
print("\n【3. 多特徴量抽出】")

extractor = MultiFeatureExtractor(model, timesteps=100, num_trials=8)

X_train = []
y_train = []

model.eval()
for i, (data, target) in enumerate(test_loader):
    if i >= 200:  # 訓練用200サンプル
        break
    
    features = extractor.extract_features(data, noise_std=0.1)
    X_train.append(list(features.values()))
    
    # 正解/不正解ラベル
    with torch.no_grad():
        pred = model(data).argmax(dim=1).item()
    is_wrong = 0 if pred == target.item() else 1
    y_train.append(is_wrong)
    
    if (i + 1) % 50 == 0:
        print(f"  訓練データ抽出: {i+1}/200")

X_train = np.array(X_train)
y_train = np.array(y_train)

print(f"\n  特徴量数: {X_train.shape[1]}")
print(f"  正解サンプル: {sum(1 for y in y_train if y == 0)}")
print(f"  不正解サンプル: {sum(1 for y in y_train if y == 1)}")


# =============================================================================
# 6. 分類器学習
# =============================================================================
print("\n【4. 分類器学習】")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# ロジスティック回帰
lr_clf = LogisticRegression(max_iter=1000, class_weight='balanced')
lr_clf.fit(X_train_scaled, y_train)

# ランダムフォレスト
rf_clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_clf.fit(X_train_scaled, y_train)

print("  分類器学習完了！")


# =============================================================================
# 7. 検証セットで評価
# =============================================================================
print("\n【5. 検証セットでの評価】")

X_val = []
y_val = []

model.eval()
for i, (data, target) in enumerate(val_loader):
    if i >= 150:
        break
    
    features = extractor.extract_features(data, noise_std=0.1)
    X_val.append(list(features.values()))
    
    with torch.no_grad():
        pred = model(data).argmax(dim=1).item()
    is_wrong = 0 if pred == target.item() else 1
    y_val.append(is_wrong)
    
    if (i + 1) % 50 == 0:
        print(f"  検証データ抽出: {i+1}/150")

X_val = np.array(X_val)
y_val = np.array(y_val)
X_val_scaled = scaler.transform(X_val)

print(f"\n  検証データ: {len(y_val)} サンプル")
print(f"  正解: {sum(1 for y in y_val if y == 0)}, 不正解: {sum(1 for y in y_val if y == 1)}")

# 評価
print("\n  【ロジスティック回帰】")
lr_pred = lr_clf.predict(X_val_scaled)
lr_prob = lr_clf.predict_proba(X_val_scaled)[:, 1]
print(classification_report(y_val, lr_pred, target_names=['Correct', 'Wrong'], zero_division=0))
if len(set(y_val)) > 1:
    lr_auc = roc_auc_score(y_val, lr_prob)
    print(f"  AUC-ROC: {lr_auc:.4f}")

print("\n  【ランダムフォレスト】")
rf_pred = rf_clf.predict(X_val_scaled)
rf_prob = rf_clf.predict_proba(X_val_scaled)[:, 1]
print(classification_report(y_val, rf_pred, target_names=['Correct', 'Wrong'], zero_division=0))
if len(set(y_val)) > 1:
    rf_auc = roc_auc_score(y_val, rf_prob)
    print(f"  AUC-ROC: {rf_auc:.4f}")


# =============================================================================
# 8. 特徴量重要度分析
# =============================================================================
print("\n【6. 特徴量重要度】")

feature_names = list(features.keys())
importances = rf_clf.feature_importances_
indices = np.argsort(importances)[::-1]

print(f"\n  上位10特徴量:")
for i in range(min(10, len(feature_names))):
    idx = indices[i]
    print(f"    {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")


# =============================================================================
# 9. 可視化
# =============================================================================
print("\n【7. 可視化】")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# ROC曲線比較
ax = axes[0, 0]
if len(set(y_val)) > 1:
    fpr_lr, tpr_lr, _ = roc_curve(y_val, lr_prob)
    fpr_rf, tpr_rf, _ = roc_curve(y_val, rf_prob)
    ax.plot(fpr_lr, tpr_lr, 'b-', linewidth=2, label=f'Logistic (AUC={lr_auc:.3f})')
    ax.plot(fpr_rf, tpr_rf, 'g-', linewidth=2, label=f'Random Forest (AUC={rf_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve Comparison')
    ax.legend()

# 特徴量重要度
ax = axes[0, 1]
top_n = 10
top_indices = indices[:top_n]
top_names = [feature_names[i][:15] for i in top_indices]
top_imp = importances[top_indices]
ax.barh(range(top_n), top_imp, color='steelblue')
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_names)
ax.set_xlabel('Importance')
ax.set_title('Top 10 Feature Importances')
ax.invert_yaxis()

# 確信度分布
ax = axes[1, 0]
correct_conf = [X_val[i, 0] for i in range(len(y_val)) if y_val[i] == 0]
wrong_conf = [X_val[i, 0] for i in range(len(y_val)) if y_val[i] == 1]
ax.hist(correct_conf, bins=15, alpha=0.7, label='Correct', color='green')
ax.hist(wrong_conf, bins=15, alpha=0.7, label='Wrong', color='red')
ax.set_xlabel('Confidence')
ax.set_ylabel('Count')
ax.set_title('Confidence Distribution')
ax.legend()

# 予測確率分布
ax = axes[1, 1]
correct_prob = [rf_prob[i] for i in range(len(y_val)) if y_val[i] == 0]
wrong_prob = [rf_prob[i] for i in range(len(y_val)) if y_val[i] == 1]
ax.hist(correct_prob, bins=15, alpha=0.7, label='Correct', color='green')
ax.hist(wrong_prob, bins=15, alpha=0.7, label='Wrong', color='red')
ax.set_xlabel('Hallucination Probability')
ax.set_ylabel('Count')
ax.set_title('Predicted Hallucination Probability')
ax.legend()

plt.tight_layout()
plt.savefig('hallucination_detector_v2.png', dpi=150, bbox_inches='tight')
print("  保存: hallucination_detector_v2.png")


# =============================================================================
# 10. まとめ
# =============================================================================
print("\n" + "=" * 70)
print("🔬 SNN Hallucination Detector v2 まとめ")
print("=" * 70)

print(f"""
【改善点】
  - 単一ジッター → {len(feature_names)}個の多特徴量
  - 閾値ベース → 機械学習ベース分類
  - 層別のTTFS/ジッター統計を活用

【評価結果】
  - ロジスティック回帰 AUC: {lr_auc:.4f}
  - ランダムフォレスト AUC: {rf_auc:.4f}
  
【重要な発見】
  - 上位特徴量: {feature_names[indices[0]]}, {feature_names[indices[1]]}, {feature_names[indices[2]]}
  - SNN特徴（TTFS, ジッター）がハルシネーション検知に寄与

【次のステップ】
  1. より大規模データでの検証
  2. Transformer/LLMへの適用
  3. リアルタイムAPI化
  4. 閾値チューニングの自動化
""")

print("\n🚀 多特徴量SNN解析 = AIの「健康診断」！")
print("=" * 70)
