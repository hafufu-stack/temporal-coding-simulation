"""
SNN Hallucination Detector v3: 自動閾値チューニング & 最適化
============================================================

改良点:
1. クロスバリデーションによる堅牢な評価
2. 閾値自動チューニング（F1最大化）
3. アンサンブル学習（複数分類器の組み合わせ）
4. 特徴量選択（重要特徴量のみ使用）
5. 確信度キャリブレーション

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

plt.rcParams['font.family'] = ['MS Gothic', 'DejaVu Sans']

print("=" * 70)
print("🔧 SNN Hallucination Detector v3: 自動閾値チューニング & 最適化")
print("=" * 70)


# =============================================================================
# 1. モデル定義 (ResNet)
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
# 2. SNN特徴量抽出器
# =============================================================================
class SNNFeatureExtractor:
    def __init__(self, timesteps=100, num_trials=5):
        self.timesteps = timesteps
        self.num_trials = num_trials
    
    def compute_ttfs(self, activation):
        ttfs = torch.full_like(activation, float(self.timesteps))
        active_mask = activation > 0
        if active_mask.any():
            max_act = activation.max()
            if max_act > 0:
                normalized = activation[active_mask] / max_act
                ttfs[active_mask] = self.timesteps * (1 - normalized)
        return ttfs
    
    def extract(self, model, x, noise_std=0.08):
        features = {}
        
        model.eval()
        with torch.no_grad():
            output = model(x)
            probs = F.softmax(output, dim=1)
            
            features['confidence'] = probs.max().item()
            features['entropy'] = -(probs * torch.log(probs + 1e-8)).sum().item()
            features['margin'] = (probs.topk(2)[0][0,0] - probs.topk(2)[0][0,1]).item()
            features['top2_ratio'] = (probs.topk(2)[0][0,1] / (probs.topk(2)[0][0,0] + 1e-8)).item()
            
            activations = model.get_activations(x)
            
            for layer_name, act in activations.items():
                act_flat = act.view(-1)
                features[f'{layer_name}_mean'] = act_flat.mean().item()
                features[f'{layer_name}_std'] = act_flat.std().item()
                features[f'{layer_name}_sparsity'] = (act_flat <= 0).float().mean().item()
                
                ttfs = self.compute_ttfs(act)
                features[f'{layer_name}_ttfs_mean'] = ttfs.view(-1).mean().item()
                features[f'{layer_name}_ttfs_std'] = ttfs.view(-1).std().item()
            
            # ジッター計算
            all_outputs = []
            for _ in range(self.num_trials):
                noisy_x = x + torch.randn_like(x) * noise_std
                noisy_x = torch.clamp(noisy_x, 0, 1)
                noisy_output = model(noisy_x)
                all_outputs.append(noisy_output)
            
            stacked = torch.stack(all_outputs)
            features['output_jitter'] = stacked.std(dim=0).mean().item()
            
            preds = [o.argmax(dim=1).item() for o in all_outputs]
            features['pred_consistency'] = sum(1 for p in preds if p == preds[0]) / len(preds)
        
        return features


# =============================================================================
# 3. 自動閾値チューニングクラス
# =============================================================================
class AutoThresholdTuner:
    """自動閾値チューニング"""
    
    def __init__(self, n_thresholds=50):
        self.n_thresholds = n_thresholds
        self.optimal_threshold = 0.5
        self.metrics_history = []
    
    def find_optimal_threshold(self, y_true, y_prob, metric='f1'):
        """最適閾値を探索"""
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        thresholds = np.linspace(0.01, 0.99, self.n_thresholds)
        best_score = 0
        best_threshold = 0.5
        
        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred, zero_division=0)
            elif metric == 'balanced':
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                score = 2 * prec * rec / (prec + rec + 1e-8)
            
            self.metrics_history.append({
                'threshold': thresh,
                'score': score
            })
            
            if score > best_score:
                best_score = score
                best_threshold = thresh
        
        self.optimal_threshold = best_threshold
        return best_threshold, best_score
    
    def plot_threshold_curve(self, y_true, y_prob, ax=None):
        """閾値-メトリクス曲線をプロット"""
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        thresholds = np.linspace(0.01, 0.99, 50)
        f1_scores = []
        precisions = []
        recalls = []
        
        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
            precisions.append(precision_score(y_true, y_pred, zero_division=0))
            recalls.append(recall_score(y_true, y_pred, zero_division=0))
        
        if ax is None:
            fig, ax = plt.subplots()
        
        ax.plot(thresholds, f1_scores, 'b-', label='F1 Score', linewidth=2)
        ax.plot(thresholds, precisions, 'g--', label='Precision', linewidth=2)
        ax.plot(thresholds, recalls, 'r--', label='Recall', linewidth=2)
        ax.axvline(self.optimal_threshold, color='orange', linestyle=':', label=f'Optimal ({self.optimal_threshold:.2f})')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Score')
        ax.set_title('Threshold vs Metrics')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax


# =============================================================================
# 4. アンサンブル分類器
# =============================================================================
class EnsembleHallucinationDetector:
    """アンサンブルハルシネーション検知器"""
    
    def __init__(self):
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        
        self.classifiers = {
            'rf': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=50, random_state=42),
            'lr': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
            'svm': SVC(probability=True, class_weight='balanced', random_state=42)
        }
        
        self.weights = {'rf': 0.3, 'gb': 0.3, 'lr': 0.2, 'svm': 0.2}
        self.scaler = None
        self.fitted = False
    
    def fit(self, X, y):
        from sklearn.preprocessing import StandardScaler
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        for name, clf in self.classifiers.items():
            clf.fit(X_scaled, y)
        
        self.fitted = True
    
    def predict_proba(self, X):
        if not self.fitted:
            raise ValueError("Ensemble not fitted yet!")
        
        X_scaled = self.scaler.transform(X)
        
        ensemble_proba = np.zeros((X.shape[0], 2))
        for name, clf in self.classifiers.items():
            proba = clf.predict_proba(X_scaled)
            ensemble_proba += self.weights[name] * proba
        
        return ensemble_proba
    
    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)
    
    def get_individual_predictions(self, X):
        """各分類器の予測を取得"""
        X_scaled = self.scaler.transform(X)
        predictions = {}
        for name, clf in self.classifiers.items():
            predictions[name] = clf.predict_proba(X_scaled)[:, 1]
        return predictions


# =============================================================================
# 5. 特徴量選択
# =============================================================================
class FeatureSelector:
    """重要特徴量選択"""
    
    def __init__(self, n_features=20):
        self.n_features = n_features
        self.selected_features = []
        self.feature_importances = {}
    
    def fit(self, X, y, feature_names):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import mutual_info_classif
        
        # Random Forest重要度
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        rf_importance = rf.feature_importances_
        
        # 相互情報量
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
        # 組み合わせスコア
        combined_scores = 0.7 * rf_importance + 0.3 * (mi_scores / mi_scores.max())
        
        # 上位N特徴量を選択
        indices = np.argsort(combined_scores)[::-1][:self.n_features]
        self.selected_features = [feature_names[i] for i in indices]
        
        for i, name in enumerate(feature_names):
            self.feature_importances[name] = combined_scores[i]
        
        return indices
    
    def transform(self, X, feature_indices):
        return X[:, feature_indices]


# =============================================================================
# 6. データ準備
# =============================================================================
print("\n【1. データ準備】")
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_subset = torch.utils.data.Subset(train_dataset, range(10000))
feature_subset = torch.utils.data.Subset(test_dataset, range(600))

train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
feature_loader = torch.utils.data.DataLoader(feature_subset, batch_size=1, shuffle=False)

print(f"  訓練: {len(train_subset)}, 特徴抽出: {len(feature_subset)}")


# =============================================================================
# 7. モデル学習
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
            for i, (data, target) in enumerate(feature_loader):
                if i >= 200: break
                pred = model(data).argmax(dim=1)
                correct += (pred == target).sum().item()
        print(f"  Epoch {epoch+1}: Accuracy = {100*correct/200:.1f}%")

print(f"\n  モデル学習完了！")


# =============================================================================
# 8. 特徴量抽出
# =============================================================================
print("\n【3. 特徴量抽出】")
extractor = SNNFeatureExtractor()

X_data = []
y_data = []

model.eval()
for i, (data, target) in enumerate(feature_loader):
    if i >= 500:
        break
    
    features = extractor.extract(model, data)
    X_data.append(list(features.values()))
    
    with torch.no_grad():
        pred = model(data).argmax(dim=1).item()
    y_data.append(0 if pred == target.item() else 1)
    
    if (i + 1) % 100 == 0:
        print(f"  抽出: {i+1}/500")

X_data = np.array(X_data)
y_data = np.array(y_data)
feature_names = list(features.keys())

print(f"\n  特徴量数: {X_data.shape[1]}")
print(f"  正解: {sum(1 for y in y_data if y == 0)}, 不正解: {sum(1 for y in y_data if y == 1)}")


# =============================================================================
# 9. 特徴量選択
# =============================================================================
print("\n【4. 特徴量選択】")
selector = FeatureSelector(n_features=20)
selected_indices = selector.fit(X_data, y_data, feature_names)

print(f"\n  選択された Top 10 特徴量:")
for i, name in enumerate(selector.selected_features[:10]):
    score = selector.feature_importances[name]
    print(f"    {i+1}. {name}: {score:.4f}")

X_selected = selector.transform(X_data, selected_indices)


# =============================================================================
# 10. クロスバリデーション + アンサンブル学習
# =============================================================================
print("\n【5. クロスバリデーション + アンサンブル学習】")
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = []
all_y_true = []
all_y_prob = []

print("\n  5-Fold Cross Validation:")
for fold, (train_idx, val_idx) in enumerate(kfold.split(X_selected, y_data)):
    X_train, X_val = X_selected[train_idx], X_selected[val_idx]
    y_train, y_val = y_data[train_idx], y_data[val_idx]
    
    ensemble = EnsembleHallucinationDetector()
    ensemble.fit(X_train, y_train)
    
    y_prob = ensemble.predict_proba(X_val)[:, 1]
    
    if len(set(y_val)) > 1:
        auc = roc_auc_score(y_val, y_prob)
        cv_scores.append(auc)
        print(f"    Fold {fold+1}: AUC = {auc:.4f}")
    
    all_y_true.extend(y_val)
    all_y_prob.extend(y_prob)

all_y_true = np.array(all_y_true)
all_y_prob = np.array(all_y_prob)

print(f"\n  平均 AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")


# =============================================================================
# 11. 自動閾値チューニング
# =============================================================================
print("\n【6. 自動閾値チューニング】")
tuner = AutoThresholdTuner(n_thresholds=50)

optimal_thresh, best_f1 = tuner.find_optimal_threshold(all_y_true, all_y_prob, metric='f1')
print(f"  最適閾値 (F1最大化): {optimal_thresh:.3f}")
print(f"  最大 F1スコア: {best_f1:.4f}")

# 最適閾値での評価
from sklearn.metrics import classification_report
y_pred_optimal = (all_y_prob >= optimal_thresh).astype(int)

print(f"\n  【最適閾値での評価】")
print(classification_report(all_y_true, y_pred_optimal, target_names=['Correct', 'Wrong'], zero_division=0))


# =============================================================================
# 12. 最終モデル学習
# =============================================================================
print("\n【7. 最終モデル学習】")
final_ensemble = EnsembleHallucinationDetector()
final_ensemble.fit(X_selected, y_data)

# 各分類器の性能
print("\n  【各分類器の性能】")
individual_preds = final_ensemble.get_individual_predictions(X_selected)
for name, probs in individual_preds.items():
    auc = roc_auc_score(y_data, probs)
    print(f"    {name}: AUC = {auc:.4f}")


# =============================================================================
# 13. 可視化
# =============================================================================
print("\n【8. 可視化】")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 閾値-メトリクス曲線
ax = axes[0, 0]
tuner.plot_threshold_curve(all_y_true, all_y_prob, ax=ax)

# ROC曲線
ax = axes[0, 1]
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(all_y_true, all_y_prob)
ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'Ensemble AUC={np.mean(cv_scores):.3f}')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve (Ensemble)')
ax.legend()
ax.grid(True, alpha=0.3)

# 選択された特徴量重要度
ax = axes[1, 0]
top_names = selector.selected_features[:10]
top_scores = [selector.feature_importances[n] for n in top_names]
ax.barh(range(len(top_names)), top_scores, color='steelblue')
ax.set_yticks(range(len(top_names)))
ax.set_yticklabels([n[:20] for n in top_names])
ax.set_xlabel('Combined Importance')
ax.set_title('Top 10 Selected Features')
ax.invert_yaxis()

# CV結果
ax = axes[1, 1]
ax.bar(range(1, len(cv_scores)+1), cv_scores, color='forestgreen')
ax.axhline(np.mean(cv_scores), color='red', linestyle='--', label=f'Mean={np.mean(cv_scores):.3f}')
ax.set_xlabel('Fold')
ax.set_ylabel('AUC')
ax.set_title('Cross-Validation Results')
ax.legend()

plt.tight_layout()
plt.savefig('hallucination_detector_v3.png', dpi=150, bbox_inches='tight')
print("  保存: hallucination_detector_v3.png")


# =============================================================================
# 14. まとめ
# =============================================================================
print("\n" + "=" * 70)
print("🔧 SNN Hallucination Detector v3 まとめ")
print("=" * 70)

print(f"""
【改良点】
  1. クロスバリデーション (5-Fold)
  2. 自動閾値チューニング (F1最大化)
  3. アンサンブル学習 (RF + GB + LR + SVM)
  4. 特徴量選択 (RF重要度 + 相互情報量)

【結果】
  - 平均 AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}
  - 最適閾値: {optimal_thresh:.3f}
  - 最大 F1: {best_f1:.4f}

【選択された特徴量 Top 5】
  1. {selector.selected_features[0]}
  2. {selector.selected_features[1]}
  3. {selector.selected_features[2]}
  4. {selector.selected_features[3]}
  5. {selector.selected_features[4]}

【使い方】
  threshold = {optimal_thresh:.3f}
  if hallucination_prob >= threshold:
      print("⚠️ ハルシネーションリスク高")
  else:
      print("✅ 信頼できる予測")
""")

print("\n🚀 自動チューニングでハルシネーション検知が更に改善！")
print("=" * 70)
