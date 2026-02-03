"""
SNN Hallucination Detection API
================================

FastAPIベースのリアルタイムハルシネーション検知API。

エンドポイント:
- POST /analyze: 画像を解析してハルシネーション確率を返す
- GET /health: ヘルスチェック

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
from typing import Dict, Any, Optional
import base64
from io import BytesIO
from PIL import Image
import json


# =============================================================================
# 1. モデル定義（ResNet）
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
# 2. SNN Feature Extractor
# =============================================================================
class SNNFeatureExtractor:
    """SNN特徴量抽出"""
    
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
            features['prediction'] = output.argmax(dim=1).item()
            
            activations = model.get_activations(x)
            
            for layer_name, act in activations.items():
                act_flat = act.view(-1)
                features[f'{layer_name}_mean'] = act_flat.mean().item()
                features[f'{layer_name}_std'] = act_flat.std().item()
                
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
            features['pred_stability'] = 1.0 if len(set(preds)) == 1 else len(set(preds)) / len(preds)
        
        return features


# =============================================================================
# 3. Hallucination Detector
# =============================================================================
class HallucinationDetector:
    """ハルシネーション検知器"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.extractor = SNNFeatureExtractor()
        self.class_names = ['airplane', 'car', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck']
        
        # 簡易判定ルール（実際はMLモデルを使用）
        self.confidence_threshold = 0.7
        self.jitter_threshold = 0.3
    
    def analyze(self, image_tensor: torch.Tensor) -> Dict[str, Any]:
        """画像を解析してハルシネーション確率を計算"""
        
        features = self.extractor.extract(self.model, image_tensor)
        
        # リスクスコア計算
        confidence = features['confidence']
        jitter = features['output_jitter']
        stability = features['pred_stability']
        entropy = features['entropy']
        
        # ハルシネーションリスク計算
        # 高エントロピー + 高ジッター + 低安定性 = 高リスク
        risk_score = (
            (entropy / 2.3) * 0.3 +  # entropyを正規化
            (jitter / 0.5) * 0.3 +   # jitterを正規化
            (1 - stability) * 0.2 +
            (1 - confidence) * 0.2
        )
        risk_score = min(max(risk_score, 0), 1)  # 0-1にクリップ
        
        # 判定
        is_reliable = risk_score < 0.4
        
        prediction = features['prediction']
        
        return {
            'prediction': {
                'class_id': prediction,
                'class_name': self.class_names[prediction],
                'confidence': round(confidence, 4)
            },
            'snn_analysis': {
                'output_jitter': round(jitter, 4),
                'prediction_stability': round(stability, 4),
                'entropy': round(entropy, 4),
                'layer3_ttfs_mean': round(features.get('layer3_ttfs_mean', 0), 4),
                'layer3_ttfs_std': round(features.get('layer3_ttfs_std', 0), 4)
            },
            'hallucination_risk': {
                'score': round(risk_score, 4),
                'is_reliable': is_reliable,
                'recommendation': 'Trust this prediction' if is_reliable else 'Verify this prediction manually'
            }
        }


# =============================================================================
# 4. API Server
# =============================================================================
def create_app():
    """FastAPIアプリ作成"""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
    except ImportError:
        print("FastAPIがインストールされていません。pip install fastapi uvicorn")
        return None
    
    app = FastAPI(
        title="SNN Hallucination Detection API",
        description="SNNベースのAIハルシネーション検知API",
        version="1.0.0"
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # モデル初期化
    model = SmallResNet(num_classes=10)
    model.eval()
    detector = HallucinationDetector(model)
    
    class ImageRequest(BaseModel):
        image_base64: str
        
    class AnalysisResponse(BaseModel):
        prediction: dict
        snn_analysis: dict
        hallucination_risk: dict
    
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "model": "SmallResNet", "api_version": "1.0.0"}
    
    @app.post("/analyze", response_model=AnalysisResponse)
    async def analyze_image(request: ImageRequest):
        try:
            # Base64デコード
            image_data = base64.b64decode(request.image_base64)
            image = Image.open(BytesIO(image_data)).convert('RGB')
            image = image.resize((32, 32))
            
            # Tensor変換
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            image_tensor = transform(image).unsqueeze(0)
            
            # 解析
            result = detector.analyze(image_tensor)
            return result
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.get("/")
    async def root():
        return {
            "message": "SNN Hallucination Detection API",
            "endpoints": {
                "/health": "ヘルスチェック",
                "/analyze": "画像解析 (POST, image_base64)"
            }
        }
    
    return app


# =============================================================================
# 5. Demo without server
# =============================================================================
def demo_analysis():
    """サーバーなしでのデモ"""
    print("=" * 60)
    print("🔍 SNN Hallucination Detection API - Demo Mode")
    print("=" * 60)
    
    # モデル作成
    model = SmallResNet(num_classes=10)
    model.eval()
    detector = HallucinationDetector(model)
    
    # ダミー画像で解析
    dummy_image = torch.randn(1, 3, 32, 32)
    
    print("\n【ダミー画像解析】")
    result = detector.analyze(dummy_image)
    
    print(f"\n  予測:")
    print(f"    クラス: {result['prediction']['class_name']} (ID: {result['prediction']['class_id']})")
    print(f"    確信度: {result['prediction']['confidence']:.4f}")
    
    print(f"\n  SNN解析:")
    for key, value in result['snn_analysis'].items():
        print(f"    {key}: {value}")
    
    print(f"\n  ハルシネーションリスク:")
    print(f"    スコア: {result['hallucination_risk']['score']:.4f}")
    print(f"    信頼性: {'✅ 信頼できる' if result['hallucination_risk']['is_reliable'] else '⚠️ 要確認'}")
    print(f"    推奨: {result['hallucination_risk']['recommendation']}")
    
    print("\n" + "=" * 60)
    print("📝 APIの起動方法:")
    print("  uvicorn hallucination_api:app --reload --host 0.0.0.0 --port 8000")
    print("=" * 60)
    
    return result


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        # サーバーモード
        app = create_app()
        if app:
            import uvicorn
            uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        # デモモード
        demo_analysis()
        
        # FastAPIアプリをエクスポート用に作成
        app = create_app()
