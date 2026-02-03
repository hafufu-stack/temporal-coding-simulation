# Temporal Coding Simulation
# 時間コーディングシミュレーション

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> 🧠 スパイキングニューラルネットワーク (SNN) の研究・実験コードリポジトリ

## 📁 Repository Structure

```
temporal-coding-simulation/
├── ann-to-snn-converter/      # 🆕 ANN→SNN変換 + AI解釈可能性
│   ├── experiments/           # 実験コード (TTFS, Synchrony, ハルシネーション検知)
│   ├── api/                   # リアルタイム検知API
│   └── README.md              # 詳細ドキュメント
├── snn-comprypto/             # 🔐 SNN暗号＋圧縮
├── snn-compression/           # 📦 SNN圧縮
├── snn-genai/                 # 🎨 SNN画像生成
├── snn-operation/             # ➕ SNN算術演算
├── 10-neuron-memory/          # 💾 10ニューロンメモリ
└── assets/                    # 📊 図・画像
```

## 🔥 Featured Projects

### 1. [ANN-to-SNN Converter](./ann-to-snn-converter/)
**AI Interpretability & Hallucination Detection**

- **Universal Threshold Formula**: $\theta = 2.0 \times \max(\text{activation})$
- **TTFS Analysis**: 思考優先順位の可視化
- **Hallucination Detection**: AUC 0.75達成
- **GPT-2/ViT解析**: Transformer/LLMにも対応

| 実験 | 結果 |
|------|------|
| ANN-SNN変換 | 100%精度維持 |
| GPT-2 TTFS | +3.1差（無意味入力検知）|
| ハルシネーション検知 | AUC 0.75 |

### 2. [SNN-Comprypto](./snn-comprypto/)
**Simultaneous Compression & Encryption**

- カオスリザバーダイナミクス
- 高圧縮率（57%の損失削減）
- Spike-only Posterior Collapse解決

### 3. [SNN-GenAI](./snn-genai/)
**Image Generation with SNNs**

- Spiking VAE
- 70/30 Hybrid Readout
- エネルギー効率的な画像生成

### 4. [SNN-Operation](./snn-operation/)
**Neural Arithmetic**

- 30ニューロン加算器
- スパイクベース演算
- 基本ALU操作

## 📊 Key Results

| プロジェクト | 主要結果 |
|--------------|----------|
| ANN-SNN変換 | $\alpha = 2.0$ で100%精度維持 |
| ハルシネーション検知 | AUC 0.75 (Ensemble + Auto-Threshold) |
| SNN-Comprypto | 57%損失削減、KL>0達成 |
| Spiking VAE | 96%スパイク率、30%膜電位貢献 |

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/hafufu-stack/temporal-coding-simulation.git
cd temporal-coding-simulation

# ANN-SNN変換（ハルシネーション検知）
cd ann-to-snn-converter
python experiments/hallucination_detector_v3.py

# SNN圧縮
cd snn-comprypto
python snn_comprypto.py
```

## 📝 Publications

| タイトル | プラットフォーム | 状態 |
|----------|------------------|------|
| Activation-Scaled ANN-to-SNN Conversion with AI Interpretability | Zenodo/arXiv | v8準備中 |
| SNN-Comprypto: Simultaneous Compression and Encryption | Zenodo | 公開済み |
| Hybrid Spiking Neural Networks | Zenodo | 公開済み |
| Von Neumann vs Brain-like Architecture | Zenodo | 公開済み |

## 🤝 Author

**Hiroto Funasaki (ろーる)**
- ORCID: 0009-0004-2517-0177
- Email: cell-activation@ymail.ne.jp
- GitHub: [@hafufu-stack](https://github.com/hafufu-stack)
- Zenodo: [Publications](https://zenodo.org/search?q=metadata.creators.person_or_org.name%3A%22Funasaki%2C%20Hiroto%22)

## 📜 License

MIT License

## 🙏 Acknowledgments

- PyTorch & TorchVision
- HuggingFace Transformers
- Neuromorphic Computing Community
