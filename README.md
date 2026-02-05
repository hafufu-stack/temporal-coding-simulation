# Temporal Coding Simulation
# 時間コーディングシミュレーション

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> 🧠 スパイキングニューラルネットワーク (SNN) の研究・実験コードリポジトリ

## 📁 Repository Structure

```
temporal-coding-simulation/
├── ann-to-snn-converter/      # 🆕 ANN→SNN変換 + AI解釈可能性 + SNNガードレール
│   ├── experiments/           # 実験コード (TTFS, ハルシネーション検知, 脱獄検知)
│   ├── api/                   # リアルタイム検知API
│   └── README.md              # 詳細ドキュメント
├── snn-comprypto/             # 🔐 SNN暗号化（カオスリザバー）
├── snn-compression/           # 📦 SNN圧縮（相関符号化）
├── snn-genai/                 # 🎨 SNN画像生成
├── snn-operation/             # ➕ SNN算術演算
├── 10-neuron-memory/          # 💾 10ニューロンメモリ
└── assets/                    # 📊 図・画像
```

## 🔥 Featured Projects

### 1. [ANN-to-SNN Converter](./ann-to-snn-converter/) 🛡️
**AI Interpretability & SNN Guardrail (v4)**

- **Universal Threshold Formula**: $\theta = 2.0 \times \max(\text{activation})$
- **TTFS Analysis**: 思考優先順位の可視化
- **Hallucination Detection**: AUC 0.75達成
- **🆕 SNN Guardrail**: 脱獄攻撃100%検知！

| 実験 | 結果 |
|------|------|
| ANN-SNN変換 | 100%精度維持 |
| GPT-2 TTFS | +3.1差（無意味入力検知）|
| TinyLlama TTFS | +4.2差（スケーリング則発見）|
| ハルシネーション検知 | AUC 0.75 |
| **脱獄検知** | **100% (8/8攻撃)** |

### 2. [SNN-Comprypto](./snn-comprypto/)
**SNN暗号化（Simultaneous Encryption）**

- カオスリザバーダイナミクス
- 高セキュリティ暗号化
- スパイクベース認証

### 3. [SNN-Compression](./snn-compression/)
**SNN圧縮（相関符号化）**

- 差分符号化 + zlib
- バイナリデータ91%改善
- 相関符号化（12.4ビット/ニューロン）

### 4. [SNN-GenAI](./snn-genai/)
**Image Generation with SNNs**

- Spiking VAE
- 70/30 Hybrid Readout
- エネルギー効率的な画像生成

### 5. [SNN-Operation](./snn-operation/)
**Neural Arithmetic**

- 30ニューロン加算器
- スパイクベース演算

## 📊 Key Results

| プロジェクト | 主要結果 |
|--------------|----------|
| ANN-SNN変換 | $\alpha = 2.0$ で100%精度維持 |
| ハルシネーション検知 | AUC 0.75 (Ensemble + Auto-Threshold) |
| **脱獄検知** | **100% (TTFS +10〜19σ偏差)** |
| SNN-Comprypto | カオス暗号化 |
| SNN-Compression | 91%圧縮改善（バイナリ）|
| Spiking VAE | 96%スパイク率、30%膜電位貢献 |

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/hafufu-stack/temporal-coding-simulation.git
cd temporal-coding-simulation

# ANN-SNN変換（ハルシネーション検知）
cd ann-to-snn-converter
python experiments/hallucination_detector_v3.py

# 脱獄検知（SNN Guardrail）
python experiments/jailbreak_detection.py

# SNN暗号化
cd ../snn-comprypto
python snn_comprypto.py

# SNN圧縮
cd ../snn-compression
python correlation_compressor.py
```

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
