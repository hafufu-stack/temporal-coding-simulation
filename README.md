# Temporal Coding Simulation

## 背景

脳の神経細胞はスパイク情報が入る順序やタイミングを調節することで、扱える情報量を増やしているのではないかと修論で考察しました。
このリポジトリでは、それをSNN（Spiking Neural Network）で検証していきたいと思います。

因みに私の修論は海馬歯状回の顆粒細胞のシミュレーションでした。
この細胞には、空間情報を運ぶMD入力と非空間情報を運ぶLD入力があり、それらは相互作用があるようです。

## 開発の方針や予定

- [x] **Phase 1: 記憶**
    - 10個のニューロンで約111ビットの保持に成功
- [x] **Phase 2: 演算**
    - 30個のニューロン（入力20＋演算10）を使用し、111ビット同士の加算に成功
- [x] **Phase 3: 暗号化** ← snn-comprypto
    - NIST SP 800-22 乱数検定 9/9 合格
    - 温度パラメータが第2の暗号鍵として機能
    - SNN vs DNN vs LSTMで乱数品質を検証
- [x] **Phase 4: 高性能圧縮 (v5 NEW!)** ← snn-compression
    - 適応圧縮エンジン（Delta/XOR/Raw自動選択）
    - **バイナリで2.9%圧縮率を達成**（zlibの104%を上回る）
- [ ] **Phase 5: 今後の展望**
    - GPU (CUDA) 対応
    - `pip install snn-comprypto` としてリリース

## ディレクトリ構成

```
temporal-coding-simulation/
├── 10-neuron-memory/     # 記憶実験（Phase 1）
├── snn-operation/        # 演算実験（Phase 2）
├── snn-comprypto/        # 暗号化システム（Phase 3）
├── snn-compression/      # 圧縮システム（Phase 4, v5 NEW!）
└── snn-genai/            # 生成AI実験（開発中）
```

---

## 1. 10-neuron-memory（記憶実験）

10個のニューロンを用い、最も効率良く情報を記憶させる方法を検証しました。

| 符号化方式 | 時間分解能 | 情報量 |
|-----------|-----------|--------|
| 独立符号化 | 0.1ms | 約82ビット |
| **相関符号化** | 0.05ms | **約111ビット** |

**結論**: 1つのニューロンを基準として「相対的なズレ」で情報を表現する方が高効率。

---

## 2. snn-operation（演算実験）

30個のニューロン（入力20 + 演算10）で111ビット同士の加算を実現。

```bash
cd snn-operation
python 30-neuron-adder.py
```

---

## 3. snn-comprypto（暗号化システム）

SNNのカオス的ダイナミクスを利用した暗号化システム。

### 主な成果

| テスト | 結果 |
|--------|------|
| NIST SP 800-22 乱数検定 | ✅ 9/9 合格 |
| 温度パラメータ（0.0001差） | 復号不能 |
| SNN vs DNN vs LSTM | SNN最優秀 (0.39%) |

### 使い方

```python
from core.comprypto_system import SNNCompryptor

encryptor = SNNCompryptor(key_seed=12345, temperature=1.0)
encrypted, _ = encryptor.compress_encrypt(data)
```

詳細は [snn-comprypto/README.md](snn-comprypto/README.md) を参照。

---

## 4. snn-compression（v5 適応圧縮システム）🆕

SNNの予測能力を活用した高性能圧縮 + カオス暗号化の統合システム。

### v5の特徴

- **適応圧縮**: Delta/XOR/Rawから最適な方式を自動選択
- **zlibを超える圧縮率**: バイナリで **2.9%** を達成
- **完全復元**: テキスト、バイナリ、画像で100%復元確認

### 圧縮率ベンチマーク

| データ種類 | SNN-Comprypto v5 | zlib | 勝者 |
|------------|------------------|------|------|
| バイナリ（連番） | **2.9%** | 104.3% | v5 🏆 |
| 日本語テキスト | **8.5%** | 11.3% | v5 🏆 |
| 英語テキスト | 15.6% | 24.4% | v5 🏆 |
| 画像（PNG） | 95.5% | - | - |

### 使い方

```python
from stdp_comprypto import STDPComprypto

# 暗号化
enc = STDPComprypto(key_seed=12345, temperature=1.0)
encrypted = enc.encrypt(data)

# 復号（同じkey_seed + temperatureが必要）
dec = STDPComprypto(key_seed=12345, temperature=1.0)
restored = dec.decrypt(encrypted)
```

### ファイル構成

```
snn-compression/
├── stdp_comprypto.py         # 📦 圧縮+暗号化統合（推奨）
├── stdp_predictive_v6.py     # 圧縮のみ（暗号化なし）
└── stdp_predictive_v1-v5.py  # 開発履歴
```

---

## 動作環境

- Python 3.10+
- NumPy
- Matplotlib
- Numba (高速化版を使う場合)
- Japanize-Matplotlib (日本語フォント表示用)

```bash
pip install numpy matplotlib japanize-matplotlib numba
```

---

## 論文

**SNN-Comprypto: High-Performance Compression and Encryption Using Spiking Neural Network Chaotic Reservoir Dynamics**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18426415.svg)](https://zenodo.org/records/18426415)

---

## 作者

**ろーる**
*   **note**：[https://note.com/cell_activation](https://note.com/cell_activation)
*   **Zenn**：[https://zenn.dev/cell_activation](https://zenn.dev/cell_activation)
*   **GitHub**：[https://github.com/hafufu-stack](https://github.com/hafufu-stack)
