# SNN-Comprypto

**スパイキングニューラルネットワークによる予測圧縮＋カオス暗号化システム**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18265447.svg)](https://doi.org/10.5281/zenodo.18265447)

## 概要

SNN-Compryptoは、脳のニューロンを模倣したスパイキングニューラルネットワーク（SNN）を使用し、**圧縮と暗号化を同時に行う**次世代セキュリティシステムです。

### 特徴

- 🧠 **予測圧縮**: SNNがデータの次の値を予測し、残差だけを記録
- 🔐 **カオス暗号化**: ニューロンの膜電位からカオス鍵を生成
- 🌡️ **温度パラメータ**: 第2の暗号鍵として機能（0.0001の差で復号不能）
- ⚡ **高速**: Numba JITで7.5倍高速化

## 実証された性能

| テスト | 結果 |
|--------|------|
| NIST SP 800-22 乱数検定 | ✅ 9/9 合格 |
| アバランチ効果（鍵1ビット差） | 0.70% 一致率 |
| 温度パラメータ（0.0001差） | 0.40% 一致率 |
| データ整合性 | 100% 復元 |

## インストール

```bash
git clone https://github.com/hafufu-stack/neural-coding-simulation.git
cd neural-coding-simulation/snn-comprypto
pip install numpy matplotlib
```

## 使い方

### 基本的な暗号化・復号

```python
from comprypto_system import SNNCompryptor

# 暗号化（シード値 + 温度が鍵）
encryptor = SNNCompryptor(key_seed=12345, temperature=1.0)
encrypted, _ = encryptor.compress_encrypt(data)

# 復号（同じシード + 同じ温度が必要）
decryptor = SNNCompryptor(key_seed=12345, temperature=1.0)
restored = decryptor.decrypt_decompress(encrypted)
```

### ベンチマーク実行

```bash
# 基本ベンチマーク
python comprypto_system.py

# 温度パラメータのアバランチ効果検証
python benchmarks/thermal_benchmark.py

# ニューロン数の相転移分析
python benchmarks/phase_transition_analysis.py

# NIST乱数検定
python nist_test.py
```

## ファイル構成

```
snn-comprypto/
├── comprypto_system.py      # メインシステム（温度パラメータ対応）
├── comprypto_numba.py       # Numba高速化版
├── nist_test.py             # NIST SP 800-22 乱数検定
├── benchmarks/              # ベンチマークスクリプト
│   ├── thermal_benchmark.py       # 温度パラメータ検証
│   ├── neuron_benchmark.py        # ニューロン数検証
│   └── phase_transition_analysis.py  # 相転移分析
├── results/                 # 実験結果（グラフ等）
│   ├── thermal_avalanche_effect.png
│   ├── neuron_count_benchmark.png
│   └── phase_transition_analysis.png
└── docs/                    # ドキュメント
    ├── 図解_中学生向け.md
    └── SNN暗号化・圧縮の先行研究調査.md
```

## 研究成果

### 発見1: 温度パラメータによるアバランチ効果
温度を0.0001だけ変えると、復号一致率が0.40%（≒ランダム理論値）に低下。
→ 温度が「第2の暗号鍵」として機能することを実証。

### 発見2: ニューロン数の相転移
- **100ニューロンで相転移**: 予測精度が劇的に向上
- **160ニューロンで収穫逓減開始**: 投資対効果が低下
- **240ニューロンが推奨値**: 最高性能の80%を達成

### 発見3: RNG Battle Royale（v3）
SNNは乱数生成において他のニューラルネットワークを圧倒！

| アーキテクチャ | 予測率 | vs SNN |
|--------------|--------|--------|
| **SNN** | **0.390%** | **1.0×** |
| Python random | 0.396% | 1.0× |
| DNN | 7.210% | 18.5× |
| LSTM | 21.954% | 56.3× |

→ **SNNが最も予測困難な乱数を生成！** (100,000ラウンドで検証)

### 発見4: カオス解析（v4 NEW）

| 指標 | SNN | DNN | LSTM |
|------|-----|-----|------|
| エントロピー (max 8.0) | **7.998** | 7.0 | 6.5 |
| 自己相関 | **0.008** | 0.51 | 0.75 |
| リアプノフ指数 | **+26.97** | +23.98 | +22.05 |

→ **SNNは真のカオス系！** (正のリアプノフ = カオス的)

### 発見5: 攻撃耐性（v4 NEW）

| 攻撃タイプ | 結果 | 判定 |
|-----------|------|------|
| 既知平文攻撃 | 2.0%予測 | ✅ SECURE |
| 選択平文攻撃 | χ²=297 | ✅ SECURE |
| 鍵回復攻撃 | 未発見 | ✅ SECURE |
| サイドチャネル攻撃 | CV=0.057 | ✅ SECURE |

→ **4/5の現実的な攻撃に耐性！**

## ベンチマーク実行

```bash
# 基本ベンチマーク
python comprypto_system.py

# 温度パラメータのアバランチ効果検証
python benchmarks/thermal_benchmark.py

# ニューロン数の相転移分析
python benchmarks/phase_transition_analysis.py

# RNG Battle Royale（SNN vs DNN vs LSTM）
python rng_battle_rigorous.py

# NIST乱数検定
python nist_test.py
```

## ファイル構成

```
snn-comprypto/
├── comprypto_system.py           # メインシステム（温度パラメータ対応）
├── comprypto_numba.py            # Numba高速化版
├── nist_test.py                  # NIST SP 800-22 乱数検定
├── rng_battle_royale.py          # RNG Battle 簡易版
├── rng_battle_v2.py              # RNG Battle ニューロン数比較版
├── rng_battle_full.py            # RNG Battle フル総当たり戦
├── rng_battle_rigorous.py        # RNG Battle 厳密検証版（100,000ラウンド）
├── rng_battle_learning.py        # RNG Battle 学習あり版（v4 NEW）
├── rng_battle_ultimate.py        # RNG Battle 並列処理版（v4 NEW）
├── chaos_analysis.py             # カオス解析（v4 NEW）
├── adversarial_analysis.py       # 攻撃耐性分析（v4 NEW）
├── rng_battle_visualization.py   # グラフ生成（v4 NEW）
├── benchmarks/                   # ベンチマークスクリプト
│   ├── thermal_benchmark.py      # 温度パラメータ検証
│   ├── neuron_benchmark.py       # ニューロン数検証
│   └── phase_transition_analysis.py  # 相転移分析
├── results/                      # 実験結果（グラフ等）
└── docs/                         # ドキュメント
```

## 論文

Funasaki, H. (2026). SNN-Comprypto: Spiking Neural Network-based Simultaneous Compression and Encryption Using Chaotic Reservoir Dynamics (v4). Zenodo. https://doi.org/10.5281/zenodo.18287761

## ライセンス

CC BY 4.0

## Author

ろーる ([@hafufu-stack](https://github.com/hafufu-stack))

- Zenn: https://zenn.dev/cell_activation
- note: https://note.com/cell_activation

