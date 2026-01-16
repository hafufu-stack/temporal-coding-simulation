# Temporal Coding Simulation

## 背景

脳の神経細胞はスパイク情報が入る順序やタイミングを調節することで、扱える情報量を増やしているのではないかと修論で考察しました。
このリポジトリでは、それをSNN（Spiking Neural Network）で検証していきたいと思います。

因みに私の修論は海馬歯状回の顆粒細胞のシミュレーションでした。
この細胞には、空間情報を運ぶMD入力と非空間情報を運ぶLD入力があり、それらは相互作用があるようです。

## 開発の方針や予定

- [x] **Phase 1: 記憶**
    - 10個のニューロンで約111ビットの保持に成功
- [x] **Phase 2: 演算（今回達成！）**
    - 30個のニューロン（入力20＋演算10）を使用し、111ビット同士の加算（時間領域演算）に成功
- [x] **Phase 3: 暗号化・圧縮（NEW!）**
    - 「予測圧縮」と「カオス暗号化」を同時に行うSNNシステム「Comprypto」を開発
    - Numbaによる高速化で **7.5倍** の処理速度向上を達成
    - 修論のTsodyks-Markramダイナミックシナプスモデルを応用した本格版も実装
- [ ] **Phase 4: 長期的な目標**
    - GPU (CUDA) 対応による更なる高速化
    - より多くのニューロンで、更なる性能向上を目指す。単にニューロン数に線形比例するのではなく、指数関数的に情報量が増えるようなネットワークを目指す。
    - 実用的なライブラリとしてリリース (`pip install snn-comprypto`)
    - 未踏ターゲット等への応募

## ディレクトリ構成

本リポジトリは実験ごとにフォルダ分けされています。

- **`10-neuron-memory/`**：10個のニューロンのSNNに情報を記憶させる（符号化の検証）
- **`snn-operation/`**：SNNで演算を行う（現在は加算のみ）
- **`snn-comprypto/`**：SNNで情報の暗号化・圧縮を行う

## プログラムの概要

### 1. 10-neuron-memory（記憶実験）
10個のニューロンを用い、最も効率良く情報を記憶させる方法を検証するため、2つの符号化方式を比較しました。

1.  **独立符号化** (`burst_phase_coding_10neurons.py`)
    - 各ニューロンが独立して「絶対時刻」で情報を表現。
    - 時間分解能：dt = 0.1ms
    - 理論情報量：**約82ビット**

2.  **相関符号化** (`correlation_coding_10neurons.py`)
    - 1つのニューロンを基準（MD）とし、他（LD）は「相対的なズレ」で情報を表現。
    - 修論の仮説「MD-LD相互作用」を再現。
    - 時間分解能：dt = 0.05ms
    - 理論情報量：**約111ビット** (大幅向上)

### 2. snn-operation（演算実験）
効率良く演算させる方法を検証していきます。

*   **加算システム** (`30-neuron-adder.py`)
    *   **構成**: 30ニューロン（入力A群10 + 入力B群10 + 演算層10）
    *   **機能**: 相関符号化された「111ビットの巨大数」同士の足し算を行います。
    *   **原理**: 電圧の加算ではなく、スパイク発火の「タイミング（位相）の加算」によって計算を行います。
    *   **成果**: 従来の64bit整数を遥かに超える桁数を、わずか30個の素子で処理できることを示しました。

### 3. snn-comprypto（暗号化・圧縮実験）
SNNの「予測能力」と「カオス的ダイナミクス」を利用し、新たなセキュリティシステム開発を目指しています。

*   **特徴**
    - **予測圧縮**：SNNがデータの文脈を学習・予測し、予測誤差のみを記録（圧縮）
    - **カオス暗号化**：ニューロン膜電位のカオス的変動を鍵生成に利用
    - **同時処理**：圧縮と暗号化を別々のプロセスではなくSNNの1ステップで実行
    - **低消費電力**：発火時のみ計算 → IoT/エッジデバイス向け
 
各ファイルについて
```
snn-comprypto/
├── README.md                 # 詳細ドキュメント
├── comprypto_system.py       # 基本実装
├── comprypto_numba.py        # Numba高速化版（7.5倍速い）
├── comprypto_benchmark.py    # AES/GZIPとの性能比較
└── tm_crypto_engine.py       # 修論ベースのTsodyks-Markramモデル版
```

## 動作環境

- Python 3.10+
- NumPy
- Matplotlib
- Numba (高速化版を使う場合)
- Japanize-Matplotlib (日本語フォント表示用)

## 実行方法

必要なライブラリをインストール後、各スクリプトを実行してください。

```bash
pip install numpy matplotlib japanize-matplotlib numba
```
### 実験1：SNNによる情報保存
#### 独立符号化モデル
```bash
cd 10-neuron-memory
python burst_phase_coding_10neurons.py
```

#### 相関符号化モデル
```bash
cd 10-neuron-memory
python correlation_coding_10neurons.py
```

### 実験2：SNNによる演算
#### 加算モデル
```bash
cd snn-operation
python 30-neuron-adder.py
```

### 実験3：SNNによる暗号化・圧縮
```bash
cd snn-comprypto

# 基本デモ
python comprypto_system.py

# ベンチマーク（AES/GZIPと比較）
python comprypto_benchmark.py

# 修論ベース版
python tm_crypto_engine.py
```

実行すると、シミュレーション結果の波形グラフが表示され、コンソールに達成した情報量や計算結果が出力されます。

## 現時点での考察

1.  **相関符号化の優位性**
    独立符号化よりも相関符号化の方がエントロピー（表現できる情報量）が上回りました。ニューロンを1つ基準用に消費（犠牲に）してでも、相対時間符号化を行う方がシステム全体の情報量は増大することが示唆されました。

2.  **SNNによる演算の可能性**
    たった30個のニューロンで111ビット（10進数で約34桁）の演算が可能であることが実証されました。これは、SNNが「発火のタイミング」という連続値に近い次元を利用することで、デジタル回路よりも高密度な計算が可能である可能性を示しています。

3.  **暗号化・圧縮への応用**
    「計算による乱数」ではなく、「振る舞いによるカオス」を利用することで、生物の脳が持つ "予測不能なゆらぎ" をセキュリティに応用できる可能性を示しました。


## 作者

**ろーる**
*   **note**：[https://note.com/cell_activation](https://note.com/cell_activation) （日記や思いを発信）
*   **Zenn**：[https://zenn.dev/cell_activation](https://zenn.dev/cell_activation) （プログラムの技術解説や構想を発信）
*   **GitHub**：[https://github.com/hafufu-stack](https://github.com/hafufu-stack)



