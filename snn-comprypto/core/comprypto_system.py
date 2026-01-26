"""
SNN Comprypto System
====================

SNNによる「予測圧縮」と「カオス暗号化」を同時に行う次世代セキュリティシステムのプロトタイプ。
NumPyによる高速行列演算を利用し、Ryzen AI + RTX環境での実行を想定。

Author: ろーる
Date: 2026-01-14
"""

import numpy as np
import time
import hashlib
import matplotlib.pyplot as plt

# import japanize_matplotlib # Removed to avoid dependency issues
plt.rcParams["font.family"] = "MS Gothic"  # Windows standard Japanese font
import zlib


class CompryptoNeuron:
    """
    カオス的挙動と予測性能を両立させた改良型LIFニューロン
    """

    def __init__(self, dt=0.5, tau=20.0):
        self.dt = dt
        self.tau = tau
        self.v = -65.0
        self.v_rest = -65.0
        self.v_thresh = -50.0
        self.v_reset = -70.0

    def step(self, I_syn):
        # 微分方程式: dv/dt = (-(v - v_rest) + I) / tau
        dv = (-(self.v - self.v_rest) + I_syn) / self.tau * self.dt
        self.v += dv

        if self.v >= self.v_thresh:
            self.v = self.v_reset
            return 1.0  # Spike
        return 0.0


class CompryptoReservoir:
    """
    ニューロ・カオス・リザーバ (Neuro-Chaotic Reservoir)

    役割:
    1. 時系列パターンの予測 (Context Memory)
    2. 暗号論的疑似乱数の生成 (Chaotic Dynamics)
    """

    def __init__(self, key_seed, num_neurons=300, density=0.1, input_scale=40.0, temperature=1.0):
        np.random.seed(key_seed)  # 秘密鍵としてのシード

        self.num_neurons = num_neurons
        self.dt = 0.5
        self.temperature = temperature  # 熱雑音パラメータ（第2の暗号鍵）
        self.key_seed = key_seed  # シードを保存（ノイズ生成用）

        # 結合荷重 (Reservoir Weights)
        # カオスの縁(Edge of Chaos)になるようにスペクトル半径を調整
        self.W_res = np.random.randn(num_neurons, num_neurons)
        rho = max(abs(np.linalg.eigvals(self.W_res)))
        self.W_res *= 1.4 / rho  # 1.4 -> ややカオス領域

        # 疎結合化
        mask = np.random.rand(num_neurons, num_neurons) < density
        self.W_res *= mask

        # 入力層 (Input Weights)
        self.W_in = np.random.randn(num_neurons, 1) * input_scale

        # 読み出し層 (Readout Weights) - Online Learning対象
        self.W_out = np.zeros(num_neurons)

        # ニューロン初期化
        self.neurons = [CompryptoNeuron(dt=self.dt) for _ in range(num_neurons)]

        # 状態変数
        self.fire_rate = np.zeros(num_neurons)  # x(t): Low-pass filtered spikes
        self.current_spikes = np.zeros(num_neurons)

        # 学習率 (LMS)
        self.alpha = 0.005

    def step_predict(self, input_val):
        """
        1ステップ実行し、予測値を出力する
        """
        # 入力の正規化 (0-255 -> -1.0~1.0)
        u = (input_val / 127.5) - 1.0

        # 電流計算: I = W_res @ x(t-1) + W_in @ u(t) + thermal_noise
        I_rec = self.W_res @ self.fire_rate
        I_ext = (self.W_in * u).flatten()
        
        # 熱雑音の注入（温度パラメータが第2の暗号鍵として機能）
        # 同じseed + 同じtemperatureなら同じノイズが生成される
        # temperatureがわずかでも違うとノイズ分散が変わり、復号不能になる
        thermal_noise = np.random.normal(0, 0.5 * self.temperature, self.num_neurons)
        I_total = I_rec + I_ext + thermal_noise

        # ニューロン更新
        spikes = np.zeros(self.num_neurons)
        for i, n in enumerate(self.neurons):
            # カオス性を高めるための非線形バイアス
            # bias = np.sin(i) * 5.0
            bias = 25.0
            if n.step(I_total[i] + bias) > 0.0:
                spikes[i] = 1.0

        self.current_spikes = spikes

        # 状態更新 (Exponential Moving Average)
        self.fire_rate = 0.7 * self.fire_rate + 0.3 * spikes

        # 線形読み出しによる予測
        y = self.W_out @ self.fire_rate

        # 非正規化
        pred_val = (y + 1.0) * 127.5
        return max(0, min(255, int(pred_val)))

    def train(self, target_val):
        """
        予測誤差に基づいて読み出し重みを更新 (Online Learning)
        """
        # 正解の正規化
        d = (target_val / 127.5) - 1.0

        # 現在の出力
        y = self.W_out @ self.fire_rate

        # 誤差
        e = d - y

        # LMS Update: w_new = w_old + alpha * error * input
        self.W_out += self.alpha * e * self.fire_rate

    def get_keystream_byte(self):
        """
        現在のリザーバ状態から1バイトの鍵ストリームを生成
        (ハッシュ関数を通して非線形性を極大化)
        """
        # 状態ベクトルをバイト列化
        # float64だと細かすぎるので、発火レジスタ(spikes)を使うのがロバストだが、
        # カオス性を最大化するため fire_rate の下位ビット等が予測困難であることを利用する
        # 改良: fire_rate(平滑化)よりも、膜電位(v)のほうが瞬時変動が大きくカオス的
        state_values = np.array([n.v for n in self.neurons])
        state_bytes = state_values.tobytes()

        # SHA-256で攪拌
        h = hashlib.sha256(state_bytes).digest()

        # 最初の1バイトを返す (全体を使っても良いが今回は等速で)
        return h[0]


class SNNCompryptor:
    """
    SNN圧縮・暗号化統合エンジン
    """

    def __init__(self, key_seed=2026, temperature=1.0):
        self.key_seed = key_seed
        self.temperature = temperature  # 第2の暗号鍵（熱雑音パラメータ）

    def compress_encrypt(self, data):
        """
        データ -> [圧縮 & 暗号化] -> 暗号化ペイロード
        """
        # 送信側「脳」の作成（温度パラメータ付き）
        brain = CompryptoReservoir(self.key_seed, temperature=self.temperature)

        encrypted_residuals = bytearray()

        # 最初の予測は前データがないので0固定
        last_val = 0

        predictions = []  # Debug用

        for val in data:
            # 1. 未来予測 (Predict)
            pred = brain.step_predict(last_val)
            predictions.append(pred)

            # 2. 残差計算 (Compress)
            # diff = 実測 - 予測 (mod 256)
            # これにより値が小さく偏る -> エントロピー低下
            residual = (val - pred) % 256

            # 3. リザーバカオス鍵生成 (Encrypt Key)
            key_byte = brain.get_keystream_byte()

            # 4. 暗号化 (XOR)
            # 圧縮された残差を、さらにカオス鍵でマスクする
            # 結果は一見ランダムに見える (エントロピー再上昇) が、
            # 復号時にキーがないと元に戻せない
            cipher_byte = residual ^ key_byte
            encrypted_residuals.append(cipher_byte)

            # 5. オンライン学習 (Train)
            # 次の予測精度を上げるため、実測値で脳を更新
            brain.train(val)

            last_val = val

        return encrypted_residuals, predictions

    def decrypt_decompress(self, encrypted_data):
        """
        暗号化ペイロード -> [復号 & 展開] -> 元データ
        """
        # 受信側「脳」の作成 (同じSeed + 同じ温度 = 同じ脳)
        brain = CompryptoReservoir(self.key_seed, temperature=self.temperature)

        restored_data = bytearray()
        last_val = 0

        for cipher_byte in encrypted_data:
            # 1. 未来予測 (受信側でも同じ予測が可能)
            pred = brain.step_predict(last_val)

            # 2. カオス鍵生成 (受信側でも同じ状態なら同じ鍵)
            key_byte = brain.get_keystream_byte()

            # 3. 復号 (XOR)
            # 暗号残差 ^ 鍵 = 元の残差
            residual = cipher_byte ^ key_byte

            # 4. 展開 (Decompress)
            # 予測 + 残差 = 元の値
            val = (pred + residual) % 256
            restored_data.append(val)

            # 5. オンライン学習
            # 復元した値を使って脳を更新 (これで送信側と同期維持)
            brain.train(val)

            last_val = val

        return restored_data


def run_benchmark():
    print("SNN Comprypto Benchmark")
    print("=======================")

    # 1. テストデータ生成 (正弦波: 予測しやすい -> 圧縮効果大)
    print("\n[Test 1: Simple Sine Wave]")
    t = np.linspace(0, 8 * np.pi, 1000)
    wave = (np.sin(t) * 100 + 128).astype(np.uint8)
    data1 = bytearray(wave)

    # 処理実行
    sys = SNNCompryptor(key_seed=12345)

    start = time.time()
    encrypted1, preds1 = sys.compress_encrypt(data1)
    enc_time = time.time() - start

    start = time.time()
    restored1 = sys.decrypt_decompress(encrypted1)
    dec_time = time.time() - start

    # 検証
    is_ok = data1 == restored1

    # 評価指標
    # 注意: 暗号化するとエントロピーは最大化(8.0)に近づくため、
    # 単純な圧縮率(ファイルサイズ)では評価できない。
    # 「復号中間体(Residual)」のエントロピーが低いことが圧縮の証拠。
    # しかし本システムは Compact & Secure を同時に行うため、
    # 出力自体はランダムに見えるのが正解。
    # 圧縮効果を確認するには、暗号化なしのResidualを見る必要があるが、
    # ここでは最終的な安全性と整合性を確認する。

    print(f"Data Size: {len(data1)} bytes")
    print(f"Encrypt Time: {enc_time*1000:.2f} ms")
    print(f"Decrypt Time: {dec_time*1000:.2f} ms")
    print(f"Integrity Check: {'✅ OK' if is_ok else '❌ NG'}")

    # 2. テストデータ生成 (テキスト)
    print("\n[Test 2: English Text]")
    text = b"Neural networks mimic the human brain. " * 50
    sys_text = SNNCompryptor(key_seed=999)
    encrypted2, _ = sys_text.compress_encrypt(text)
    restored2 = sys_text.decrypt_decompress(encrypted2)

    print(f"Data Size: {len(text)} bytes")
    print(f"Integrity Check: {'✅ OK' if (text == restored2) else '❌ NG'}")

    # 3. アバランチ効果 (鍵の敏感性)
    print("\n[Test 3: Avalanche Effect (Key Sensitivity)]")
    sys_wrong = SNNCompryptor(key_seed=12346)  # Seed +1 difference
    wrong_restore = sys_wrong.decrypt_decompress(encrypted1)

    # 正しく復号できていないことを確認 (ランダムなバイト列になっているはず)
    match_rate = np.mean(np.array(wave) == np.array(wrong_restore)) * 100
    print(f"Decryption with Wrong Key (Seed+1):")
    print(f"Match Rate: {match_rate:.2f}% (Should be ~0.4% for random 256 values)")
    if match_rate < 1.0:
        print(">> Security: ✅ Excellent (Unreadable without correct key)")
    else:
        print(">> Security: ⚠️ Weak")

    # 可視化
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(wave[:200], label="元の信号")
    plt.plot(preds1[:200], label="SNNによる予測", linestyle="dashed")
    plt.title("①入力信号とSNNの予測")
    plt.xlabel("サンプル番号")
    plt.ylabel("値 (0-255)")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(3, 1, 2)
    residuals = [(int(a) - p) % 256 for a, p in zip(wave, preds1)]
    plt.plot(
        residuals[:200], color="green", label="予測残差（圧縮された情報）"
    )
    plt.title("②中間段階：予測残差（圧縮効果）")
    plt.xlabel("サンプル番号")
    plt.ylabel("残差 (0-255)")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(3, 1, 3)
    plt.plot(encrypted1[:200], color="red", label="暗号化された出力")
    plt.title("③最終出力：カオス鍵で暗号化済み")
    plt.xlabel("サンプル番号")
    plt.ylabel("暗号値 (0-255)")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_benchmark()
