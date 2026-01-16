"""
SNN Comprypto 詳細可視化スクリプト
===================================

SNNの内部動作を理解するための詳細なグラフを生成します。

Author: ろーる
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "MS Gothic"

# comprypto_systemから必要なクラスをインポート
from comprypto_system import CompryptoReservoir, SNNCompryptor


def run_detailed_visualization():
    print("SNN Comprypto 詳細可視化")
    print("=" * 40)

    # テストデータ生成（正弦波）
    t = np.linspace(0, 8 * np.pi, 500)
    wave = (np.sin(t) * 100 + 128).astype(np.uint8)
    data = bytearray(wave)

    # リザーバを作成
    key_seed = 12345
    brain = CompryptoReservoir(key_seed)

    # 記録用リスト
    predictions = []
    residuals = []
    keys = []
    encrypted = []
    
    # ニューロンの膜電位を記録（代表的な10個）
    membrane_history = [[] for _ in range(10)]
    
    # スパイク活動を記録
    spike_history = []

    last_val = 0

    print("処理中...")
    for i, val in enumerate(data):
        # 予測
        pred = brain.step_predict(last_val)
        predictions.append(pred)

        # 残差
        residual = (val - pred) % 256
        residuals.append(residual)

        # 鍵生成
        key_byte = brain.get_keystream_byte()
        keys.append(key_byte)

        # 暗号化
        cipher = residual ^ key_byte
        encrypted.append(cipher)

        # 膜電位を記録（最初の10ニューロン）
        for j in range(10):
            membrane_history[j].append(brain.neurons[j].v)

        # スパイク活動を記録
        spike_history.append(brain.current_spikes.copy())

        # オンライン学習
        brain.train(val)

        last_val = val

    print("グラフ生成中...")

    # 大きな図を作成（6行）
    fig, axes = plt.subplots(6, 1, figsize=(14, 16))

    # 1. 入力信号 vs 予測
    ax1 = axes[0]
    ax1.plot(wave[:200], label="元の信号", color="blue", linewidth=2)
    ax1.plot(predictions[:200], label="SNNの予測", color="orange", linestyle="--", linewidth=2)
    ax1.set_title("①入力信号とSNNの予測（学習が進むと一致してくる）", fontsize=12)
    ax1.set_xlabel("サンプル番号")
    ax1.set_ylabel("値 (0-255)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. 予測残差（圧縮効果）
    ax2 = axes[1]
    ax2.plot(residuals[:200], label="予測残差", color="green", linewidth=1.5)
    ax2.axhline(y=0, color="red", linestyle=":", alpha=0.5)
    ax2.set_title("②予測残差（0に近いほど圧縮効率が良い）", fontsize=12)
    ax2.set_xlabel("サンプル番号")
    ax2.set_ylabel("残差 (0-255)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 3. 生成された鍵（カオス的に変動）
    ax3 = axes[2]
    ax3.plot(keys[:200], label="生成された鍵バイト", color="purple", linewidth=1)
    ax3.set_title("③カオス鍵（膜電位から生成、ランダムに見える）", fontsize=12)
    ax3.set_xlabel("サンプル番号")
    ax3.set_ylabel("鍵値 (0-255)")
    ax3.legend()
    ax3.grid(alpha=0.3)

    # 4. 暗号化された出力
    ax4 = axes[3]
    ax4.plot(encrypted[:200], label="暗号化された出力", color="red", linewidth=1)
    ax4.set_title("④最終暗号出力（残差 XOR 鍵）", fontsize=12)
    ax4.set_xlabel("サンプル番号")
    ax4.set_ylabel("暗号値 (0-255)")
    ax4.legend()
    ax4.grid(alpha=0.3)

    # 5. ニューロンの膜電位（代表10個）
    ax5 = axes[4]
    for j in range(10):
        ax5.plot(membrane_history[j][:200], alpha=0.7, linewidth=0.8, label=f"N{j}")
    ax5.axhline(y=-50, color="red", linestyle="--", linewidth=2, label="発火閾値 (-50mV)")
    ax5.set_title("⑤ニューロンの膜電位（赤線を超えると発火）", fontsize=12)
    ax5.set_xlabel("サンプル番号")
    ax5.set_ylabel("膜電位 (mV)")
    ax5.legend(loc="upper right", fontsize=8, ncol=2)
    ax5.grid(alpha=0.3)

    # 6. スパイク活動（ラスタープロット）
    ax6 = axes[5]
    spike_times = []
    neuron_ids = []
    for t_idx, spikes in enumerate(spike_history[:200]):
        for n_idx, spike in enumerate(spikes[:50]):  # 最初の50ニューロン
            if spike > 0:
                spike_times.append(t_idx)
                neuron_ids.append(n_idx)
    
    ax6.scatter(spike_times, neuron_ids, s=1, c="black", marker="|")
    ax6.set_title("⑥スパイク活動（ラスタープロット：各点が1回の発火）", fontsize=12)
    ax6.set_xlabel("サンプル番号")
    ax6.set_ylabel("ニューロン番号 (0-49)")
    ax6.set_ylim(-1, 50)
    ax6.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("detailed_visualization.png", dpi=150)
    print("グラフを 'detailed_visualization.png' に保存しました！")
    plt.show()


if __name__ == "__main__":
    run_detailed_visualization()
