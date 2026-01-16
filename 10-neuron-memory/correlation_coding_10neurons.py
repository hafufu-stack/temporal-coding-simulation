import numpy as np
import matplotlib.pyplot as plt
import math
import japanize_matplotlib

class LIFNeuron:
    """
    高精度LIFニューロンモデル
    時間分解能を上げるため、パラメータを微調整
    """
    def __init__(self, dt=0.05, tau=10.0, v_rest=-65.0, v_thresh=-50.0, v_reset=-70.0):
        self.dt = dt
        self.tau = tau
        self.v_rest = v_rest
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.v = v_rest
        self.spike_times = []
    
    def reset(self):
        self.v = self.v_rest
        self.spike_times = []
    
    def step(self, I_syn, t):
        dv = (-(self.v - self.v_rest) + I_syn) / self.tau * self.dt
        self.v += dv
        if self.v >= self.v_thresh:
            self.spike_times.append(t)
            self.v = self.v_reset
            return True
        return False

class TenNeuronCorrelationCoding:
    """
    MD-LD相互作用模倣型：相関バースト符号化システム
    
    - Neuron 0 (MD役): 基準クロックとして動作
    - Neuron 1-9 (LD役): N0との「相対位相(τ)」と「ISI」で情報を符号化
    
    相対時間を使うことでノイズ耐性を高め、時間分解能を限界まで上げる設計
    """
    
    def __init__(self, dt=0.05, sim_time=200.0, num_neurons=10):
        self.dt = dt
        self.sim_time = sim_time
        self.num_neurons = num_neurons
        self.neurons = [LIFNeuron(dt=dt) for _ in range(num_neurons)]
        
        # --- 符号化パラメータ（高密度化） ---
        
        # 1. 相対位相 (MD-LD間の遅れ τ)
        # 修論の知見「位相差が重要」→ 0.2ms刻みという超高解像度を設定
        self.tau_min = 0.0
        self.tau_max = 50.0
        self.tau_step = 0.2  # 0.2msステップ (dt=0.05msなので十分検出可能)
        self.taus = np.arange(self.tau_min, self.tau_max + 0.001, self.tau_step) # 浮動小数点の誤差対策
        
        # 2. ISI (バースト間隔)
        # これも0.5ms刻みに細かくする
        self.isi_min = 5.0
        self.isi_max = 15.0
        self.isi_step = 0.5
        self.isis = np.arange(self.isi_min, self.isi_max + 0.001, self.isi_step)
        
        # 状態数の計算
        # 基準ニューロン(N0)は情報を持たないので、自由度は9個
        self.states_per_neuron = len(self.taus) * len(self.isis)
        self.free_neurons = num_neurons - 1
        
        self.total_states = self.states_per_neuron ** self.free_neurons
        # math.log2を使って巨大な数の対数を計算
        self.total_bits = math.log2(self.total_states)
        
        print("=" * 60)
        print("MD-LD相関バースト符号化システム（10ニューロン）初期化")
        print("=" * 60)
        print(f"時間分解能(dt): {dt} ms")
        print(f"役割分担:")
        print(f"  - Neuron 0 (MD役): 基準クロック（位相0固定）")
        print(f"  - Neuron 1-9 (LD役): MDとの相対時間差で符号化")
        print("-" * 60)
        print(f"相対位相(τ)範囲: {self.tau_min}-{self.tau_max}ms (刻み{self.tau_step}ms) -> {len(self.taus)}通り")
        print(f"ISI範囲        : {self.isi_min}-{self.isi_max}ms (刻み{self.isi_step}ms) -> {len(self.isis)}通り")
        print(f"1ニューロンあたりの状態数: {len(self.taus)} × {len(self.isis)} = {self.states_per_neuron}通り")
        print("-" * 60)
        print(f"理論的な総状態数: {self.total_states:.2e} 通り")
        print(f"理論的情報量    : {self.total_bits:.2f} ビット")
        print("=" * 60)

    def generate_input_current(self, spike_times, t, amplitude=200.0, duration=1.0):
        """入力電流生成（durationを1.0msに短縮して過発火を完全防止）"""
        for spike_time in spike_times:
            if spike_time <= t < spike_time + duration:
                return amplitude
        return 0.0
    
    def encode(self, value):
        """値をMD-LD相関パターンに変換"""
        if value >= self.total_states:
             raise ValueError(f"値が大きすぎます。上限: {self.total_states:.2e}")

        patterns = []
        
        # Neuron 0 (MD役) は基準パターン固定
        # 位相=0, ISI=10 (固定)
        patterns.append((0.0, 10.0))
        
        # Neuron 1-9 (LD役) のパターンを決定
        remaining = value
        
        for _ in range(self.free_neurons):
            # 巨大な整数を扱うため、一度Pythonのint計算で余りを出す
            digit = int(remaining % self.states_per_neuron)
            remaining //= self.states_per_neuron
            
            # digit を (τ index, ISI index) に分解
            tau_idx = digit // len(self.isis)
            isi_idx = digit % len(self.isis)
            
            tau = self.taus[tau_idx]
            isi = self.isis[isi_idx]
            
            patterns.append((tau, isi))
            
        return patterns

    def simulate(self, patterns):
        """シミュレーション実行"""
        for neuron in self.neurons:
            neuron.reset()
        
        base_time = 30.0  # MDの基準発火時刻
        
        # 入力タイミングのリスト作成
        input_times_list = []
        
        # Neuron 0 (MD)
        md_tau, md_isi = patterns[0]
        t0_1 = base_time + md_tau # md_tauは0
        t0_2 = t0_1 + md_isi
        input_times_list.append([t0_1, t0_2])
        
        # Neuron 1-9 (LD)
        for i in range(1, self.num_neurons):
            tau, isi = patterns[i]
            # MDの発火時刻(t0_1)からの相対遅延 τ で入力
            t_1 = t0_1 + tau
            t_2 = t_1 + isi
            input_times_list.append([t_1, t_2])
            
        # 実行
        t = 0.0
        v_histories = [[] for _ in range(self.num_neurons)]  # 10ニューロン全て記録
        t_history = []
        
        while t <= self.sim_time:
            for i, neuron in enumerate(self.neurons):
                I_syn = self.generate_input_current(input_times_list[i], t)
                neuron.step(I_syn, t)
                v_histories[i].append(neuron.v)
            
            t_history.append(t)
            t += self.dt
            
        return {
            'spike_times_list': [n.spike_times for n in self.neurons],
            'v_histories': v_histories,
            't_history': np.array(t_history),
            'md_base_time': t0_1
        }

    def decode(self, spike_times_list):
        """発火時刻から値を復元（相対時間を使用）"""
        
        # 1. MD (Neuron 0) の1回目の発火時刻を取得
        if len(spike_times_list[0]) < 1:
            return None # MDが発火しないと基準がない
        
        md_start_time = spike_times_list[0][0]
        
        decoded_value = 0
        
        # Neuron 1-9 から情報を復元
        for i in range(1, self.num_neurons):
            spikes = spike_times_list[i]
            if len(spikes) < 2:
                return None # バースト失敗
            
            # 相対位相 τ = LD発火時刻 - MD発火時刻
            detected_tau = spikes[0] - md_start_time
            
            # ISI
            detected_isi = spikes[1] - spikes[0]
            
            # 最も近い設定値を探す（量子化誤差の吸収）
            # abs誤差が最小になるインデックスを取得
            tau_idx = np.argmin(np.abs(self.taus - detected_tau))
            isi_idx = np.argmin(np.abs(self.isis - detected_isi))
            
            # 復元したパラメータ
            # tau = self.taus[tau_idx]
            # isi = self.isis[isi_idx]
            
            # 数値に戻す
            digit = int(tau_idx * len(self.isis) + isi_idx)
            
            # 桁の重みを掛けて足す
            weight = self.states_per_neuron ** (i - 1)
            decoded_value += digit * weight
            
        return decoded_value

    def test_run(self):
        """デモ実行"""
        # テストする値（非常に大きな値も含む）
        test_values = [
            0, 
            123456789,
            self.total_states // 2, # 中間値
            int(self.total_states - 1)   # 最大値
        ]
        
        results = []
        print("\n=== シミュレーション開始 ===")
        
        for val in test_values:
            print(f"\nTarget: {val:.2e} (または {val})")
            
            # 1. Encode
            patterns = self.encode(val)
            
            # 2. Simulate
            sim_res = self.simulate(patterns)
            
            # 3. Decode
            decoded = self.decode(sim_res['spike_times_list'])
            
            success = (val == decoded)
            mark = "[OK]" if success else "[NG]"
            print(f"Result: {mark} Decoded: {decoded}")
            
            results.append({
                'input': val,
                'success': success,
                'sim_res': sim_res
            })
            
        return results

    def visualize_results(self, results):
        """すべてのテスト結果を可視化（10ニューロン全て）"""
        num_results = len(results)
        fig, axes = plt.subplots(num_results, 1, figsize=(16, 5 * num_results))
        
        if num_results == 1:
            axes = [axes]
        
        # カラーマップ（10色）
        cmap = plt.cm.tab10
        colors = [cmap(i) for i in range(10)]
        
        for i, (ax, result) in enumerate(zip(axes, results)):
            sim = result['sim_res']
            v_hists = sim['v_histories']
            t_hist = sim['t_history']
            
            # 10ニューロン全てをプロット
            for n_idx in range(len(v_hists)):
                label = 'MD (基準)' if n_idx == 0 else f'LD{n_idx}'
                ax.plot(t_hist, v_hists[n_idx], color=colors[n_idx], alpha=0.8, 
                       linewidth=1.2, label=label)
            
            status = "[OK]" if result['success'] else "[NG]"
            ax.set_title(f"{status} 入力値: {result['input']:,}", fontsize=12)
            ax.set_ylabel("膜電位 [mV]")
            ax.set_xlim(25, 100)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', ncol=5, fontsize=8)
            
            if i == num_results - 1:
                ax.set_xlabel("Time [ms]")
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    import random
    
    # ============================================================
    # ランダムモード設定
    # 0: 固定のテスト値を使用
    # 1: 乱数を使う（大きさにばらつきあり）
    # ============================================================
    RANDOM_MODE = 0
    
    # システム構築
    system = TenNeuronCorrelationCoding()
    
    if RANDOM_MODE == 1:
        # ランダムモード：桁数にばらつきを持たせた乱数
        test_values = []
        for _ in range(4):
            digits = random.randint(1, 12)  # 1〜12桁
            val = random.randint(10**(digits-1), min(10**digits - 1, int(system.total_states - 1)))
            test_values.append(val)
        
        print("\n" + "=" * 60)
        print("ランダムモード（桁数ばらつきあり）")
        print("=" * 60)
        for i, val in enumerate(test_values):
            print(f"  値{i+1}: {val:,} ({len(str(val))}桁)")
        print("=" * 60)
    else:
        # 固定モード：ここでテストしたい値を指定
        test_values = [0, 12345, system.total_states // 2, int(system.total_states - 1)]
        
        print("\n" + "=" * 60)
        print("固定モード")
        print("=" * 60)
        print("テスト値:")
        for val in test_values:
            print(f"  {val:,}")
        print("=" * 60)
        
    # テスト実行
    results = []
    for val in test_values:
        patterns = system.encode(val)
        sim_res = system.simulate(patterns)
        decoded = system.decode(sim_res['spike_times_list'])
        success = (val == decoded)
        mark = "[OK]" if success else "[NG]"
        print(f"{mark} {val:,} -> {decoded:,}" if decoded is not None else f"[NG] {val:,} -> 失敗")
        results.append({'input': val, 'success': success, 'sim_res': sim_res})
    
    # 成功率表示
    success_count = sum(r['success'] for r in results)
    print("\n" + "="*60)
    print(f"最終テスト結果: {success_count}/{len(results)} 成功")
    print(f"達成情報量: {system.total_bits:.2f} bits")
    print(f"ビット効率: {system.total_bits / system.num_neurons:.2f} ビット/ニューロン")
    print("="*60)

    # 全結果をグラフ表示
    if len(results) > 0:
        system.visualize_results(results)