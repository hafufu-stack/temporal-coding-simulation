import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib  # 日本語表示用

class LIFNeuron:
    """
    Leaky Integrate-and-Fire (LIF) ニューロンモデル
    
    膜電位が漏れながら（Leaky）積分されていき（Integrate）、
    閾値を超えたら発火（Fire）するシンプルな神経細胞モデル
    """
    
    def __init__(self, dt=0.1, tau=10.0, v_rest=-65.0, v_thresh=-50.0, v_reset=-70.0):
        """
        LIFニューロンのパラメータを初期化
        
        Parameters:
        -----------
        dt : float
            時間刻み幅 [ms]
        tau : float  
            膜時定数 [ms]
        v_rest : float
            静止膜電位 [mV]
        v_thresh : float
            発火閾値 [mV]
        v_reset : float
            リセット電位 [mV]
        """
        self.dt = dt
        self.tau = tau
        self.v_rest = v_rest
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        
        # 現在の膜電位
        self.v = v_rest
        
        # 発火時刻を記録するリスト
        self.spike_times = []
    
    def reset(self):
        """ニューロンの状態をリセット"""
        self.v = self.v_rest
        self.spike_times = []
    
    def step(self, I_syn, t):
        """
        1ステップ分の膜電位を計算
        
        Parameters:
        -----------
        I_syn : float
            シナプス入力電流
        t : float
            現在時刻 [ms]
        
        Returns:
        --------
        bool : 発火したかどうか
        """
        # オイラー法で膜電位を更新
        dv = (-(self.v - self.v_rest) + I_syn) / self.tau * self.dt
        self.v += dv
        
        # 発火判定
        if self.v >= self.v_thresh:
            self.spike_times.append(t)
            self.v = self.v_reset
            return True
        
        return False


class TenNeuronBurstPhaseCoding:
    """
    10個のLIFニューロンを使ったバースト位相符号化システム
    
    各ニューロンが2回発火（バースト）して、
    - 1回目の発火タイミング（位相）
    - 発火間隔（ISI）
    の組み合わせで情報を符号化
    
    理論的情報量：約81ビット（561^9通り）
    """
    
    def __init__(self, dt=0.1, sim_time=200.0, num_neurons=10):
        """
        Parameters:
        -----------
        dt : float
            時間刻み幅 [ms]
        sim_time : float
            シミュレーション時間 [ms]（バーストを入れるため長めに）
        num_neurons : int
            ニューロンの数（デフォルト10）
        """
        self.dt = dt
        self.sim_time = sim_time
        self.num_neurons = num_neurons
        
        # 10個のLIFニューロンを作成
        self.neurons = [LIFNeuron(dt=dt) for _ in range(num_neurons)]
        
        # 符号化パラメータ
        self.phase_min = 0      # 位相の最小値 [ms]
        self.phase_max = 50     # 位相の最大値 [ms]
        self.phase_step = 1     # 位相の刻み幅 [ms]
        
        self.isi_min = 5        # ISIの最小値 [ms]
        self.isi_max = 15       # ISIの最大値 [ms]
        self.isi_step = 1       # ISIの刻み幅 [ms]
        
        # 位相とISIの取りうる値のリスト
        self.phases = np.arange(self.phase_min, self.phase_max + 1, self.phase_step)
        self.isis = np.arange(self.isi_min, self.isi_max + 1, self.isi_step)
        
        # 各ニューロンの状態数
        self.states_per_neuron = len(self.phases) * len(self.isis)
        
        # 基準ニューロン（N1）を固定するため、自由度は9個
        self.free_neurons = num_neurons - 1
        
        # 理論的な状態数と情報量
        self.total_states = self.states_per_neuron ** self.free_neurons
        self.total_bits = np.log2(float(self.total_states))
        
        print("=" * 60)
        print("10ニューロン・バースト位相符号化システム初期化完了")
        print("=" * 60)
        print(f"ニューロン数: {num_neurons}")
        print(f"各ニューロンの発火回数: 2回（バースト）")
        print(f"位相の範囲: {self.phase_min}-{self.phase_max}ms ({len(self.phases)}通り)")
        print(f"ISIの範囲: {self.isi_min}-{self.isi_max}ms ({len(self.isis)}通り)")
        print(f"各ニューロンの状態数: {self.states_per_neuron}通り")
        print(f"自由度（基準ニューロンを除く）: {self.free_neurons}個")
        print(f"理論的な総状態数: {self.total_states:.2e}通り")
        print(f"理論的情報量: {self.total_bits:.2f}ビット")
        print("=" * 60)
    
    def generate_input_current(self, spike_times, t, amplitude=150.0, duration=2.0):
        """
        複数回のスパイク入力を発生させる関数
        
        Parameters:
        -----------
        spike_times : list of float
            入力を与える時刻のリスト [ms]
        t : float
            現在時刻 [ms]
        amplitude : float
            入力の大きさ
        duration : float
            各入力の持続時間 [ms]
        
        Returns:
        --------
        float : 現在時刻での入力電流
        """
        # いずれかの入力時刻の範囲内にいるかチェック
        for spike_time in spike_times:
            if spike_time <= t < spike_time + duration:
                return amplitude
        return 0.0
    
    def encode(self, value):
        """
        整数値を各ニューロンの発火パターン（位相とISI）に変換
        
        Parameters:
        -----------
        value : int
            符号化したい値（0 〜 total_states-1）
        
        Returns:
        --------
        list of tuple : 各ニューロンの（位相, ISI）のリスト
        """
        if value < 0 or value >= self.total_states:
            raise ValueError(f"入力値は0〜{self.total_states-1}の範囲で指定してください")
        
        # 基数変換的に値を分解
        # value を states_per_neuron 進数で表現
        remaining = value
        patterns = []
        
        # 基準ニューロン（N1）は固定パターン
        # 位相=0ms、ISI=10msで固定
        base_pattern = (0, 10)
        patterns.append(base_pattern)
        
        # 残り9個のニューロンのパターンを決定
        for i in range(self.free_neurons):
            # 現在の桁の値を取得（0 〜 states_per_neuron-1）
            digit = remaining % self.states_per_neuron
            remaining //= self.states_per_neuron
            
            # digit を（位相index, ISI index）に変換
            phase_idx = digit // len(self.isis)
            isi_idx = digit % len(self.isis)
            
            # 実際の位相とISIの値
            phase = self.phases[phase_idx]
            isi = self.isis[isi_idx]
            
            patterns.append((phase, isi))
        
        return patterns
    
    def decode(self, spike_times_list):
        """
        各ニューロンの発火時刻から元の値を復元
        
        Parameters:
        -----------
        spike_times_list : list of list
            各ニューロンの発火時刻のリスト
            spike_times_list[i] = [1回目の時刻, 2回目の時刻]
        
        Returns:
        --------
        int : 復元された値
        """
        # 基準ニューロン（N1）の発火時刻を取得
        if len(spike_times_list[0]) < 2:
            raise ValueError("基準ニューロンの発火が2回検出されませんでした")
        
        base_first = spike_times_list[0][0]
        
        # 各ニューロンの（位相, ISI）を計算
        patterns = []
        
        for i, spike_times in enumerate(spike_times_list):
            if len(spike_times) < 2:
                raise ValueError(f"ニューロン{i}の発火が2回検出されませんでした")
            
            # 位相：基準ニューロンの1回目からの相対時間
            phase = spike_times[0] - base_first
            
            # ISI：1回目と2回目の時間差
            isi = spike_times[1] - spike_times[0]
            
            # 最も近い設定値に丸める
            phase = self.phases[np.argmin(np.abs(self.phases - phase))]
            isi = self.isis[np.argmin(np.abs(self.isis - isi))]
            
            patterns.append((phase, isi))
        
        # 基準ニューロンを除いた残り9個から値を復元
        value = 0
        for i in range(self.free_neurons):
            phase, isi = patterns[i + 1]  # patterns[0]は基準なのでスキップ
            
            # （位相, ISI）から digit を計算
            phase_idx = np.where(self.phases == phase)[0][0]
            isi_idx = np.where(self.isis == isi)[0][0]
            digit =int(phase_idx * len(self.isis) + isi_idx)
            
            # 基数変換で値を復元
            value += digit * (self.states_per_neuron ** i)
        
        return value
    
    def simulate(self, patterns):
        """
        各ニューロンの発火パターンに従ってシミュレーションを実行
        
        Parameters:
        -----------
        patterns : list of tuple
            各ニューロンの（位相, ISI）のリスト
        
        Returns:
        --------
        dict : シミュレーション結果
        """
        # 全ニューロンをリセット
        for neuron in self.neurons:
            neuron.reset()
        
        # 基準時刻（余裕を持って30msから開始）
        base_time = 30.0
        
        # 各ニューロンへの入力タイミングを計算
        input_times_list = []
        for phase, isi in patterns:
            # 1回目の入力：base_time + phase
            first_spike = base_time + phase
            # 2回目の入力：1回目 + ISI
            second_spike = first_spike + isi
            input_times_list.append([first_spike, second_spike])
        
        # 膜電位の履歴を記録（10ニューロン全て）
        v_histories = [[] for _ in range(self.num_neurons)]
        t_history = []
        
        # シミュレーション実行
        t = 0.0
        while t <= self.sim_time:
            # 各ニューロンへの入力電流を計算して1ステップ進める
            for i, neuron in enumerate(self.neurons):
                I_syn = self.generate_input_current(input_times_list[i], t)
                neuron.step(I_syn, t)
                v_histories[i].append(neuron.v)
            
            t_history.append(t)
            t += self.dt
        
        # 結果をまとめる
        spike_times_list = [neuron.spike_times for neuron in self.neurons]
        
        return {
            'spike_times_list': spike_times_list,
            'v_histories': [np.array(v) for v in v_histories],
            't_history': np.array(t_history),
            'patterns': patterns
        }
    
    def test_encoding(self, test_values):
        """
        複数の値でエンコード・シミュレーション・デコードをテスト
        
        Parameters:
        -----------
        test_values : list
            テストする値のリスト
        
        Returns:
        --------
        list : 各値の結果を含むリスト
        """
        results = []
        
        print("\n" + "=" * 60)
        print("エンコード・デコードテスト開始")
        print("=" * 60)
        
        for idx, value in enumerate(test_values):
            print(f"\n[{idx+1}/{len(test_values)}] テスト値: {value}")
            
            # エンコード
            patterns = self.encode(value)
            print(f"  符号化パターン（最初の3ニューロンのみ表示）:")
            for i, (phase, isi) in enumerate(patterns[:3]):
                print(f"    N{i+1}: 位相={phase:2.0f}ms, ISI={isi:2.0f}ms")
            if len(patterns) > 3:
                print(f"    ... (残り{len(patterns)-3}個)")
            
            # シミュレーション
            sim_result = self.simulate(patterns)
            
            # デコード
            try:
                decoded_value = self.decode(sim_result['spike_times_list'])
                success = (value == decoded_value)
                
                if success:
                    print(f"  [OK] デコード成功: {decoded_value} （一致）")
                else:
                    print(f"  [NG] デコード失敗: {decoded_value} （不一致）")
            except Exception as e:
                decoded_value = None
                success = False
                print(f" エラー: {e}")
            
            results.append({
                'input': value,
                'patterns': patterns,
                'decoded': decoded_value,
                'success': success,
                'sim_result': sim_result
            })
        
        # 成功率を計算
        success_count = sum(r['success'] for r in results)
        success_rate = success_count / len(results) * 100
        
        print("\n" + "=" * 60)
        print(f"テスト完了: 成功率 {success_rate:.1f}% ({success_count}/{len(results)})")
        print("=" * 60)
        
        return results
    
    def visualize_results(self, results, num_examples=3):
        """
        シミュレーション結果を可視化（10ニューロン全ての膜電位）
        
        Parameters:
        -----------
        results : list
            test_encoding()の結果
        num_examples : int
            表示する例の数
        """
        # 表示する結果を選択
        indices = np.linspace(0, len(results)-1, min(num_examples, len(results)), dtype=int)
        selected_results = [results[i] for i in indices]
        
        # カラーマップ（10色）
        cmap = plt.cm.tab10
        colors = [cmap(i) for i in range(10)]
        
        # グラフを作成
        fig, axes = plt.subplots(len(selected_results), 1, 
                                figsize=(16, 5*len(selected_results)))
        
        if len(selected_results) == 1:
            axes = [axes]
        
        for ax, result in zip(axes, selected_results):
            sim = result['sim_result']
            
            # 10ニューロン全ての膜電位をプロット
            for i, v_hist in enumerate(sim['v_histories']):
                label = 'MD (基準)' if i == 0 else f'LD{i}'
                ax.plot(sim['t_history'], v_hist, 
                       label=label, linewidth=1.2, 
                       alpha=0.8, color=colors[i])
            
            # 発火閾値
            ax.axhline(y=-50, color='gray', linestyle='--', 
                      alpha=0.5, label='発火閾値')
            
            # タイトル
            status = "✓" if result['success'] else "✗"
            decoded_str = f"{result['decoded']:,}" if result['decoded'] is not None else "失敗"
            input_str = f"{result['input']:,}"
            ax.set_title(
                f"{status} 入力値: {input_str} → 復元値: {decoded_str}",
                fontsize=12
            )
            
            ax.set_xlabel('時刻 (ms)', fontsize=11)
            ax.set_ylabel('膜電位 (mV)', fontsize=11)
            ax.legend(loc='upper right', ncol=5, fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(25, 100)
        
        plt.tight_layout()
        plt.show()
        
        return fig


# 実行例
if __name__ == "__main__":
    import random
    
    # ============================================================
    # ランダムモード設定
    # 0: 固定のテスト値を使用
    # 1: 乱数を使う（大きさにばらつきあり）
    # ============================================================
    RANDOM_MODE = 0
    
    # システムを作成
    system = TenNeuronBurstPhaseCoding(dt=0.1, sim_time=200.0, num_neurons=10)
    
    if RANDOM_MODE == 1:
        # ランダムモード：桁数にばらつきを持たせた乱数
        test_values = []
        for _ in range(4):
            digits = random.randint(1, 10)  # 1〜10桁
            val = random.randint(10**(digits-1), min(10**digits - 1, int(system.total_states - 1)))
            test_values.append(val)
        
        print("\n" + "=" * 60)
        print("ランダムモード（桁数ばらつきあり）")
        print("=" * 60)
        for i, val in enumerate(test_values):
            print(f"  値{i+1}: {val:,} ({len(str(val))}桁)")
        print("=" * 60)
    else:
        # 固定モード
        test_values = [0, 100, 10000, 100000]
        print("\n注意: 固定テスト値でテストします")
        print(f"（理論上は0〜{system.total_states-1:.2e}の範囲が可能）")
    
    # エンコード・デコードテスト
    results = system.test_encoding(test_values)
    
    # 結果を可視化
    system.visualize_results(results, num_examples=len(test_values))
    
    # 情報量の確認
    print("\n" + "=" * 60)
    print("情報量の確認")
    print("=" * 60)
    print(f"2ニューロン版: 4.39ビット")
    print(f"10ニューロン版（今回）: {system.total_bits:.2f}ビット")
    print(f"情報密度の向上: {system.total_bits / 4.39:.1f}倍")
    print(f"ニューロン数あたり: {system.total_bits / system.num_neurons:.1f}ビット/ニューロン")
    print("=" * 60)