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
            時間刻み幅 [ms]。0.1msならかなり精密なシミュレーション
        tau : float  
            膜時定数 [ms]。大きいほど膜電位がゆっくり減衰する
        v_rest : float
            静止膜電位 [mV]。何も入力がないときの電位
        v_thresh : float
            発火閾値 [mV]。この値を超えたら発火
        v_reset : float
            リセット電位 [mV]。発火後にこの値にリセットされる
        """
        self.dt = dt              # 時間刻み幅
        self.tau = tau            # 膜時定数
        self.v_rest = v_rest      # 静止膜電位
        self.v_thresh = v_thresh  # 発火閾値
        self.v_reset = v_reset    # リセット電位
        
        # 現在の膜電位（最初は静止膜電位から開始）
        self.v = v_rest
        
        # 発火時刻を記録するリスト
        self.spike_times = []
    
    def reset(self):
        """
        ニューロンの状態をリセット（シミュレーション開始前に呼ぶ）
        """
        self.v = self.v_rest      # 膜電位を静止膜電位に戻す
        self.spike_times = []     # 発火記録をクリア
    
    def step(self, I_syn, t):
        """
        1ステップ分の膜電位を計算（オイラー法による微分方程式の数値計算）
        
        LIFの微分方程式:
        tau * dV/dt = -(V - V_rest) + I_syn
        
        つまり：
        - 膜電位は静止膜電位に向かって減衰する: -(V - V_rest)
        - シナプス入力で増加する: +I_syn
        
        Parameters:
        -----------
        I_syn : float
            シナプス入力電流 [任意単位]。正なら興奮性、負なら抑制性
        t : float
            現在時刻 [ms]
        
        Returns:
        --------
        bool : 発火したかどうか（True=発火、False=発火しない）
        """
        # オイラー法で膜電位を更新
        # dV/dt を dt 時間だけ進める
        dv = (-(self.v - self.v_rest) + I_syn) / self.tau * self.dt
        self.v += dv
        
        # 発火判定: 膜電位が閾値を超えたか？
        if self.v >= self.v_thresh:
            self.spike_times.append(t)  # 発火時刻を記録
            self.v = self.v_reset        # 膜電位をリセット
            return True  # 発火した
        
        return False  # 発火しなかった


class TwoNeuronLIFPhaseCoding:
    """
    2つのLIFニューロンを使った相対位相符号化システム
    
    ニューロンAとBの発火時間差で情報を符号化する
    """
    
    def __init__(self, dt=0.1, sim_time=100.0):
        """
        Parameters:
        -----------
        dt : float
            時間刻み幅 [ms]
        sim_time : float
            シミュレーション時間 [ms]
        """
        self.dt = dt                    # 時間刻み幅
        self.sim_time = sim_time        # シミュレーション時間
        
        # 2つのLIFニューロンを作成
        self.neuron_A = LIFNeuron(dt=dt)
        self.neuron_B = LIFNeuron(dt=dt)
        
        # ルックアップテーブル: 入力値 <-> 時間差のマッピング
        # 0〜20の整数を-10〜+10msの時間差に対応させる
        self.value_to_delta = {}   # 入力値 -> 時間差
        self.delta_to_value = {}   # 時間差 -> 入力値
        
        for value in range(21):    # 0から20まで
            delta = value - 10     # -10から+10に変換
            self.value_to_delta[value] = delta
            self.delta_to_value[delta] = value
        
        print("LIF 2ニューロン相対位相符号化システム初期化完了")
        print(f"時間刻み幅: {dt} ms")
        print(f"符号化可能な値: 0〜20 (21通り = {np.log2(21):.2f}ビット)")
    
    def generate_input_current(self, spike_time, t, amplitude=150.0, duration=5.0):
        """
        特定時刻にスパイク入力を発生させる関数
        
        実際のシナプスでは、プレシナプスが発火すると
        ポストシナプスに一時的な電流が流れる。
        ここでは矩形波（短い間だけ一定の電流）で近似
        
        Parameters:
        -----------
        spike_time : float
            入力を与える時刻 [ms]
        t : float
            現在時刻 [ms]
        amplitude : float
            入力の大きさ（大きいほど膜電位の上昇が速い）
            ※修正：50→150に増強（確実に発火するように）
        duration : float
            入力の持続時間 [ms]
            ※修正：2→5msに延長（十分な電流を流す）
        
        Returns:
        --------
        float : 現在時刻での入力電流
        """
        # spike_time から spike_time+duration の間だけ電流を流す
        if spike_time <= t < spike_time + duration:
            return amplitude  # 入力あり
        else:
            return 0.0        # 入力なし
    
    def encode_and_simulate(self, value):
        """
        入力値を符号化して、2つのLIFニューロンをシミュレーション
        
        手順:
        1. 入力値から目標の時間差を決定（ルックアップテーブル）
        2. ニューロンAは常に0msで発火するように入力
        3. ニューロンBは時間差分ずらして入力
        4. シミュレーション実行
        5. 実際の発火時刻を記録
        
        Parameters:
        -----------
        value : int
            符号化したい値 (0〜20)
        
        Returns:
        --------
        dict : シミュレーション結果
            - 'value': 入力値
            - 'target_delta': 目標の時間差
            - 'A_spike_time': ニューロンAの発火時刻
            - 'B_spike_time': ニューロンBの発火時刻  
            - 'actual_delta': 実際の時間差
            - 'v_history_A': ニューロンAの膜電位履歴
            - 'v_history_B': ニューロンBの膜電位履歴
            - 't_history': 時刻の履歴
        """
        # 入力値チェック
        if value not in self.value_to_delta:
            raise ValueError(f"入力値は0〜20の範囲で指定してください。入力値: {value}")
        
        # 目標の時間差を取得
        target_delta = self.value_to_delta[value]
        
        # ニューロンをリセット（前回のシミュレーション結果をクリア）
        self.neuron_A.reset()
        self.neuron_B.reset()
        
        # ニューロンAへの入力タイミング（基準として20msに設定）
        # ※20msから始めることで、負の時間差でも問題なく処理できる
        input_time_A = 20.0
        
        # ニューロンBへの入力タイミング（時間差分ずらす）
        input_time_B = input_time_A + target_delta
        
        # 膜電位の履歴を記録するリスト
        v_history_A = []
        v_history_B = []
        t_history = []
        
        # シミュレーション実行
        # 0msからsim_timeまで、dtステップずつ時間を進める
        t = 0.0
        while t <= self.sim_time:
            # 各ニューロンへの入力電流を計算
            I_A = self.generate_input_current(input_time_A, t)
            I_B = self.generate_input_current(input_time_B, t)
            
            # 各ニューロンを1ステップ進める
            self.neuron_A.step(I_A, t)
            self.neuron_B.step(I_B, t)
            
            # 膜電位を記録（後でグラフ化するため）
            v_history_A.append(self.neuron_A.v)
            v_history_B.append(self.neuron_B.v)
            t_history.append(t)
            
            # 時刻を進める
            t += self.dt
        
        # 実際の発火時刻を取得
        # 各ニューロンは複数回発火する可能性があるが、最初の発火時刻を使う
        A_spike_time = self.neuron_A.spike_times[0] if self.neuron_A.spike_times else None
        B_spike_time = self.neuron_B.spike_times[0] if self.neuron_B.spike_times else None
        
        # 実際の時間差を計算
        if A_spike_time is not None and B_spike_time is not None:
            actual_delta = B_spike_time - A_spike_time
        else:
            actual_delta = None  # どちらか発火しなかった
        
        # 結果を辞書にまとめて返す
        return {
            'value': value,
            'target_delta': target_delta,
            'A_spike_time': A_spike_time,
            'B_spike_time': B_spike_time,
            'actual_delta': actual_delta,
            'v_history_A': np.array(v_history_A),
            'v_history_B': np.array(v_history_B),
            't_history': np.array(t_history)
        }
    
    def decode(self, A_spike_time, B_spike_time):
        """
        2つのニューロンの発火時刻から元の値を復元
        
        Parameters:
        -----------
        A_spike_time : float
            ニューロンAの発火時刻 [ms]
        B_spike_time : float  
            ニューロンBの発火時刻 [ms]
        
        Returns:
        --------
        int : 復元された値
        """
        if A_spike_time is None or B_spike_time is None:
            raise ValueError("発火が検出されませんでした")
        
        # 時間差を計算して、最も近い整数に丸める
        delta = int(round(B_spike_time - A_spike_time))
        
        # ルックアップテーブルで値を取得
        if delta not in self.delta_to_value:
            raise ValueError(f"時間差が範囲外です。delta: {delta}ms")
        
        return self.delta_to_value[delta]
    
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
        
        print("\n=== LIFニューロンでのエンコード・デコードテスト ===")
        
        for value in test_values:
            # エンコードしてシミュレーション実行
            sim_result = self.encode_and_simulate(value)
            
            # デコード（発火時刻から値を復元）
            try:
                decoded_value = self.decode(
                    sim_result['A_spike_time'],
                    sim_result['B_spike_time']
                )
                success = (value == decoded_value)
            except ValueError as e:
                decoded_value = None
                success = False
                print(f"✗ エラー: {e}")
            
            # 結果を記録
            result = {
                'input': value,
                'target_delta': sim_result['target_delta'],
                'A_spike_time': sim_result['A_spike_time'],
                'B_spike_time': sim_result['B_spike_time'],
                'actual_delta': sim_result['actual_delta'],
                'decoded': decoded_value,
                'success': success,
                'sim_result': sim_result
            }
            results.append(result)
            
            # 結果を表示
            status = "✓" if success else "✗"
            if success:
                print(f"{status} 入力: {value:2d} -> 目標Δt: {sim_result['target_delta']:3.0f}ms, "
                      f"実際Δt: {sim_result['actual_delta']:5.2f}ms -> 復元: {decoded_value:2d}")
            else:
                print(f"{status} 入力: {value:2d} -> デコード失敗")
        
        # 成功率を計算
        success_count = sum(r['success'] for r in results)
        success_rate = success_count / len(results) * 100
        print(f"\n成功率: {success_rate:.1f}% ({success_count}/{len(results)})")
        
        return results
    
    def visualize_simulation(self, results, num_examples=5):
        """
        シミュレーション結果を可視化
        
        各値について：
        - 膜電位の時間変化
        - 発火タイミング（縦線）
        - 時間差の比較（目標 vs 実際）
        
        Parameters:
        -----------
        results : list
            test_encoding()の結果
        num_examples : int
            表示する例の数
        """
        # 表示する結果を選択（均等に分散）
        indices = np.linspace(0, len(results)-1, min(num_examples, len(results)), dtype=int)
        selected_results = [results[i] for i in indices]
        
        # グラフを作成（縦に並べる）
        fig, axes = plt.subplots(len(selected_results), 1, figsize=(14, 3*len(selected_results)))
        
        # 1つだけの場合はリストにする
        if len(selected_results) == 1:
            axes = [axes]
        
        for idx, (ax, result) in enumerate(zip(axes, selected_results)):
            sim = result['sim_result']
            
            # 膜電位をプロット
            ax.plot(sim['t_history'], sim['v_history_A'], 
                   label='ニューロンA', linewidth=2, alpha=0.7)
            ax.plot(sim['t_history'], sim['v_history_B'], 
                   label='ニューロンB', linewidth=2, alpha=0.7)
            
            # 発火閾値を点線で表示
            ax.axhline(y=self.neuron_A.v_thresh, color='r', 
                      linestyle='--', alpha=0.5, label='発火閾値')
            
            # 発火時刻に縦線を引く
            if sim['A_spike_time'] is not None:
                ax.axvline(x=sim['A_spike_time'], color='blue', 
                          linestyle=':', alpha=0.7, linewidth=2)
            if sim['B_spike_time'] is not None:
                ax.axvline(x=sim['B_spike_time'], color='orange', 
                          linestyle=':', alpha=0.7, linewidth=2)
            
            # タイトルと軸ラベル
            status = "✓" if result['success'] else "✗"
            
            # decoded が None の場合の処理を追加
            decoded_str = str(result['decoded']) if result['decoded'] is not None else "失敗"
            actual_delta_str = f"{result['actual_delta']:.2f}" if result['actual_delta'] is not None else "N/A"
            
            ax.set_title(
                f"{status} 入力値: {result['input']} | "
                f"目標Δt: {result['target_delta']:.1f}ms | "
                f"実際Δt: {actual_delta_str}ms | "
                f"復元値: {decoded_str}", 
                fontsize=11
            )
            ax.set_xlabel('時刻 (ms)', fontsize=10)
            ax.set_ylabel('膜電位 (mV)', fontsize=10)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig


# 実行例
if __name__ == "__main__":
    # システムを作成
    # dt=0.1ms（高精度）、sim_time=100ms（十分な時間）
    system = TwoNeuronLIFPhaseCoding(dt=0.1, sim_time=100.0)
    
    # テストする値（代表的な値を選択）
    test_values = [0, 5, 10, 15, 20]
    
    # エンコード・シミュレーション・デコードを実行
    results = system.test_encoding(test_values)
    
    # 結果を可視化（失敗していても表示）
    system.visualize_simulation(results, num_examples=5)
    
    # より詳細なテスト（全ての値）
    print("\n=== 全ての値(0〜20)でテスト ===")
    all_values = list(range(21))
    all_results = system.test_encoding(all_values)
    
    print("\n=== 時間差の精度を確認 ===")
    for r in all_results:
        if r['success']:
            error = abs(r['actual_delta'] - r['target_delta'])
            print(f"入力: {r['input']:2d} | 目標: {r['target_delta']:3.0f}ms | "
                  f"実際: {r['actual_delta']:6.2f}ms | 誤差: {error:.3f}ms")