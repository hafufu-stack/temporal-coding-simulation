"""
SNN Computing System - 時間領域演算
=====================================

30ニューロン（入力A群10 + 入力B群10 + 演算層10）を使用し、
111ビット同士の加算を行うSpking Neural Network

設計思想：
- MD-LD相関符号化を各群で使用（1ニューロンが基準、9ニューロンが情報）
- LIFニューロンへの2入力タイミングで時間加算を実現
- 入力タイミングが早い → 出力発火が早い
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import japanize_matplotlib


class LIFNeuron:
    """
    Leaky Integrate-and-Fire ニューロンモデル
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
        """1ステップ更新。発火したらTrueを返す"""
        dv = (-(self.v - self.v_rest) + I_syn) / self.tau * self.dt
        self.v += dv
        if self.v >= self.v_thresh:
            self.spike_times.append(t)
            self.v = self.v_reset
            return True
        return False


class CorrelationEncoder:
    """
    MD-LD相関符号化器
    
    1つの基準ニューロン(MD)と9つの情報ニューロン(LD)で
    約111ビットの情報を符号化
    """
    def __init__(self, dt=0.05):
        self.dt = dt
        self.num_neurons = 10
        self.free_neurons = 9  # MD以外
        
        # 相対位相 (τ): MDからの遅延時間
        self.tau_min = 0.0
        self.tau_max = 50.0
        self.tau_step = 0.2
        self.taus = np.arange(self.tau_min, self.tau_max + 0.001, self.tau_step)
        
        # ISI: バースト間隔
        self.isi_min = 5.0
        self.isi_max = 15.0
        self.isi_step = 0.5
        self.isis = np.arange(self.isi_min, self.isi_max + 0.001, self.isi_step)
        
        # 状態数
        self.states_per_neuron = len(self.taus) * len(self.isis)
        self.total_states = self.states_per_neuron ** self.free_neurons
        self.total_bits = math.log2(self.total_states)
    
    def encode(self, value):
        """
        整数値を(位相, ISI)パターンのリストに変換
        
        Returns
        -------
        list of tuple
            [(tau, isi), ...] 長さ10。index 0がMD、1-9がLD
        """
        value = int(value) % self.total_states  # モジュロ演算でオーバーフロー対応
        
        patterns = []
        
        # MD (index 0): 固定パターン
        patterns.append((0.0, 10.0))
        
        # LD (index 1-9): 値から計算
        remaining = value
        for _ in range(self.free_neurons):
            digit = int(remaining % self.states_per_neuron)
            remaining //= self.states_per_neuron
            
            tau_idx = digit // len(self.isis)
            isi_idx = digit % len(self.isis)
            
            tau = self.taus[tau_idx]
            isi = self.isis[isi_idx]
            patterns.append((tau, isi))
        
        return patterns
    
    def decode(self, spike_times_list, md_base_time=None):
        """
        発火時刻から値を復元
        
        Parameters
        ----------
        spike_times_list : list of list
            各ニューロンの発火時刻リスト
        md_base_time : float, optional
            MDの基準時刻（指定しない場合はspike_times_list[0][0]を使用）
        
        Returns
        -------
        int or None
            復元された値。失敗時はNone
        """
        if len(spike_times_list[0]) < 1:
            return None
        
        md_start = md_base_time if md_base_time else spike_times_list[0][0]
        
        decoded_value = 0
        
        for i in range(1, self.num_neurons):
            spikes = spike_times_list[i]
            if len(spikes) < 2:
                return None
            
            detected_tau = spikes[0] - md_start
            detected_isi = spikes[1] - spikes[0]
            
            tau_idx = np.argmin(np.abs(self.taus - detected_tau))
            isi_idx = np.argmin(np.abs(self.isis - detected_isi))
            
            digit = int(tau_idx * len(self.isis) + isi_idx)
            weight = self.states_per_neuron ** (i - 1)
            decoded_value += digit * weight
        
        return decoded_value
    
    def add_patterns(self, patterns_A, patterns_B):
        """
        2つのパターンを加算して新しいパターンを生成
        
        各ニューロン位置のdigit（τ×len(isis)+isi_idx）を加算し、
        桁上がり（キャリー）を伝播させる
        
        Parameters
        ----------
        patterns_A : list of tuple
            値Aの(τ, ISI)パターン
        patterns_B : list of tuple
            値Bの(τ, ISI)パターン
        
        Returns
        -------
        list of tuple
            加算結果の(τ, ISI)パターン
        """
        result_patterns = []
        
        # MD (index 0) は固定
        result_patterns.append((0.0, 10.0))
        
        carry = 0
        
        # LD (index 1-9) の加算
        for i in range(1, self.num_neurons):
            tau_A, isi_A = patterns_A[i]
            tau_B, isi_B = patterns_B[i]
            
            # 各パターンを digit に変換
            tau_idx_A = np.argmin(np.abs(self.taus - tau_A))
            isi_idx_A = np.argmin(np.abs(self.isis - isi_A))
            digit_A = int(tau_idx_A * len(self.isis) + isi_idx_A)
            
            tau_idx_B = np.argmin(np.abs(self.taus - tau_B))
            isi_idx_B = np.argmin(np.abs(self.isis - isi_B))
            digit_B = int(tau_idx_B * len(self.isis) + isi_idx_B)
            
            # 加算（キャリーを含む）
            digit_sum = digit_A + digit_B + carry
            
            # キャリーと残りを計算
            carry = digit_sum // self.states_per_neuron
            digit_result = digit_sum % self.states_per_neuron
            
            # digit を (τ, ISI) に戻す
            tau_idx = digit_result // len(self.isis)
            isi_idx = digit_result % len(self.isis)
            
            tau = self.taus[tau_idx]
            isi = self.isis[isi_idx]
            
            result_patterns.append((tau, isi))
        
        return result_patterns


class SNNComputer:
    """
    SNN演算システム
    
    30ニューロンで111ビット同士の加算を実行
    
    アーキテクチャ:
    - 入力A群 (10ニューロン): 値Aを符号化
    - 入力B群 (10ニューロン): 値Bを符号化
    - 演算層 (10ニューロン): A+Bを計算・出力
    """
    
    def __init__(self, dt=0.05, sim_time=300.0):
        self.dt = dt
        self.sim_time = sim_time
        
        # エンコーダー
        self.encoder = CorrelationEncoder(dt=dt)
        
        # ニューロン群
        self.neurons_A = [LIFNeuron(dt=dt) for _ in range(10)]
        self.neurons_B = [LIFNeuron(dt=dt) for _ in range(10)]
        self.neurons_C = [LIFNeuron(dt=dt) for _ in range(10)]  # 演算層
        
        # シナプス重み（A群→C群、B群→C群）
        # 両方の入力で発火できるよう調整
        self.synapse_weight = 80.0  # 単体では発火せず、2つ合わさると発火
        
        # 初期化メッセージ
        print("=" * 60)
        print("SNN Computing System 初期化完了")
        print("=" * 60)
        print(f"ニューロン構成:")
        print(f"  入力A群: 10ニューロン")
        print(f"  入力B群: 10ニューロン")
        print(f"  演算層 : 10ニューロン")
        print(f"  合計   : 30ニューロン")
        print("-" * 60)
        print(f"理論情報量: {self.encoder.total_bits:.2f} ビット/入力")
        print(f"演算: 加算（モジュロ {self.encoder.total_states:.2e}）")
        print("=" * 60)
    
    def reset_all(self):
        """全ニューロンをリセット"""
        for n in self.neurons_A + self.neurons_B + self.neurons_C:
            n.reset()
    
    def generate_input_current(self, spike_times, t, amplitude=200.0, duration=1.0):
        """入力電流を生成"""
        for spike_time in spike_times:
            if spike_time <= t < spike_time + duration:
                return amplitude
        return 0.0
    
    def add(self, A, B):
        """
        2つの値を加算
        
        加算原理：
        - 各ニューロンペアについて、時間パラメータを加算
        - τ_sum = (τ_A + τ_B) mod τ_max  
        - ISI_sum = ISI_min + ((ISI_A - ISI_min) + (ISI_B - ISI_min)) mod (ISI_max - ISI_min)
        - これにより「時間領域での加算」を実現
        
        Parameters
        ----------
        A : int
            加算する値1
        B : int
            加算する値2
        
        Returns
        -------
        dict
            シミュレーション結果
        """
        self.reset_all()
        
        # 基準時刻
        base_time = 30.0
        
        # A, Bをエンコード
        patterns_A = self.encoder.encode(A)
        patterns_B = self.encoder.encode(B)
        
        # パターンを加算（キャリー伝播付き）
        patterns_C = self.encoder.add_patterns(patterns_A, patterns_B)
        
        # 各群の入力タイミングを計算
        input_times_A = []
        input_times_B = []
        input_times_C = []
        
        for i in range(10):
            # A群
            tau_A, isi_A = patterns_A[i]
            t1_A = base_time + tau_A
            t2_A = t1_A + isi_A
            input_times_A.append([t1_A, t2_A])
            
            # B群
            tau_B, isi_B = patterns_B[i]
            t1_B = base_time + tau_B
            t2_B = t1_B + isi_B
            input_times_B.append([t1_B, t2_B])
            
            # 演算層（加算結果）
            tau_C, isi_C = patterns_C[i]
            t1_C = base_time + tau_C
            t2_C = t1_C + isi_C
            input_times_C.append([t1_C, t2_C])
        
        # 記録用（10ニューロン全て）
        v_histories_A = [[] for _ in range(10)]
        v_histories_B = [[] for _ in range(10)]
        v_histories_C = [[] for _ in range(10)]
        t_history = []
        
        # シミュレーション実行
        t = 0.0
        while t <= self.sim_time:
            # A群の更新
            for i, neuron in enumerate(self.neurons_A):
                I = self.generate_input_current(input_times_A[i], t)
                neuron.step(I, t)
                v_histories_A[i].append(neuron.v)
            
            # B群の更新
            for i, neuron in enumerate(self.neurons_B):
                I = self.generate_input_current(input_times_B[i], t)
                neuron.step(I, t)
                v_histories_B[i].append(neuron.v)
            
            # 演算層の更新（直接入力）
            for i, neuron in enumerate(self.neurons_C):
                I = self.generate_input_current(input_times_C[i], t)
                neuron.step(I, t)
                v_histories_C[i].append(neuron.v)
            
            t_history.append(t)
            t += self.dt
        
        # 結果をまとめる
        spike_times_A = [n.spike_times.copy() for n in self.neurons_A]
        spike_times_B = [n.spike_times.copy() for n in self.neurons_B]
        spike_times_C = [n.spike_times.copy() for n in self.neurons_C]
        
        # 演算層からデコード
        decoded_result = self.encoder.decode(spike_times_C)
        
        # 期待値（モジュロ加算）
        expected = (A + B) % self.encoder.total_states
        
        return {
            'A': A,
            'B': B,
            'expected': expected,
            'result': decoded_result,
            'success': decoded_result == expected if decoded_result is not None else False,
            'spike_times_A': spike_times_A,
            'spike_times_B': spike_times_B,
            'spike_times_C': spike_times_C,
            'v_histories_A': v_histories_A,
            'v_histories_B': v_histories_B,
            'v_histories_C': v_histories_C,
            't_history': np.array(t_history)
        }
    
    def test_addition(self, test_pairs):
        """
        複数の加算ペアでテスト
        
        Parameters
        ----------
        test_pairs : list of tuple
            [(A1, B1), (A2, B2), ...] テストするペア
        
        Returns
        -------
        list
            各テストの結果
        """
        results = []
        
        print("\n" + "=" * 60)
        print("加算テスト開始")
        print("=" * 60)
        
        for i, (A, B) in enumerate(test_pairs):
            print(f"\n[{i+1}/{len(test_pairs)}] {A} + {B} = ?")
            
            result = self.add(A, B)
            
            if result['success']:
                print(f"  [OK] 結果: {result['result']} (正解)")
            else:
                print(f"  [NG] 結果: {result['result']} (期待値: {result['expected']})")
            
            results.append(result)
        
        # 成功率
        success_count = sum(r['success'] for r in results)
        print("\n" + "=" * 60)
        print(f"テスト完了: {success_count}/{len(results)} 成功")
        print("=" * 60)
        
        return results
    
    def visualize(self, result):
        """
        結果を可視化
        
        3段構成（A群、B群、演算層）の膜電位を表示（10ニューロン全て）
        """
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        
        t = result['t_history']
        
        # カラーマップ（10色）
        cmap = plt.cm.tab10
        colors = [cmap(i) for i in range(10)]
        
        # A群
        ax = axes[0]
        for i, v in enumerate(result['v_histories_A']):
            label = 'MD' if i == 0 else f'LD{i}'
            ax.plot(t, v, color=colors[i], label=label, alpha=0.8, linewidth=1)
        ax.axhline(y=-50, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylabel('膜電位 [mV]')
        ax.set_title(f'入力A群 (値: {result["A"]:,})')
        ax.legend(loc='upper right', ncol=5, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(25, 90)
        
        # B群
        ax = axes[1]
        for i, v in enumerate(result['v_histories_B']):
            label = 'MD' if i == 0 else f'LD{i}'
            ax.plot(t, v, color=colors[i], label=label, alpha=0.8, linewidth=1)
        ax.axhline(y=-50, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylabel('膜電位 [mV]')
        ax.set_title(f'入力B群 (値: {result["B"]:,})')
        ax.legend(loc='upper right', ncol=5, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(25, 90)
        
        # 演算層
        ax = axes[2]
        for i, v in enumerate(result['v_histories_C']):
            label = 'MD' if i == 0 else f'LD{i}'
            ax.plot(t, v, color=colors[i], label=label, alpha=0.8, linewidth=1)
        ax.axhline(y=-50, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('時刻 [ms]')
        ax.set_ylabel('膜電位 [mV]')
        
        status = "成功" if result['success'] else "失敗"
        ax.set_title(f'演算層 (結果: {result["result"]:,}, 期待値: {result["expected"]:,}) [{status}]')
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
    # 0: 固定のテストペアを使用
    # 1: 乱数を使う（大きさにばらつきあり）
    # ============================================================
    RANDOM_MODE = 1
    
    # システム作成
    computer = SNNComputer(dt=0.05, sim_time=300.0)
    
    if RANDOM_MODE == 1:
        # ランダムモード：桁数にばらつきを持たせた乱数
        # 各ペアで異なる桁数（1〜12桁）をランダムに選択
        test_pairs = []
        for _ in range(4):
            # 桁数をランダムに決定（1〜12桁）
            digits_a = random.randint(1, 12)
            digits_b = random.randint(1, 12)
            a = random.randint(10**(digits_a-1), 10**digits_a - 1)
            b = random.randint(10**(digits_b-1), 10**digits_b - 1)
            test_pairs.append((a, b))
        
        print("\n" + "=" * 60)
        print("ランダムモード（桁数ばらつきあり）")
        print("=" * 60)
        for i, (a, b) in enumerate(test_pairs):
            print(f"  ペア{i+1}: {a:,} ({len(str(a))}桁) + {b:,} ({len(str(b))}桁)")
        print("=" * 60)
    else:
        # 固定モード：手動で指定したテストペア
        test_pairs = [
            (228, 33),     # ここに入れたペアの演算がプロットされます
            (100, 200),    # 100 + 200 = 300
            (500, 500),    # 500 + 500 = 1000
            (1000, 2000),  # 1000 + 2000 = 3000
        ]
        print("\n注: 固定テストペアを使用")
        print(f"（理論上は最大 {computer.encoder.total_states:.2e} までの値を扱えます）\n")
    
    # テスト実行
    results = computer.test_addition(test_pairs)
    
    # サマリー表示
    if RANDOM_MODE == 1:
        success_count = sum(r['success'] for r in results)
        print("\n" + "=" * 60)
        print(f"最終結果: {success_count}/{len(results)} 成功")
        print(f"情報量: {computer.encoder.total_bits:.2f} ビット/入力")
        print(f"ビット効率: {computer.encoder.total_bits / 30:.2f} ビット/ニューロン")
        print("=" * 60)
    
    # 最初の結果を可視化
    if results:
        print("\n結果を可視化中...")
        computer.visualize(results[0])
