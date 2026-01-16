import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

class TwoNeuronPhaseCoding:
    """
    2つのニューロンの発火時間差で情報を符号化・復号化するクラス
    
    ニューロンAは常に0msで発火
    ニューロンBは入力値に応じて-10ms〜+10msで発火
    時間差 = B_time - A_time
    """
    
    def __init__(self):
        # ルックアップテーブル: 入力値 -> 時間差(ms)
        # 0〜20の整数を-10〜+10msの時間差にマッピング
        self.value_to_delta = {}
        self.delta_to_value = {}
        
        for value in range(21):  # 0〜20
            delta = value - 10  # -10〜+10に変換
            self.value_to_delta[value] = delta
            self.delta_to_value[delta] = value
        
        print("ルックアップテーブル作成完了")
        print(f"符号化可能な値の範囲: 0〜20 (21通り)")
        print(f"時間差の範囲: -10ms〜+10ms")
    
    def encode(self, value):
        """
        入力値を2つのニューロンの発火タイミングに変換
        
        Parameters:
        -----------
        value : int
            符号化したい値（0〜20）
        
        Returns:
        --------
        tuple : (neuron_A_time, neuron_B_time)
        """
        if value not in self.value_to_delta:
            raise ValueError(f"入力値は0〜20の範囲で指定してください。入力値: {value}")
        
        neuron_A_time = 0  # ニューロンAは常に0ms
        delta = self.value_to_delta[value]
        neuron_B_time = delta  # ニューロンBは時間差分ずれる
        
        return neuron_A_time, neuron_B_time
    
    def decode(self, neuron_A_time, neuron_B_time):
        """
        2つのニューロンの発火タイミングから元の値を復元
        
        Parameters:
        -----------
        neuron_A_time : float
            ニューロンAの発火時刻
        neuron_B_time : float
            ニューロンBの発火時刻
        
        Returns:
        --------
        int : 復元された値
        """
        delta = int(round(neuron_B_time - neuron_A_time))
        
        if delta not in self.delta_to_value:
            raise ValueError(f"時間差が範囲外です。delta: {delta}ms")
        
        return self.delta_to_value[delta]
    
    def test_encoding(self, test_values):
        """
        複数の値でエンコード・デコードをテスト
        
        Parameters:
        -----------
        test_values : list
            テストする値のリスト
        """
        results = []
        
        print("\n=== エンコード・デコードテスト ===")
        for value in test_values:
            # エンコード
            A_time, B_time = self.encode(value)
            
            # デコード
            decoded_value = self.decode(A_time, B_time)
            
            # 結果を記録
            success = (value == decoded_value)
            results.append({
                'input': value,
                'A_time': A_time,
                'B_time': B_time,
                'delta': B_time - A_time,
                'decoded': decoded_value,
                'success': success
            })
            
            status = "✓" if success else "✗"
            print(f"{status} 入力: {value:2d} -> A: {A_time:3.0f}ms, B: {B_time:3.0f}ms (Δ={B_time-A_time:3.0f}ms) -> 復元: {decoded_value:2d}")
        
        # 成功率を計算
        success_rate = sum(r['success'] for r in results) / len(results) * 100
        print(f"\n成功率: {success_rate:.1f}% ({sum(r['success'] for r in results)}/{len(results)})")
        
        return results
    
    def visualize_encoding(self, test_values):
        """
        エンコード結果を可視化
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 各値についてエンコード
        A_times = []
        B_times = []
        deltas = []
        
        for value in test_values:
            A_time, B_time = self.encode(value)
            A_times.append(A_time)
            B_times.append(B_time)
            deltas.append(B_time - A_time)
        
        # グラフ1: 発火タイミング
        ax1.scatter(test_values, A_times, label='ニューロンA', s=100, marker='o', alpha=0.7)
        ax1.scatter(test_values, B_times, label='ニューロンB', s=100, marker='s', alpha=0.7)
        ax1.set_xlabel('入力値', fontsize=12)
        ax1.set_ylabel('発火時刻 (ms)', fontsize=12)
        ax1.set_title('入力値と各ニューロンの発火タイミング', fontsize=14)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # グラフ2: 時間差
        ax2.plot(test_values, deltas, marker='o', linewidth=2, markersize=8)
        ax2.set_xlabel('入力値', fontsize=12)
        ax2.set_ylabel('時間差 Δt = B - A (ms)', fontsize=12)
        ax2.set_title('入力値と時間差の関係（符号化関数）', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Δt=0 (同時発火)')
        ax2.legend(fontsize=11)
        
        plt.tight_layout()
        plt.show()
        
        return fig

# 実行例
if __name__ == "__main__":
    # インスタンス作成
    coder = TwoNeuronPhaseCoding()
    
    # テストする値
    test_values = [0, 5, 10, 15, 20]  # 代表的な値
    
    # エンコード・デコードテスト
    results = coder.test_encoding(test_values)
    
    # 全ての値でテスト
    print("\n=== 全ての値(0〜20)でテスト ===")
    all_values = list(range(21))
    all_results = coder.test_encoding(all_values)
    
    # 可視化
    coder.visualize_encoding(all_values)
    
    print("\n=== 情報量の計算 ===")
    num_states = 21
    bits = np.log2(num_states)
    print(f"表現可能な状態数: {num_states}通り")
    print(f"情報量: {bits:.2f}ビット")
    print(f"従来の2ニューロン(発火/非発火): 2ビット")
    print(f"情報密度の向上: {bits/2:.2f}倍")