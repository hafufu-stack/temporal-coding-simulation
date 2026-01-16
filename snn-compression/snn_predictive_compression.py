"""
SNN予測圧縮システム (SNN Predictive Compression)
================================================

SNNのリザーバ計算とオンライン学習を利用した次世代圧縮プロトタイプ。
「予測符号化 (Predictive Coding)」の原理に基づき、
未来のデータを予測し、予測誤差（Residual）のみを記録する。

主な機能:
1. NeuroPredictor: データの文脈を学習し、次のバイト値を予測
2. Online Learning: 処理しながらリアルタイムでSNNの結合荷重を更新
3. Residual Encoding: 予測誤差 = 実測値 - 予測値 を記録

"""

import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import zlib
import time

class LIFNeuron:
    """高速LIFニューロン"""
    def __init__(self, dt=0.5, tau=20.0):
        self.dt = dt
        self.tau = tau
        self.v = -65.0
        self.v_rest = -65.0
        self.v_thresh = -50.0
        self.v_reset = -70.0
    
    def step(self, I_syn):
        dv = (-(self.v - self.v_rest) + I_syn) / self.tau * self.dt
        self.v += dv
        if self.v >= self.v_thresh:
            self.v = self.v_reset
            return 1.0
        return 0.0

class NeuroPredictor:
    """
    SNNリザーバ + オンライン学習(LMS/RLS)による時系列予測器
    """
    def __init__(self, num_neurons=200, input_scale=50.0, alpha=0.01):
        # 圧縮・展開で同じSNNを生成するためにシードを固定
        # これにより、CompressorとDecompressorが「同じ脳」を持つことが保証される
        np.random.seed(42)
        
        self.num_neurons = num_neurons
        self.dt = 0.5
        
        # リザーバ層 (固定)
        # スペクトル半径を調整してEcho State Propertyを持たせる
        self.W_res = np.random.randn(num_neurons, num_neurons) 
        # スペクトル半径の調整
        rho = max(abs(np.linalg.eigvals(self.W_res)))
        self.W_res *= (1.2 / rho) # 少しカオス寄りに設定
        
        # 疎結合化
        mask = np.random.rand(num_neurons, num_neurons) < 0.1
        self.W_res *= mask
        
        # 入力層 (固定)
        # 入力1次元(バイト値) -> リザーバ
        self.W_in = np.random.randn(num_neurons, 1) * input_scale
        
        # 読み出し層 (学習対象)
        # リザーバ状態 -> 予測値
        self.W_out = np.zeros(num_neurons)
        
        # ニューロン群
        self.neurons = [LIFNeuron(dt=self.dt) for _ in range(num_neurons)]
        self.states = np.zeros(num_neurons) # 現在の発火状態(レート近似)
        
        # 学習率 (LMS用)
        self.alpha = alpha
        
        # 内部状態
        self.current_fire_rate = np.zeros(num_neurons) # 平滑化された発火率

    def predict(self, input_val):
        """
        現在の入力から「次の値」を予測する
        注意: 学習は別メソッドで行う（正解がわかってから）
        """
        # 入力値の正規化 (0-255 -> -1.0~1.0)
        norm_input = (input_val / 127.5) - 1.0
        
        # リザーバ更新
        # r(t) = tanh( W_res @ r(t-1) + W_in @ u(t) )
        # SNN版: I = W_res @ spikes + W_in @ input
        
        # 入力電流計算
        # リザーバ内部再帰入力 + 外部入力
        I_res = self.W_res @ self.current_fire_rate
        I_in = (self.W_in * norm_input).flatten()
        I_total = I_res + I_in
        
        # ニューロン発火更新
        s = np.zeros(self.num_neurons)
        for i, neuron in enumerate(self.neurons):
            # 予測の一貫性を保つため、ランダムノイズは除去
            # ノイズがあるとCompressorとDecompressorで状態が分岐してしまう
            noise = 0.0
            if neuron.step(I_total[i] + 30.0 + noise) > 0.0: # バイアス+30で自発発火
                s[i] = 1.0
        
        # 発火率の平滑化 (ローパスフィルタ) -> これを状態として使う
        self.current_fire_rate = 0.8 * self.current_fire_rate + 0.2 * s
        
        # 予測値の計算 (線形読み出し)
        # y = W_out @ state
        pred_norm = self.W_out @ self.current_fire_rate
        
        # 非正規化 (-1.0~1.0 -> 0-255)
        pred_val = (pred_norm + 1.0) * 127.5
        
        # クリップ
        return max(0, min(255, int(pred_val)))

    def train(self, target_val):
        """
        正解値を使って読み出し重みを更新 (Online Learning)
        LMS (Least Mean Squares) アルゴリズム
        """
        # 正規化
        target_norm = (target_val / 127.5) - 1.0
        
        # 現在の予測値（正規化後）
        pred_norm = self.W_out @ self.current_fire_rate
        
        # 誤差
        error = target_norm - pred_norm
        
        # 重み更新: W_new = W_old + alpha * error * input_state
        self.W_out += self.alpha * error * self.current_fire_rate

class SNNCompressor:
    def __init__(self):
        self.predictor = NeuroPredictor()
        
    def compress(self, data_bytes):
        """
        データを圧縮（予測誤差列に変換）
        """
        residuals = []
        predictions = []
        
        # 最初の値は予測しようがないのでそのまま記録（または0予測）
        last_val = 0
        
        for val in data_bytes:
            # 1. 前回の値をもとに今回を予測
            pred = self.predictor.predict(last_val)
            predictions.append(pred)
            
            # 2. 予測誤差を計算 (実測 - 予測)
            # Modulo 256を使うことで、常に0-255の範囲に収め、可逆性を保証する
            res = (val - pred) % 256
            residuals.append(res)
            
            # 3. 正解値でSNNを学習
            self.predictor.train(val)
            
            # 次のステップ用
            last_val = val
            
        return residuals, predictions

class SNNDecompressor:
    def __init__(self):
        self.predictor = NeuroPredictor()
        
    def decompress(self, residuals):
        """
        予測誤差列から元のデータを復元
        """
        restored_data = bytearray()
        last_val = 0
        
        for res in residuals:
            # 1. 予測 (Compressorと同じ状態のSNNが行うので同じ値が出るはず)
            pred = self.predictor.predict(last_val)
            
            # 2. 復元 (予測 + 誤差)
            # Modulo 256で元に戻す
            val = (pred + res) % 256
            
            restored_data.append(val)
            
            # 3. 学習 (Compressorと同じ更新を行う)
            self.predictor.train(val)
            
            last_val = val
            
        return restored_data

def entropy_score(data):
    """シャノン・エントロピー計算（情報のばらつき具合）"""
    if len(data) == 0: return 0
    counts = np.bincount(np.abs(data)) # 絶対値の分布を見る
    probs = counts[counts > 0] / len(data)
    return -np.sum(probs * np.log2(probs))

def run_experiment(name, data):
    print(f"\n[{name}] データサイズ: {len(data)} bytes")
    
    # 1. SNN圧縮
    compressor = SNNCompressor()
    start_time = time.time()
    residuals, predictions = compressor.compress(data)
    enc_time = time.time() - start_time
    
    # 残差の絶対値平均（これが小さいほど圧縮しやすい）
    mean_abs_res = np.mean(np.abs(residuals))
    
    # 2. SNN復元テスト
    decompressor = SNNDecompressor()
    restored = decompressor.decompress(residuals)
    
    is_success = (list(data) == list(restored))
    
    # 3. 比較 (Entropy)
    # 生データのエントロピー vs 残差のエントロピー
    # エントロピーが低いほど、Huffman符号などで小さく圧縮できる
    raw_entropy = entropy_score(np.frombuffer(data, dtype=np.uint8))
    res_entropy = entropy_score(np.array(residuals))
    
    # 4. 参考: ZIP圧縮
    zip_data = zlib.compress(data)
    zip_ratio = len(zip_data) / len(data) * 100
    
    print(f"  SNN処理時間: {enc_time:.2f}秒")
    print(f"  復元成功: {'✅ YES' if is_success else '❌ NO'}")
    print(f"  平均予測誤差: {mean_abs_res:.2f} (小さいほど良い)")
    print(f"  Entropy比較: {raw_entropy:.2f} -> {res_entropy:.2f} (小さいほど高圧縮)")
    print(f"  (参考) ZIP圧縮率: {zip_ratio:.1f}%")
    
    # 可視化
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(list(data)[:200], label='Original', alpha=0.7)
    plt.plot(predictions[:200], label='SNN Prediction', alpha=0.7)
    plt.title(f"SNN Prediction (First 200 bytes) - {name}")
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(residuals[:200], label='Residuals (Error)', color='red', alpha=0.7)
    plt.axhline(0, color='black', alpha=0.5)
    plt.title("Prediction Residuals (Target for Compression)")
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    print("SNN Predictive Compression Demo")
    print("===============================")
    
    # テストデータ1: サイン波（規則的）
    t = np.linspace(0, 4*np.pi, 500)
    wave = (np.sin(t) + 1) * 100 + 20 # 0-255におさめる
    data_wave = bytearray(wave.astype(np.uint8))
    
    run_experiment("Sine Wave Data", data_wave)
    
    # テストデータ2: テキストデータ（少し複雑）
    text = "SNN compression uses predictive coding principles inspired by the brain. " * 20
    data_text = text.encode('utf-8')
    
    run_experiment("Text Data", data_text)

if __name__ == "__main__":
    main()
