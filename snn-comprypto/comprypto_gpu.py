"""
SNN Comprypto System (GPU Accelerated / CuPy)
=============================================

CuPyを使用したGPU高速化バージョン。
RTX 5080のCUDAコアで行列演算を並列実行します。

Author: ろーる
"""

import cupy as cp
import numpy as np
import time
import hashlib

class CompryptoReservoirGPU:
    """
    GPU上で動作するニューロ・カオス・リザーバ
    """
    def __init__(self, key_seed, num_neurons=300, density=0.1, input_scale=40.0):
        # NumPyでシード設定してから重みを生成し、GPUに転送
        np.random.seed(key_seed)
        
        self.num_neurons = num_neurons
        self.dt = 0.5
        self.tau = 20.0
        self.v_rest = -65.0
        self.v_thresh = -50.0
        self.v_reset = -70.0
        self.alpha = 0.005
        
        # 重み行列をGPUに転送
        W_res_np = np.random.randn(num_neurons, num_neurons).astype(np.float32)
        rho = max(abs(np.linalg.eigvals(W_res_np)))
        W_res_np *= (1.4 / rho)
        mask = np.random.rand(num_neurons, num_neurons) < density
        W_res_np *= mask
        self.W_res = cp.asarray(W_res_np)
        
        W_in_np = (np.random.randn(num_neurons) * input_scale).astype(np.float32)
        self.W_in = cp.asarray(W_in_np)
        
        self.W_out = cp.zeros(num_neurons, dtype=cp.float32)
        
        # ニューロン状態をGPU上に保持
        self.v = cp.full(num_neurons, self.v_rest, dtype=cp.float32)
        self.fire_rate = cp.zeros(num_neurons, dtype=cp.float32)
        
    def step_predict(self, input_val):
        """
        1ステップ実行（全てGPU上で計算）
        """
        # 入力正規化
        u = (input_val / 127.5) - 1.0
        
        # 電流計算（GPU行列演算）
        I_rec = self.W_res @ self.fire_rate
        I_ext = self.W_in * u
        I_total = I_rec + I_ext + 25.0  # バイアス
        
        # ニューロン更新（ベクトル化 - GPU並列）
        dv = (-(self.v - self.v_rest) + I_total) / self.tau * self.dt
        self.v += dv
        
        # スパイク判定（GPU並列）
        spikes = (self.v >= self.v_thresh).astype(cp.float32)
        self.v = cp.where(self.v >= self.v_thresh, self.v_reset, self.v)
        
        # 発火率更新
        self.fire_rate = 0.7 * self.fire_rate + 0.3 * spikes
        
        # 予測値計算
        y = float(self.W_out @ self.fire_rate)
        pred_val = (y + 1.0) * 127.5
        
        return max(0, min(255, int(pred_val)))
    
    def train(self, target_val):
        """オンライン学習（GPU上で実行）"""
        d = (target_val / 127.5) - 1.0
        y = self.W_out @ self.fire_rate
        error = d - y
        self.W_out += self.alpha * error * self.fire_rate
        
    def get_keystream_byte(self):
        """膜電位からカオス鍵を生成"""
        # GPU -> CPU転送してハッシュ計算
        v_cpu = cp.asnumpy(self.v)
        h = hashlib.sha256(v_cpu.tobytes()).digest()
        return h[0]

class SNNCompryptorGPU:
    """
    GPU高速化版のSNN圧縮・暗号化エンジン
    """
    def __init__(self, key_seed=2026):
        self.key_seed = key_seed
        
    def compress_encrypt(self, data):
        """
        GPUを使って圧縮・暗号化
        """
        brain = CompryptoReservoirGPU(self.key_seed)
        
        encrypted_residuals = bytearray(len(data))
        last_val = 0
        
        for i, val in enumerate(data):
            # 1. 予測（GPU）
            pred = brain.step_predict(last_val)
            
            # 2. 残差計算
            residual = (val - pred) % 256
            
            # 3. カオス鍵生成
            key_byte = brain.get_keystream_byte()
            
            # 4. 暗号化
            encrypted_residuals[i] = residual ^ key_byte
            
            # 5. 学習（GPU）
            brain.train(val)
            
            last_val = val
            
        return encrypted_residuals
    
    def decrypt_decompress(self, encrypted_data):
        """
        GPUを使って復号・展開
        """
        brain = CompryptoReservoirGPU(self.key_seed)
        
        restored_data = bytearray(len(encrypted_data))
        last_val = 0
        
        for i, cipher_byte in enumerate(encrypted_data):
            # 1. 予測
            pred = brain.step_predict(last_val)
            
            # 2. カオス鍵生成
            key_byte = brain.get_keystream_byte()
            
            # 3. 復号
            residual = cipher_byte ^ key_byte
            
            # 4. 展開
            val = (pred + residual) % 256
            restored_data[i] = val
            
            # 5. 学習
            brain.train(val)
            
            last_val = val
            
        return restored_data

def run_gpu_benchmark():
    print("=== SNN Comprypto GPU ベンチマーク ===")
    print(f"CuPy Version: {cp.__version__}")
    print(f"GPU Available: {cp.cuda.is_available()}")
    
    # テストデータ
    print("\n[テスト: 正弦波データ 5000バイト]")
    t = np.linspace(0, 100*np.pi, 5000)
    wave = (np.sin(t) * 100 + 128).astype(np.uint8)
    data = bytearray(wave)
    
    # ウォームアップ（JIT的な初回コンパイル）
    print("ウォームアップ中...")
    warmup_sys = SNNCompryptorGPU(key_seed=1)
    warmup_sys.compress_encrypt(b'\x00' * 100)
    
    # 本番計測
    print("本番計測中...")
    sys = SNNCompryptorGPU(key_seed=12345)
    
    start = time.time()
    encrypted = sys.compress_encrypt(data)
    enc_time = time.time() - start
    
    print(f"暗号化時間: {enc_time*1000:.2f} ms")
    print(f"スループット: {len(data)/enc_time/1024:.2f} KB/s")
    
    # 復号テスト
    sys2 = SNNCompryptorGPU(key_seed=12345)
    restored = sys2.decrypt_decompress(encrypted)
    
    is_ok = (data == restored)
    print(f"復元チェック: {'✅ OK' if is_ok else '❌ NG'}")
    
    return enc_time

if __name__ == "__main__":
    run_gpu_benchmark()
