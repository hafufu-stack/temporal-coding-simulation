"""
SNN Comprypto System (High Performance / Numba Optimized)
=========================================================

Numba JITコンパイルを利用した高速化バージョン。
PythonのforループをC++レベルのマシン語にコンパイルして実行します。

Author: ろーる
"""

import numpy as np
import time
import hashlib
from numba import jit, float64, int64, uint8, void
import sys

# --- Numba Optimized Kernels ---

@jit(nopython=True, cache=True)
def neuron_step_jit(v, I_syn, dt, tau, v_rest, v_thresh, v_reset):
    """LIFニューロンの更新 (1個分)"""
    dv = (-(v - v_rest) + I_syn) / tau * dt
    v += dv
    spike = 0.0
    if v >= v_thresh:
        v = v_reset
        spike = 1.0
    return v, spike

@jit(nopython=True, cache=True)
def reservoir_step_predict_jit(
    neurons_v,           # 膜電位ベクトル (State)
    input_val,           # 入力値 (0-255)
    fire_rate,           # 発火率ベクトル (State)
    W_res, W_in, W_out,  # 重み行列
    num_neurons, dt, tau, v_rest, v_thresh, v_reset,
    learning_rate        # 学習率
):
    """
    リザーバの1ステップ実行 + 予測計算
    (クラスメソッドではなく、純粋な関数として実装してJITにかける)
    """
    # 1. 入力正規化
    u = (input_val / 127.5) - 1.0
    
    # 2. 電流計算 (行列演算はNumbaでも高速)
    # I = W_res @ fire_rate + W_in * u
    I_rec = W_res @ fire_rate
    # I_ext = W_in * u (ベクトル同士の積ではなく、定数倍)
    # W_inは (N, 1) なので flatten して使う
    I_total = I_rec + (W_in.flatten() * u)
    
    # 3. ニューロン更新ループ (これがPythonだと遅い)
    spikes = np.zeros(num_neurons, dtype=np.float64)
    for i in range(num_neurons):
        bias = 25.0
        v_new, s = neuron_step_jit(neurons_v[i], I_total[i] + bias, dt, tau, v_rest, v_thresh, v_reset)
        neurons_v[i] = v_new
        spikes[i] = s
        
    # 4. 状態更新 (EMA)
    fire_rate = 0.7 * fire_rate + 0.3 * spikes
    
    # 5. 予測
    y = W_out @ fire_rate
    pred_val = (y + 1.0) * 127.5
    
    # Clip (0-255)
    if pred_val < 0: pred_val = 0
    if pred_val > 255: pred_val = 255
    
    return int(pred_val), spikes

@jit(nopython=True, cache=True)
def online_learning_jit(
    W_out, fire_rate, target_val, alpha
):
    """読み出し層のLMS学習"""
    d = (target_val / 127.5) - 1.0
    y = W_out @ fire_rate
    error = d - y
    W_out += alpha * error * fire_rate

# --- Main Wrapper Class ---

class CompryptoReservoirFast:
    def __init__(self, key_seed, num_neurons=300, density=0.1, input_scale=40.0):
        np.random.seed(key_seed)
        
        self.num_neurons = num_neurons
        # Constants for Numba
        self.dt = 0.5
        self.tau = 20.0
        self.v_rest = -65.0
        self.v_thresh = -50.0
        self.v_reset = -70.0
        self.alpha = 0.005
        
        # Weights
        self.W_res = np.random.randn(num_neurons, num_neurons)
        rho = max(abs(np.linalg.eigvals(self.W_res)))
        self.W_res *= (1.4 / rho)
        mask = np.random.rand(num_neurons, num_neurons) < density
        self.W_res *= mask
        self.W_res = self.W_res.astype(np.float64) # Ensure Types
        
        self.W_in = (np.random.randn(num_neurons, 1) * input_scale).astype(np.float64)
        self.W_out = np.zeros(num_neurons, dtype=np.float64)
        
        # State
        self.neurons_v = np.full(num_neurons, self.v_rest, dtype=np.float64)
        self.fire_rate = np.zeros(num_neurons, dtype=np.float64)

    def get_keystream_byte(self):
        # Numba内でのHashlibは難しいので、ここはPython側でやる
        # (ここがボトルネックになるなら、簡易的なカオスハッシュをJITで作るのもあり)
        state_bytes = self.neurons_v.tobytes()
        h = hashlib.sha256(state_bytes).digest()
        return h[0]

class SNNCompryptorFast:
    def __init__(self, key_seed=2026):
        self.key_seed = key_seed
        
    def compress_encrypt(self, data):
        brain = CompryptoReservoirFast(self.key_seed)
        
        # Local variables optimize access speed
        W_res = brain.W_res
        W_in = brain.W_in
        W_out = brain.W_out
        neurons_v = brain.neurons_v
        fire_rate = brain.fire_rate
        
        # Constants
        N = brain.num_neurons
        dt, tau = brain.dt, brain.tau
        vr, vt, vreset = brain.v_rest, brain.v_thresh, brain.v_reset
        alpha = brain.alpha
        
        encrypted_residuals = bytearray(len(data))
        last_val = 0
        
        # Pre-compile check (Dummy run to trigger JIT)
        # 最初の1回はコンパイルが入るので遅いが、ベンチマーク前に一度回しておくと良いかも
        
        for i, val in enumerate(data):
            # 1. 予測 (JIT)
            # 戻り値をアンパックする際に型を意識
            pred, _ = reservoir_step_predict_jit(
                neurons_v, last_val, fire_rate,
                W_res, W_in, W_out,
                N, dt, tau, vr, vt, vreset, alpha
            )
            
            # 2. 残差計算 (Py)
            residual = (val - pred) % 256
            
            # 3. 鍵生成 (Py/Hash) -> これが重ければ後で軽量化対象
            # brain.get_keystream_byte()のインライン化
            # ここだけPython関数のオーバーヘッドがかかる
            state_bytes = neurons_v.tobytes()
            # SHA256 is fast enough usually, but call overhead accumulates
            key_byte = hashlib.sha256(state_bytes).digest()[0]
            
            # 4. 暗号化
            encrypted_residuals[i] = residual ^ key_byte
            
            # 5. 学習 (JIT)
            online_learning_jit(W_out, fire_rate, val, alpha)
            
            last_val = val
            
        return encrypted_residuals

    # decrypt_decompressも同様だが、今回はベンチマーク用なのでencryptのみで速度を見る
