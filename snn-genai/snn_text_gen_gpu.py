"""
SNN Text Generator (PyTorch GPU Version)
========================================

PyTorchを使用したGPU加速型SNN生成AI（Nano-LLM）。
Liquid State Machine (LSM) アーキテクチャを採用。
RTX 5080などの最新GPUでも動作するPyTorchバックエンドを使用。

Features:
- PyTorch (CUDA) による数千ニューロンのリアルタイムシミュレーション
- RLS (Recursive Least Squares) アルゴリズムによる高速オンライン学習
- JITコンパイル (torch.compile) によるさらなる高速化（オプション）

"""

import torch
import torch.nn as nn
import numpy as np
import time
import sys

# デバイス設定
# RTX 5080互換性チェック
try:
    if torch.cuda.is_available():
        # 実際に計算してカーネルがロードできるか確認
        dummy = torch.tensor([1.0], device="cuda")
        dummy + 1.0
        DEVICE = torch.device("cuda")
        print(f"[{__file__}] Using Device: {DEVICE} (Name: {torch.cuda.get_device_name(0)})")
    else:
        raise RuntimeError("CUDA not available")
except RuntimeError as e:
    print(f"[{__file__}] WARNING: GPU initialization failed ({e}).")
    print(f"[{__file__}] RTX 5080 binary compatibility issue detected. Falling back to CPU.")
    DEVICE = torch.device("cpu")


class Torch_LIF_Reservoir(nn.Module):
    """
    PyTorchで動作する高速LIFリザーバ
    """
    def __init__(self, num_neurons=2048, spectral_radius=1.5, density=0.1, dt=1.0):
        super().__init__()
        
        # CPUモードならニューロン数を減らす
        if DEVICE.type == 'cpu' and num_neurons > 512:
            print(f"[{__file__}] reducing neurons from {num_neurons} to 512 for CPU efficiency")
            num_neurons = 512
            
        self.N = num_neurons
        self.dt = dt
        self.tau = 20.0
        
        # 状態変数
        self.register_buffer('v', torch.full((self.N,), -65.0, device=DEVICE))
        self.register_buffer('filtered_spikes', torch.zeros(self.N, device=DEVICE))
        
        # 定数
        self.v_rest = -65.0
        self.v_thresh = -50.0
        self.v_reset = -70.0
        self.alpha = float(dt / self.tau)
        
        # 重み行列 (Recurrent)
        # 疎行列の生成などはCPUでやってからGPUへ転送
        W = torch.randn(self.N, self.N)
        mask = torch.rand(self.N, self.N) < density
        W *= mask
        # スペクトル半径調整
        eigenvalues = torch.linalg.eigvals(W)
        rho = torch.max(torch.abs(eigenvalues))
        W *= (spectral_radius / rho)
        
        self.register_buffer('W_rec', W.to(DEVICE))
        
        # 入力重み
        self.input_dim = 64
        self.register_buffer('W_in', (torch.randn(self.N, self.input_dim) * 5.0).to(DEVICE))

    def forward(self, input_vec):
        """
        1ステップ更新 (forwardだが勾配計算は不要なのでno_grad推奨)
        input_vec: (input_dim,)
        """
        # 入力電流: Recurrent + External
        I_rec = torch.mv(self.W_rec, self.filtered_spikes)
        I_ext = torch.mv(self.W_in, input_vec)
        I_total = I_rec + I_ext
        
        # 膜電位更新
        dv = (-(self.v - self.v_rest) + I_total) / self.tau * self.dt
        self.v += dv
        
        # 発火判定
        spikes = (self.v >= self.v_thresh).float()
        
        # リセット
        self.v = torch.where(spikes > 0, self.v_reset, self.v)
        
        # フィルタリング
        self.filtered_spikes = (1.0 - self.alpha) * self.filtered_spikes + self.alpha * spikes
        
        return self.filtered_spikes

class Torch_RLS_Readout(nn.Module):
    """
    PyTorchによるRLSオンライン学習層
    """
    def __init__(self, input_size, output_size, lambda_reg=1.0):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.lambda_reg = lambda_reg
        
        # 重み行列 (Output x Input)
        self.register_buffer('W_out', torch.zeros(output_size, input_size, device=DEVICE))
        
        # 逆相関行列 P (Input x Input)
        self.register_buffer('P', torch.eye(input_size, device=DEVICE) * 1.0)
        
    def forward(self, r):
        # 予測: y = W_out @ r
        return torch.mv(self.W_out, r)
    
    def train_step(self, r, target):
        """
        RLS学習ステップ
        target: (output_size,) one-hot vector
        """
        # 1. Pの更新
        # k = (P @ r) / (lambda + r.T @ P @ r)
        Pr = torch.mv(self.P, r)
        rPr = torch.dot(r, Pr)
        gain_k = Pr / (self.lambda_reg + rPr)
        
        # P = (P - k @ r.T @ P) / lambda
        outer = torch.outer(gain_k, Pr)
        self.P = (self.P - outer) / self.lambda_reg
        
        # 2. 誤差計算
        pred = torch.mv(self.W_out, r)
        err = target - pred
        
        # 3. 重み更新
        # W = W + e @ k.T
        self.W_out += torch.outer(err, gain_k)
        
        return torch.mean(torch.abs(err))

class SNNTextGenerator:
    def __init__(self):
        self.vocab = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?-_\n"
        self.char_to_id = {c: i for i, c in enumerate(self.vocab)}
        self.id_to_char = {i: c for i, c in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        
        self.input_dim = 64
        # EmbeddingもGPUへ
        self.embedding = torch.randn(self.vocab_size, self.input_dim, device=DEVICE)
        
        # Reservoir & Readout
        self.reservoir = Torch_LIF_Reservoir(
            num_neurons=2048, # RTX 5080ならもっと増やせるがまずはこれで
            density=0.1,
            spectral_radius=1.5
        ).to(DEVICE)
        
        self.readout = Torch_RLS_Readout(
            input_size=self.reservoir.N, # Dynnamic input size
            output_size=self.vocab_size
        ).to(DEVICE)

    def encode(self, char):
        idx = self.char_to_id.get(char, 0)
        return self.embedding[idx]

    def one_hot(self, char):
        vec = torch.zeros(self.vocab_size, device=DEVICE)
        idx = self.char_to_id.get(char, 0)
        vec[idx] = 1.0
        return vec
    
    def train_text(self, text, epochs=1):
        print(f"Start Training (Text len: {len(text)} chars)...")
        start_time = time.time()
        loss_hist = []
        
        # 学習ループでは勾配計算は不要（RLSで手動更新するため）
        with torch.no_grad():
            for epoch in range(epochs):
                for i in range(len(text) - 1):
                    char_in = text[i]
                    char_target = text[i+1]
                    
                    # GPU転送と計算
                    u = self.encode(char_in)
                    r = self.reservoir(u)
                    
                    target_vec = self.one_hot(char_target)
                    loss = self.readout.train_step(r, target_vec)
                    
                    if i % 100 == 0:
                        loss_hist.append(loss.item())
                        sys.stdout.write(f"\rEpoch {epoch+1}, Step {i}/{len(text)}, Loss: {loss:.4f}")
                        sys.stdout.flush()
        
        elapsed = time.time() - start_time
        print(f"\nTraining Finished. Time: {elapsed:.2f}s")
        return loss_hist

    def generate(self, seed_text, length=100, temperature=1.0):
        print(f"\nGenerating text with seed: '{seed_text}'")
        generated = seed_text
        
        with torch.no_grad():
            # Warmup
            for char in seed_text:
                u = self.encode(char)
                r = self.reservoir(u)
            
            current_char = seed_text[-1]
            
            for _ in range(length):
                u = self.encode(current_char)
                r = self.reservoir(u)
                
                # Predict
                logits = self.readout(r)
                
                # Temperature Scaling
                logits /= temperature
                probs = torch.softmax(logits, dim=0)
                
                # Sampling
                next_idx = torch.multinomial(probs, 1).item()
                next_char = self.id_to_char[next_idx]
                
                generated += next_char
                current_char = next_char
                
        return generated

def main():
    print("=== SNN Text Generator (PyTorch Nano-LLM) ===")
    
    # メモリ不足対策: ガベージコレクション
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    generator = SNNTextGenerator()
    
    # サンプルテキスト(英語)
    text = """
Alice was beginning to get very tired of sitting by her sister on the bank,
and of having nothing to do: once or twice she had peeped into the book her sister was reading,
but it had no pictures or conversations in it, 'and what is the use of a book,' thought Alice
'without pictures or conversation?'
So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid),
whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies,
when suddenly a White Rabbit with pink eyes ran close by her.
""".replace("\n", " ").strip()
    
    # データを長くする
    training_text = (text + " ") * 20 
    
    # 学習
    generator.train_text(training_text, epochs=2)
    
    # 生成テスト
    seed = "Alice was "
    generated = generator.generate(seed, length=200, temperature=0.5)
    
    print("-" * 40)
    print("Generated Result:")
    print(generated)
    print("-" * 40)

if __name__ == "__main__":
    main()
