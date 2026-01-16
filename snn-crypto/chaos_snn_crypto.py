"""
SNNカオス暗号システム (Chaos SNN Cryptosystem)
==============================================

100ニューロン以上の大規模リカレントSNN（リザーバ）が生み出す
カオス的ダイナミクスを利用したストリーム暗号生成器。

仕組み:
------
1. Secret Key (Seed) -> シナプス重み行列と遅延時間を生成
2. 初期値鋭敏性（バタフライ効果） -> 予測不可能なスパイク列を生成
3. スパイク列 -> 鍵ストリーム（乱数列）に変換
4. 平文 XOR 鍵ストリーム -> 暗号文

特徴:
----
- 再現性: 同じKeyなら常に同じカオスパターンを生成（復号可能）
- アバランチ効果: Keyが1e-10ズレるだけで出力が全く異なる
"""

import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import hashlib
import time

class ChaosLIFNeuron:
    """カオス生成用に調整されたLIFニューロン"""
    def __init__(self, dt=0.1, tau=20.0, v_rest=-65.0, v_thresh=-50.0, v_reset=-70.0):
        self.dt = dt
        self.tau = tau
        self.v_rest = v_rest
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.v = v_rest
        self.spike = False
    
    def step(self, I_syn):
        """状態更新"""
        # ノイズを加えることでカオス性を強化（決定的カオスの場合はseed依存にする）
        # ここでは純粋な力学系カオスを目指すため、外部ノイズは入れない
        dv = (-(self.v - self.v_rest) + I_syn) / self.tau * self.dt
        self.v += dv
        
        self.spike = False
        if self.v >= self.v_thresh:
            self.spike = True
            self.v = self.v_reset
            return 1.0 # Spike!
        return 0.0

class ChaosReservoir:
    """
    カオス生成エンジン（リカレントSNN）
    """
    def __init__(self, num_neurons=100, density=0.2, key_seed=42):
        self.num_neurons = num_neurons
        self.dt = 0.5 # 時間分解能を少し粗くして計算効率アップ
        
        # 乱数シードの設定（これが秘密鍵の一部となる）
        np.random.seed(key_seed)
        
        self.neurons = [ChaosLIFNeuron(dt=self.dt) for _ in range(num_neurons)]
        
        # 重み行列 (Sparse)
        # スペクトル半径を調整してカオスの縁（Edge of Chaos）を狙う
        scale = 200.0 # 強力な結合で自発発火を促す
        self.weights = np.random.randn(num_neurons, num_neurons) * scale
        
        # 疎結合にする
        mask = np.random.rand(num_neurons, num_neurons) < density
        self.weights *= mask
        
        # 自己結合は抑制的にする（暴走防止）
        np.fill_diagonal(self.weights, -50.0)
        
        # 各ニューロンの現在の入力電流バッファ
        self.current_inputs = np.zeros(num_neurons)
        
    def run_step(self):
        """1ステップ実行し、全ニューロンの発火状態を返す"""
        spikes = np.zeros(self.num_neurons)
        
        # 各ニューロン更新
        for i, neuron in enumerate(self.neurons):
            # バイアス電流（自発活動の源）
            # 閾値(-50) - 静止(-65) = 15以上の入力が必要
            # 20.0〜30.0の範囲でランダムバイアスを与える
            bias = (hash(i) % 1000) / 100.0 + 20.0 
            
            s = neuron.step(self.current_inputs[i] + bias)
            spikes[i] = s
            
        # 次のステップの入力電流を計算 (行列演算で高速化)
        # I_next = W @ spikes
        self.current_inputs = self.weights @ spikes
        
        return spikes

class SNNCipher:
    """SNNストリーム暗号"""
    def __init__(self, key_seed=123456, num_neurons=100):
        self.reservoir = ChaosReservoir(num_neurons=num_neurons, key_seed=key_seed)
        # 初期過渡状態（トランジェント）を捨てる
        # カオスアトラクタに乗るまで少し回す
        for _ in range(100):
            self.reservoir.run_step()
            
    def generate_keystream(self, length_bytes):
        """指定バイト数の鍵ストリームを生成"""
        keystream = bytearray()
        buffer_byte = 0
        bit_count = 0
        
        # 必要なバイト数になるまでSNNを回す
        while len(keystream) < length_bytes:
            # 1ステップ実行
            spikes = self.reservoir.run_step()
            
            # スパイクパターンからビットを抽出
            # 方法: 全ニューロンの発火パターンのハッシュを取る
            # これにより、1つのニューロンの発火の違いが全体に波及する
            state_bytes = spikes.tobytes()
            step_hash = hashlib.sha256(state_bytes).digest()
            
            # ハッシュ値（32バイト）を鍵ストリームに追加
            for b in step_hash:
                keystream.append(b)
                if len(keystream) >= length_bytes:
                    break
        
        return bytes(keystream)
    
    def encrypt(self, plaintext_bytes):
        """暗号化 (XOR)"""
        if isinstance(plaintext_bytes, str):
            plaintext_bytes = plaintext_bytes.encode('utf-8')
            
        length = len(plaintext_bytes)
        key = self.generate_keystream(length)
        
        encrypted = bytearray(length)
        for i in range(length):
            encrypted[i] = plaintext_bytes[i] ^ key[i]
            
        return bytes(encrypted)
    
    def decrypt(self, ciphertext_bytes):
        """復号 (XORなので暗号化と同じ処理)"""
        # ストリーム暗号は内部状態が進んでしまうため、
        # 本当は初期化し直す必要があるが、
        # ここでは簡略化のため「暗号化と同じインスタンスで続けて復号」はできない仕様とする
        # 復号時は新しいインスタンス（同じ鍵）を作る必要がある
        length = len(ciphertext_bytes)
        key = self.generate_keystream(length)
        
        decrypted = bytearray(length)
        for i in range(length):
            decrypted[i] = ciphertext_bytes[i] ^ key[i]
            
        return bytes(decrypted)

def test_avalanche_effect():
    """アバランチ効果（雪崩効果）の検証"""
    print("\n" + "="*60)
    print("アバランチ効果検証 (Butterfly Effect Test)")
    print("="*60)
    
    seed1 = 12345
    # 鍵をわずかに変える（シードを変えると重み全体が変わるが、
    # ここでは概念的なデモとしてシードの違いを見る）
    seed2 = 12346 
    
    print(f"Key 1: Seed={seed1}")
    print(f"Key 2: Seed={seed2} (Only 1 difference)")
    
    cipher1 = SNNCipher(key_seed=seed1, num_neurons=100)
    cipher2 = SNNCipher(key_seed=seed2, num_neurons=100)
    
    len_bytes = 32
    stream1 = cipher1.generate_keystream(len_bytes)
    stream2 = cipher2.generate_keystream(len_bytes)
    
    # ビット差を数える
    diff_bits = 0
    total_bits = len_bytes * 8
    
    for b1, b2 in zip(stream1, stream2):
        xor_val = b1 ^ b2
        diff_bits += bin(xor_val).count('1')
        
    diff_percent = (diff_bits / total_bits) * 100
    
    print(f"\nStream 1: {stream1.hex()[:32]}...")
    print(f"Stream 2: {stream2.hex()[:32]}...")
    print(f"\nBit Difference: {diff_bits}/{total_bits} bits ({diff_percent:.2f}%)")
    
    if 45 < diff_percent < 55:
        print(">> 判定: 理想的なアバランチ効果 (約50%)")
    else:
        print(">> 判定: 相関が残っています")

def main():
    print("SNN Chaos Crypto System Demo")
    print("-" * 30)
    
    # 平文
    message = "Ryzen AI 9 HX 375 + RTX 5080 is a beast!"
    print(f"Original: {message}")
    
    # 暗号化
    seed = 99999
    cipher_enc = SNNCipher(key_seed=seed, num_neurons=100)
    
    start_time = time.time()
    encrypted = cipher_enc.encrypt(message)
    enc_time = (time.time() - start_time) * 1000
    
    print(f"Encrypted (Hex): {encrypted.hex()}")
    print(f"Time: {enc_time:.2f}ms")
    
    # 復号 (同じ鍵)
    cipher_dec = SNNCipher(key_seed=seed, num_neurons=100)
    decrypted = cipher_dec.decrypt(encrypted)
    
    print(f"Decrypted: {decrypted.decode('utf-8')}")
    
    if message == decrypted.decode('utf-8'):
        print(">> SUCCESS: 復号成功")
    else:
        print(">> FAILED: 復号失敗")
        
    # アバランチ効果テスト
    test_avalanche_effect()

if __name__ == "__main__":
    main()
