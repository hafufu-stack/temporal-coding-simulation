"""
SNN Comprypto Benchmark Suite
=============================

SNN Comprypto と 既存の標準技術 (GZIP, AES) を比較検証するためのスクリプト。
「圧縮率」「処理速度」「エントロピー」の観点から性能を測定します。

Comparison Targets:
- Compression: zlib (Deflate/GZIP equivalent)
- Encryption: AES-256-CTR (via Cryptography library) OR Simulated AES overhead if lib missing

Author: ろーる
"""

import numpy as np
import time
import zlib
import os
import sys

# SNN System Import
try:
    from comprypto_system import SNNCompryptor
except ImportError:
    print("Error: comprypto_system.py not found.")
    sys.exit(1)

# AES Lib Import (Optional)
try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.backends import default_backend
    HAS_AES = True
except ImportError:
    HAS_AES = False
    print("info: 'cryptography' library not found. AES comparison will be simulated.")

def entropy_score(data_bytes):
    """シャノン・エントロピー (bit/byte)"""
    if len(data_bytes) == 0: return 0
    arr = np.frombuffer(data_bytes, dtype=np.uint8)
    counts = np.bincount(arr, minlength=256)
    probs = counts[counts > 0] / len(arr)
    return -np.sum(probs * np.log2(probs))

def run_aes_benchmark(data):
    """AES-256-CTR Encryption Benchmark"""
    key = os.urandom(32) # 256 bits
    nonce = os.urandom(16)
    
    start_t = time.time()
    if HAS_AES:
        cipher = Cipher(algorithms.AES(key), modes.CTR(nonce), backend=default_backend())
        encryptor = cipher.encryptor()
        encrypted = encryptor.update(data) + encryptor.finalize()
    else:
        # Simulate AES overhead (approx 0.5GB/s on modern CPU without AES-NI optimization logic in Py)
        # Just random XOR for speed reference
        encrypted = bytes([b ^ 0xAA for b in data]) # Dummy
        time.sleep(len(data) / (500 * 1024 * 1024)) # Simulate 500MB/s
        
    return time.time() - start_t, encrypted

def run_suite(name, data):
    print(f"\n--- Benchmark Target: {name} ---")
    data_len = len(data)
    print(f"Original Size: {data_len:,} bytes")
    original_entropy = entropy_score(data)
    print(f"Original Entropy: {original_entropy:.2f} bits/byte")

    # 1. Standard Method: GZIP + AES
    # --------------------------------
    print("\n[Standard: GZIP -> AES]")
    
    # Step A: Compression
    t0 = time.time()
    compressed = zlib.compress(data, level=6) # Default level
    t_gzip = time.time() - t0
    
    # Step B: Encryption
    t0 = time.time()
    t_aes, aes_out = run_aes_benchmark(compressed)
    
    total_std_time = t_gzip + t_aes
    std_size = len(aes_out)
    std_ratio = std_size / data_len * 100
    
    print(f"  Total Time: {total_std_time*1000:.2f} ms")
    print(f"  Final Size: {std_size:,} bytes ({std_ratio:.1f}%)")
    print(f"  Breakdown: GZIP {t_gzip*1000:.1f}ms / AES {t_aes*1000:.1f}ms")

    # 2. SNN Comprypto (Simultaneous)
    # --------------------------------
    print("\n[Proposed: SNN Comprypto]")
    
    t0 = time.time()
    snn = SNNCompryptor(key_seed=123)
    snn_out, _ = snn.compress_encrypt(data) # list of ints or bytes
    snn_out_bytes = bytes(snn_out)
    t_snn = time.time() - t0
    
    snn_size = len(snn_out_bytes)
    snn_ratio = snn_size / data_len * 100
    snn_ent = entropy_score(snn_out_bytes)
    
    print(f"  Total Time: {t_snn*1000:.2f} ms")
    print(f"  Final Size: {snn_size:,} bytes ({snn_ratio:.1f}%)")
    print(f"  Entropy: {snn_ent:.2f} bits/byte (Ideal: 8.00)")

    # 3. SNN Comprypto (Numba Optimized)
    # ----------------------------------
    print("\n[Proposed: SNN Comprypto (Numba H/A)]")
    
    try:
        from comprypto_numba import SNNCompryptorFast
        t0 = time.time()
        snn_fast = SNNCompryptorFast(key_seed=123)
        # First Run (triggers JIT compile) - exclude from timing if we want pure throughput, 
        # but for fairness we include it or do a warmup. Let's do a tiny warmup.
        snn_fast.compress_encrypt(b'\x00'*10)
        
        t0 = time.time()
        snn_out_fast = snn_fast.compress_encrypt(data)
        t_snn_fast = time.time() - t0
        
        print(f"  Total Time: {t_snn_fast*1000:.2f} ms")
        if t_snn > 0:
            print(f"  🚀 Speedup vs Normal: {t_snn/t_snn_fast:.1f}x")
        
    except ImportError as e:
        print(f"  Error: Numba module import failed: {e}")

    # 4. Verdict
    # ----------
    print("\n>> 速報判定 SNN(Numba) vs Standard")
    
    # Speed
    if 't_snn_fast' in locals():
        target_time = t_snn_fast
        name = "SNN(Numba)"
    else:
        target_time = t_snn
        name = "SNN(Normal)"

    if target_time < total_std_time:
        print(f"  ⚡ 速度: {name} Win! ({total_std_time/target_time:.1f}x Faster)")
    else:
        print(f"  🐢 速度: {name} Slow ({target_time/total_std_time:.1f}x Slower)")
        
    # Size
    if snn_size < std_size:
        print(f"  📦 圧縮: {name} Win! ({(std_size-snn_size)/std_size*100:.1f}% Smaller)")
    else:
        print(f"  🎈 圧縮: {name} Loose")
    
    # Security (Entropy)
    if snn_ent > 7.95:
        print(f"  🔒 暗号強度: 合格 (Entropy {snn_ent:.2f} ≒ Random)")
    else:
        print(f"  ⚠️ 暗号強度: 要改善")

def main():
    print("=== SNN Comprypto Performance Benchmark ===")
    print(f"Running on: Python {sys.version.split()[0]}")
    if HAS_AES:
        print("Standard Crypto Lib: ✅ Detected (cryptography)")
    else:
        print("Standard Crypto Lib: ❌ Not Found (Simulating AES speed)")
        
    # Generate Synthetic Data
    # 1. Repetitive (Highly Compressible)
    print("\nGenerating Data 1: Log Data Pattern (Compressible)...")
    base_log = b"ERROR: Connection timeout at 192.168.1.1 [retry 3] "
    data_log = base_log * 200 # ~10KB
    run_suite("Server Log (Text)", data_log)
    
    # 2. Waveform (Sensor Data)
    print("\nGenerating Data 2: Sensor Wave (Continuous)...")
    t = np.linspace(0, 100*np.pi, 5000)
    wave = (np.sin(t) * 100 + 128).astype(np.uint8)
    data_wave = wave.tobytes()
    run_suite("Sensor Signal (Binary)", data_wave)
    
    # 3. Random (Incompressible)
    print("\nGenerating Data 3: Random Noise (Hard Limit)...")
    data_rnd = os.urandom(5000)
    run_suite("Random Noise", data_rnd)

if __name__ == "__main__":
    main()
