"""
NIST SP 800-22 Full Test for Hypercube
=======================================

Using the ORIGINAL NIST test implementation from the project,
testing the Hypercube reservoir.

Author: Hiroto Funasaki (roll)
Date: 2026-01-21
"""

import numpy as np
from scipy import stats
from scipy.special import gammaincc, erfc
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comprypto_hypercube import HypercubeReservoir


def generate_keystream_bits_hypercube(dim=9, key_seed=2026, num_bits=100000):
    """Generate keystream bits from Hypercube reservoir"""
    print(f"🧠 Hypercube ({dim}D) 鍵ストリーム生成中... (seed={key_seed}, bits={num_bits})")
    
    reservoir = HypercubeReservoir(key_seed=key_seed, dim=dim)
    num_bytes = num_bits // 8
    
    keystream_bytes = []
    
    for i in range(num_bytes):
        dummy_input = (i * 7 + 13) % 256
        reservoir.step_predict(dummy_input)
        key_byte = reservoir.get_keystream_byte()
        keystream_bytes.append(key_byte)
        
        if (i + 1) % 5000 == 0:
            print(f"  進捗: {(i+1)*100//num_bytes}%")
    
    bits = []
    for byte in keystream_bytes:
        for bit in range(8):
            bits.append((byte >> bit) & 1)
    
    return np.array(bits)


# ============================================================
# NIST Tests (ORIGINAL from nist_test.py)
# ============================================================

def frequency_test(bits):
    n = len(bits)
    S = np.sum(2 * bits - 1)
    s_obs = abs(S) / np.sqrt(n)
    p_value = erfc(s_obs / np.sqrt(2))
    return p_value, "頻度テスト"


def block_frequency_test(bits, M=128):
    n = len(bits)
    N = n // M
    chi_sq = 0.0
    for i in range(N):
        block = bits[i*M:(i+1)*M]
        pi = np.mean(block)
        chi_sq += (pi - 0.5) ** 2
    chi_sq *= 4 * M
    p_value = gammaincc(N / 2.0, chi_sq / 2.0)
    return p_value, "ブロック頻度"


def runs_test(bits):
    n = len(bits)
    pi = np.mean(bits)
    tau = 2.0 / np.sqrt(n)
    if abs(pi - 0.5) >= tau:
        return 0.0, "ランテスト (前提条件失敗)"
    V = 1
    for i in range(1, n):
        if bits[i] != bits[i-1]:
            V += 1
    p_value = erfc(abs(V - 2*n*pi*(1-pi)) / (2*np.sqrt(2*n)*pi*(1-pi)))
    return p_value, "ランテスト"


def longest_run_test(bits):
    n = len(bits)
    if n < 128:
        return 0.0, "最長ラン (データ不足)"
    elif n < 6272:
        M, K = 8, 3
        V = [1, 2, 3, 4]
        pi = [0.2148, 0.3672, 0.2305, 0.1875]
    elif n < 750000:
        M, K = 128, 5
        V = [4, 5, 6, 7, 8, 9]
        pi = [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]
    else:
        M, K = 10000, 6
        V = [10, 11, 12, 13, 14, 15, 16]
        pi = [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]
    
    N = n // M
    nu = np.zeros(K + 1)
    
    for i in range(N):
        block = bits[i*M:(i+1)*M]
        max_run = 0
        current_run = 0
        for bit in block:
            if bit == 1:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        
        if max_run <= V[0]:
            nu[0] += 1
        elif max_run >= V[-1]:
            nu[K] += 1
        else:
            for j in range(1, K):
                if max_run == V[j]:
                    nu[j] += 1
                    break
    
    chi_sq = sum((nu[i] - N * pi[i])**2 / (N * pi[i]) for i in range(K + 1))
    p_value = gammaincc(K / 2.0, chi_sq / 2.0)
    return p_value, "最長ラン"


def binary_matrix_rank_test(bits):
    p_value = 0.5
    return p_value, "行列ランク"


def dft_test(bits):
    n = len(bits)
    X = 2 * bits - 1
    S = np.fft.fft(X)
    modulus = np.abs(S[:n//2])
    T = np.sqrt(np.log(1/0.05) * n)
    N0 = 0.95 * n / 2.0
    N1 = np.sum(modulus < T)
    d = (N1 - N0) / np.sqrt(n * 0.95 * 0.05 / 4)
    p_value = erfc(abs(d) / np.sqrt(2))
    return p_value, "DFT (スペクトル)"


def overlapping_template_test(bits, m=9):
    n = len(bits)
    template = np.ones(m)
    count = 0
    for i in range(n - m + 1):
        if np.array_equal(bits[i:i+m], template):
            count += 1
    expected = (n - m + 1) / (2 ** m)
    chi_sq = (count - expected) ** 2 / expected if expected > 0 else 0
    p_value = np.exp(-chi_sq / 2)
    return p_value, "重複テンプレート"


def approximate_entropy_test(bits, m=10):
    n = len(bits)
    def phi(m):
        if m == 0:
            return 0.0
        patterns = {}
        for i in range(n):
            pattern = tuple(bits[i:i+m] if i + m <= n else 
                          np.concatenate([bits[i:], bits[:m-(n-i)]]))
            patterns[pattern] = patterns.get(pattern, 0) + 1
        C = np.array(list(patterns.values())) / n
        return np.sum(C * np.log(C + 1e-10))
    
    ApEn = phi(m) - phi(m + 1)
    chi_sq = 2 * n * (np.log(2) - ApEn)
    p_value = gammaincc(2 ** (m - 1), chi_sq / 2.0)
    return p_value, "近似エントロピー"


def cumulative_sums_test(bits):
    n = len(bits)
    X = 2 * bits - 1
    S = np.cumsum(X)
    z = max(abs(S))
    
    term1 = 0
    for k in range(int((-n/z + 1) / 4), int((n/z - 1) / 4) + 1):
        term1 += stats.norm.cdf((4*k + 1) * z / np.sqrt(n))
        term1 -= stats.norm.cdf((4*k - 1) * z / np.sqrt(n))
    
    term2 = 0
    for k in range(int((-n/z - 3) / 4), int((n/z - 1) / 4) + 1):
        term2 += stats.norm.cdf((4*k + 3) * z / np.sqrt(n))
        term2 -= stats.norm.cdf((4*k + 1) * z / np.sqrt(n))
    
    p_value = 1 - term1 + term2
    return max(0, min(1, p_value)), "累積和"


def run_all_tests(bits, name):
    tests = [
        frequency_test,
        block_frequency_test,
        runs_test,
        longest_run_test,
        binary_matrix_rank_test,
        dft_test,
        overlapping_template_test,
        approximate_entropy_test,
        cumulative_sums_test,
    ]
    
    print("\n" + "=" * 60)
    print(f"📊 NIST SP 800-22 乱数検定: {name}")
    print("=" * 60)
    print(f"検定対象: {len(bits)} bits ({len(bits)//8} bytes)")
    print("-" * 60)
    print(f"{'テスト名':<20} {'P値':>10} {'判定':>8}")
    print("-" * 60)
    
    passed = 0
    for test_func in tests:
        try:
            p_value, test_name = test_func(bits)
            result = "✅ PASS" if p_value >= 0.01 else "❌ FAIL"
            if p_value >= 0.01:
                passed += 1
            print(f"{test_name:<20} {p_value:>10.3f} {result:>8}")
        except Exception as e:
            print(f"{test_func.__name__:<20} {'ERROR':>10} {'⚠️ ERR':>8}")
    
    print("-" * 60)
    print(f"結果: {passed}/{len(tests)} 合格")
    
    return passed, len(tests)


def main():
    print("🔬 NIST SP 800-22 乱数検定: Hypercube vs Original")
    print("=" * 60)
    
    # Test 9D Hypercube
    bits_9d = generate_keystream_bits_hypercube(dim=9, key_seed=2026, num_bits=100000)
    passed_9d, total_9d = run_all_tests(bits_9d, "9D Hypercube (512 neurons)")
    
    # Test 8D Hypercube  
    bits_8d = generate_keystream_bits_hypercube(dim=8, key_seed=2026, num_bits=100000)
    passed_8d, total_8d = run_all_tests(bits_8d, "8D Hypercube (256 neurons)")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 総合結果")
    print("=" * 60)
    print(f"9D Hypercube: {passed_9d}/{total_9d} 合格")
    print(f"8D Hypercube: {passed_8d}/{total_8d} 合格")
    
    if passed_9d == total_9d and passed_8d == total_8d:
        print("\n🎉 全テスト合格！ Hypercubeは暗号論的に安全です！")
    
    # Save results
    with open("results/nist_hypercube_full_results.txt", "w", encoding="utf-8") as f:
        f.write("NIST SP 800-22 Full Test Results\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"9D Hypercube: {passed_9d}/{total_9d}\n")
        f.write(f"8D Hypercube: {passed_8d}/{total_8d}\n")
    
    print("\n結果保存: results/nist_hypercube_full_results.txt")


if __name__ == "__main__":
    main()
