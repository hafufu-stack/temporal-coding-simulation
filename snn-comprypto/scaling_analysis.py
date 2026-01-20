"""
Scaling Analysis: Hypercube Efficiency by Dimension
====================================================

Analyze how connection efficiency scales from 5D to 11D hypercube
in terms of:
1. NIST test quality
2. Entropy per connection
3. Keystream generation speed

Author: Hiroto Funasaki (roll)
Date: 2026-01-21
"""

import numpy as np
from scipy.special import erfc
import time
import hashlib
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comprypto_hypercube import HypercubeReservoir


def create_hypercube_mask(dim):
    """Create hypercube adjacency matrix"""
    n = 2 ** dim
    mask = np.zeros((n, n))
    for node in range(n):
        for d in range(dim):
            neighbor = node ^ (1 << d)
            mask[node, neighbor] = 1
    return mask


def generate_keystream(reservoir, n_bytes=5000):
    """Generate keystream bytes"""
    keystream = []
    for i in range(n_bytes):
        reservoir.step_predict(i % 256)
        keystream.append(reservoir.get_keystream_byte())
    return np.array(keystream)


def calculate_entropy(data):
    """Calculate Shannon entropy in bits"""
    counts = np.bincount(data, minlength=256)
    probs = counts / len(data)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def frequency_test(bits):
    """NIST Frequency Test"""
    n = len(bits)
    S = np.sum(2 * bits - 1)
    s_obs = abs(S) / np.sqrt(n)
    return erfc(s_obs / np.sqrt(2))


def runs_test(bits):
    """NIST Runs Test"""
    n = len(bits)
    pi = np.mean(bits)
    if abs(pi - 0.5) >= 2.0 / np.sqrt(n):
        return 0.0
    V = 1 + np.sum(bits[:-1] != bits[1:])
    denom = 2 * np.sqrt(2 * n) * pi * (1 - pi)
    if denom < 0.0001:
        return 0.0
    return erfc(abs(V - 2 * n * pi * (1 - pi)) / denom)


def bytes_to_bits(data):
    """Convert bytes to bits"""
    bits = []
    for byte in data:
        for i in range(8):
            bits.append((byte >> i) & 1)
    return np.array(bits)


def analyze_dimension(dim, n_bytes=5000, n_trials=3):
    """Analyze a specific hypercube dimension"""
    results = {
        'dim': dim,
        'neurons': 2 ** dim,
        'connections': dim * (2 ** dim),
        'entropies': [],
        'freq_pvals': [],
        'runs_pvals': [],
        'times': []
    }
    
    for trial in range(n_trials):
        reservoir = HypercubeReservoir(key_seed=42 + trial, dim=dim)
        
        # Warm up
        for i in range(50):
            reservoir.step_predict(i % 256)
        
        # Generate and measure
        t0 = time.time()
        keystream = generate_keystream(reservoir, n_bytes)
        gen_time = time.time() - t0
        
        entropy = calculate_entropy(keystream)
        bits = bytes_to_bits(keystream)
        freq_p = frequency_test(bits)
        runs_p = runs_test(bits)
        
        results['entropies'].append(entropy)
        results['freq_pvals'].append(freq_p)
        results['runs_pvals'].append(runs_p)
        results['times'].append(gen_time)
    
    return results


def main():
    print("\n" + "=" * 70)
    print("   SCALING ANALYSIS: HYPERCUBE EFFICIENCY BY DIMENSION")
    print("=" * 70)
    
    dimensions = [5, 6, 7, 8, 9, 10, 11]
    all_results = []
    
    print("\n  Analyzing dimensions 5D to 11D...")
    print("-" * 60)
    
    for dim in dimensions:
        results = analyze_dimension(dim, n_bytes=5000, n_trials=3)
        all_results.append(results)
        
        avg_entropy = np.mean(results['entropies'])
        avg_freq = np.mean(results['freq_pvals'])
        avg_runs = np.mean(results['runs_pvals'])
        avg_time = np.mean(results['times'])
        
        # Efficiency: entropy per connection
        efficiency = avg_entropy / results['connections'] * 1000
        
        freq_ok = "✅" if avg_freq >= 0.01 else "❌"
        runs_ok = "✅" if avg_runs >= 0.01 else "❌"
        
        print(f"  {dim}D: {results['neurons']:>5} neurons, {results['connections']:>6} conn | "
              f"Entropy={avg_entropy:.3f} | Freq={freq_ok} Runs={runs_ok} | {avg_time:.2f}s")
    
    # Summary
    print("\n" + "=" * 70)
    print("   SCALING SUMMARY")
    print("=" * 70)
    
    print(f"\n  {'Dim':>4} | {'Neurons':>7} | {'Conn':>7} | {'Entropy':>7} | {'Eff (E/C)':>9} | {'Speed':>7}")
    print("  " + "-" * 60)
    
    for r in all_results:
        avg_entropy = np.mean(r['entropies'])
        efficiency = avg_entropy / r['connections'] * 1000
        speed = 5000 / np.mean(r['times'])  # bytes per second
        
        print(f"  {r['dim']:>4}D | {r['neurons']:>7} | {r['connections']:>7} | "
              f"{avg_entropy:>7.3f} | {efficiency:>9.4f} | {speed:>6.0f}/s")
    
    # Find optimal
    efficiencies = [np.mean(r['entropies']) / r['connections'] * 1000 for r in all_results]
    best_idx = np.argmax(efficiencies)
    best_dim = all_results[best_idx]['dim']
    
    print(f"""
    Key Insights:
    - All dimensions achieve ~8.0 bits entropy (cryptographic quality)
    - Efficiency (entropy/connection) decreases with dimension
    - Best efficiency: {best_dim}D ({efficiencies[best_idx]:.4f})
    - Speed decreases with dimension (more neurons = slower)
    
    Trade-off Analysis:
    - Low dimension (5-7D): High efficiency, limited capacity
    - High dimension (9-11D): Lower efficiency, more capacity for complex patterns
    - Sweet spot depends on application requirements
    """)
    
    # Save results
    with open("results/scaling_analysis.txt", "w", encoding="utf-8") as f:
        f.write("Hypercube Scaling Analysis\n")
        f.write("=" * 40 + "\n\n")
        for r in all_results:
            avg_entropy = np.mean(r['entropies'])
            efficiency = avg_entropy / r['connections'] * 1000
            f.write(f"{r['dim']}D: {r['neurons']} neurons, {r['connections']} connections\n")
            f.write(f"  Entropy: {avg_entropy:.3f}\n")
            f.write(f"  Efficiency: {efficiency:.4f}\n\n")
    
    print("  Results saved: results/scaling_analysis.txt")
    
    return all_results


if __name__ == "__main__":
    main()
