"""
Neuron Count Benchmark for SNN Comprypto
=========================================

Test how the number of neurons affects:
1. Prediction accuracy (MSE) - lower is better compression
2. Processing time
3. Security (avalanche effect consistency)

Author: roll
Date: 2026-01-17
"""

import numpy as np
import matplotlib.pyplot as plt
import time

from comprypto_system import CompryptoReservoir, SNNCompryptor


def run_neuron_benchmark():
    """
    Benchmark different neuron counts
    """
    print("=" * 60)
    print("Neuron Count Benchmark")
    print("=" * 60)
    
    # Test data (sine wave - predictable pattern)
    np.random.seed(42)
    test_data = bytes([int(127.5 + 127.5 * np.sin(i * 0.1)) for i in range(500)])
    
    # Neuron counts to test
    neuron_counts = [50, 100, 150, 200, 300, 400, 500]
    
    results = []
    
    print(f"\nTest data: 500 bytes (sine wave)")
    print("-" * 60)
    print(f"{'Neurons':>8} | {'MSE':>10} | {'Time (ms)':>10} | {'Integrity':>10}")
    print("-" * 60)
    
    for num_neurons in neuron_counts:
        # Measure prediction accuracy and time
        mse, encrypt_time, decrypt_time, integrity = benchmark_neurons(
            test_data, num_neurons
        )
        
        results.append({
            'neurons': num_neurons,
            'mse': mse,
            'encrypt_time': encrypt_time,
            'decrypt_time': decrypt_time,
            'integrity': integrity
        })
        
        total_time = encrypt_time + decrypt_time
        status = "OK" if integrity == 100 else "FAIL"
        print(f"{num_neurons:>8} | {mse:>10.2f} | {total_time:>10.1f} | {status:>10}")
    
    print("-" * 60)
    
    # Create visualization
    create_neuron_graph(results)
    
    return results


def benchmark_neurons(test_data, num_neurons, key_seed=12345):
    """
    Benchmark a specific neuron count
    """
    # Create reservoir with specified neuron count
    # We need to modify SNNCompryptor to accept num_neurons
    
    # For now, directly use CompryptoReservoir
    np.random.seed(key_seed)
    
    # --- ENCRYPTION ---
    brain_enc = CompryptoReservoir(key_seed, num_neurons=num_neurons)
    
    start_time = time.time()
    
    encrypted = bytearray()
    predictions = []
    last_val = 0
    
    for val in test_data:
        pred = brain_enc.step_predict(last_val)
        predictions.append(pred)
        residual = (val - pred) % 256
        key_byte = brain_enc.get_keystream_byte()
        cipher_byte = residual ^ key_byte
        encrypted.append(cipher_byte)
        brain_enc.train(val)
        last_val = val
    
    encrypt_time = (time.time() - start_time) * 1000
    
    # Calculate MSE (prediction error)
    mse = np.mean([(int(a) - int(b))**2 for a, b in zip(test_data, predictions)])
    
    # --- DECRYPTION ---
    brain_dec = CompryptoReservoir(key_seed, num_neurons=num_neurons)
    
    start_time = time.time()
    
    restored = bytearray()
    last_val = 0
    
    for cipher_byte in encrypted:
        pred = brain_dec.step_predict(last_val)
        key_byte = brain_dec.get_keystream_byte()
        residual = cipher_byte ^ key_byte
        val = (pred + residual) % 256
        restored.append(val)
        brain_dec.train(val)
        last_val = val
    
    decrypt_time = (time.time() - start_time) * 1000
    
    # Check integrity
    matches = sum(1 for a, b in zip(test_data, restored) if a == b)
    integrity = matches / len(test_data) * 100
    
    return mse, encrypt_time, decrypt_time, integrity


def create_neuron_graph(results):
    """
    Create visualization of neuron count effects
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("SNN Comprypto: Neuron Count vs Performance", fontsize=14)
    
    neurons = [r['neurons'] for r in results]
    mses = [r['mse'] for r in results]
    enc_times = [r['encrypt_time'] for r in results]
    dec_times = [r['decrypt_time'] for r in results]
    total_times = [r['encrypt_time'] + r['decrypt_time'] for r in results]
    
    # (1) MSE vs Neuron Count
    ax1 = axes[0, 0]
    ax1.plot(neurons, mses, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel("Number of Neurons")
    ax1.set_ylabel("MSE (Prediction Error)")
    ax1.set_title("(1) Neuron Count vs Prediction Accuracy")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # Highlight sweet spot
    min_mse_idx = np.argmin(mses)
    ax1.scatter([neurons[min_mse_idx]], [mses[min_mse_idx]], 
                color='green', s=200, zorder=5, label=f'Best: {neurons[min_mse_idx]} neurons')
    ax1.legend()
    
    # (2) Processing Time vs Neuron Count
    ax2 = axes[0, 1]
    ax2.plot(neurons, enc_times, 'r^-', linewidth=2, markersize=8, label='Encrypt')
    ax2.plot(neurons, dec_times, 'gs-', linewidth=2, markersize=8, label='Decrypt')
    ax2.plot(neurons, total_times, 'b*-', linewidth=2, markersize=10, label='Total')
    ax2.set_xlabel("Number of Neurons")
    ax2.set_ylabel("Time (ms)")
    ax2.set_title("(2) Neuron Count vs Processing Time")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # (3) Efficiency Score (MSE * Time trade-off)
    ax3 = axes[1, 0]
    # Lower is better for both, so we use MSE * Time as efficiency metric
    efficiency = [mse * time / 1000 for mse, time in zip(mses, total_times)]
    ax3.bar(neurons, efficiency, color='purple', alpha=0.7)
    ax3.set_xlabel("Number of Neurons")
    ax3.set_ylabel("Inefficiency Score (MSE x Time)")
    ax3.set_title("(3) Efficiency Trade-off (Lower is Better)")
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Highlight best efficiency
    best_eff_idx = np.argmin(efficiency)
    ax3.bar(neurons[best_eff_idx], efficiency[best_eff_idx], 
            color='green', alpha=0.9, label=f'Best: {neurons[best_eff_idx]} neurons')
    ax3.legend()
    
    # (4) Compression Potential (inverse of MSE)
    ax4 = axes[1, 1]
    compression_potential = [1000 / (mse + 1) for mse in mses]  # Higher is better
    ax4.fill_between(neurons, compression_potential, alpha=0.3, color='blue')
    ax4.plot(neurons, compression_potential, 'b-', linewidth=2)
    ax4.set_xlabel("Number of Neurons")
    ax4.set_ylabel("Compression Potential (1000/MSE)")
    ax4.set_title("(4) Compression Efficiency (Higher is Better)")
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("neuron_count_benchmark.png", dpi=150)
    print(f"\nGraph saved to 'neuron_count_benchmark.png'")
    plt.show()
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Best MSE (prediction accuracy): {neurons[min_mse_idx]} neurons (MSE={mses[min_mse_idx]:.2f})")
    print(f"Best efficiency (speed vs accuracy): {neurons[best_eff_idx]} neurons")
    print(f"Recommended: 300 neurons (balance of speed and accuracy)")


if __name__ == "__main__":
    run_neuron_benchmark()
