"""
Thermal Benchmark for SNN Comprypto
====================================

Verify the avalanche effect of the temperature parameter (2nd encryption key).
Demonstrates that even a tiny temperature difference makes decryption impossible.

Author: roll
Date: 2026-01-17
"""

import numpy as np
import matplotlib.pyplot as plt

from comprypto_system import SNNCompryptor


def run_thermal_benchmark():
    """
    Verify the avalanche effect of temperature parameter
    """
    print("=" * 60)
    print("Thermal Parameter Avalanche Effect Verification")
    print("=" * 60)
    
    # Test data (sine wave)
    test_data = bytes([int(127.5 + 127.5 * np.sin(i * 0.1)) for i in range(500)])
    
    # Encryption parameters
    KEY_SEED = 12345
    ENCRYPT_TEMP = 1.0
    
    # Encrypt
    encryptor = SNNCompryptor(key_seed=KEY_SEED, temperature=ENCRYPT_TEMP)
    encrypted, _ = encryptor.compress_encrypt(test_data)
    
    print(f"\nTest data size: {len(test_data)} bytes")
    print(f"Key seed: {KEY_SEED}")
    print(f"Encryption temperature: {ENCRYPT_TEMP}")
    print("-" * 60)
    
    # Try decryption with different temperatures
    test_temperatures = [
        (ENCRYPT_TEMP, "Correct"),
        (ENCRYPT_TEMP + 0.0001, "+0.0001"),
        (ENCRYPT_TEMP + 0.001, "+0.001"),
        (ENCRYPT_TEMP + 0.01, "+0.01"),
        (ENCRYPT_TEMP + 0.1, "+0.1"),
        (ENCRYPT_TEMP + 1.0, "+1.0"),
    ]
    
    results = []
    
    for temp, label in test_temperatures:
        # Create decryptor with different temperature
        decryptor = SNNCompryptor(key_seed=KEY_SEED, temperature=temp)
        restored = decryptor.decrypt_decompress(encrypted)
        
        # Calculate match rate
        matches = sum(1 for a, b in zip(test_data, restored) if a == b)
        match_rate = matches / len(test_data) * 100
        
        results.append((label, temp, match_rate))
        
        status = "PERFECT" if match_rate == 100 else f"Match: {match_rate:.2f}%"
        print(f"  Temp {temp:10.4f} ({label:12s}): {status}")
    
    # Theoretical random value
    print("-" * 60)
    print(f"Random theory: {100/256:.2f}% (approx 0.39%)")
    print("=" * 60)
    
    # Create graph
    create_thermal_graph(results, test_data, encrypted, KEY_SEED)
    
    return results


def create_thermal_graph(results, test_data, encrypted, key_seed):
    """
    Create temperature sensitivity graph
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("SNN Comprypto: Thermal Parameter (2nd Key) Avalanche Effect", fontsize=14)
    
    # (1) Match rate vs temperature difference
    ax1 = axes[0, 0]
    temp_diffs = ['+0', '+0.0001', '+0.001', '+0.01', '+0.1', '+1.0']
    match_rates = [r[2] for r in results]
    colors = ['green' if r > 99 else 'red' for r in match_rates]
    ax1.bar(temp_diffs, match_rates, color=colors, alpha=0.8)
    ax1.set_ylabel("Match Rate (%)")
    ax1.set_xlabel("Temperature Difference")
    ax1.set_title("(1) Temperature Diff vs Decryption Match Rate")
    ax1.axhline(y=100/256, color='blue', linestyle='--', label=f"Random: {100/256:.2f}%")
    ax1.legend()
    ax1.set_ylim(0, 110)
    
    # (2) Correct temperature decryption
    ax2 = axes[0, 1]
    correct_decryptor = SNNCompryptor(key_seed=key_seed, temperature=1.0)
    correct_restored = correct_decryptor.decrypt_decompress(encrypted)
    ax2.plot(list(test_data[:200]), 'b-', alpha=0.7, label='Original')
    ax2.plot(list(correct_restored[:200]), 'g--', alpha=0.7, label='Decrypted (temp=1.0)')
    ax2.set_title("(2) Correct Temperature: Perfect Decryption")
    ax2.set_xlabel("Byte Position")
    ax2.set_ylabel("Value")
    ax2.legend()
    
    # (3) Wrong temperature decryption (chaos)
    ax3 = axes[1, 0]
    wrong_decryptor = SNNCompryptor(key_seed=key_seed, temperature=1.001)
    wrong_restored = wrong_decryptor.decrypt_decompress(encrypted)
    ax3.plot(list(test_data[:200]), 'b-', alpha=0.7, label='Original')
    ax3.plot(list(wrong_restored[:200]), 'r-', alpha=0.7, label='Decrypted (temp=1.001)')
    ax3.set_title("(3) Wrong Temperature: CHAOS! (temp=1.001)")
    ax3.set_xlabel("Byte Position")
    ax3.set_ylabel("Value")
    ax3.legend()
    
    # (4) Difference histogram
    ax4 = axes[1, 1]
    diff = np.abs(np.array(list(test_data[:200])) - np.array(list(wrong_restored[:200])))
    ax4.bar(range(len(diff)), diff, alpha=0.7, color='red')
    ax4.set_title("(4) Residual Error (Chaotic)")
    ax4.set_xlabel("Byte Position")
    ax4.set_ylabel("Difference")
    
    plt.tight_layout()
    plt.savefig("thermal_avalanche_effect.png", dpi=150)
    print("\nGraph saved to 'thermal_avalanche_effect.png'")
    plt.show()


if __name__ == "__main__":
    run_thermal_benchmark()
