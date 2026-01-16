"""
Neuron Count Phase Transition Analysis
======================================

Find the "sweet spot" or phase transition point where:
- Compression efficiency suddenly improves
- Or starts to plateau (diminishing returns)

Author: roll
Date: 2026-01-17
"""

import numpy as np
import matplotlib.pyplot as plt
import time

from comprypto_system import CompryptoReservoir


def run_phase_transition_analysis():
    """
    Fine-grained analysis to find phase transition points
    """
    print("=" * 60)
    print("Phase Transition Analysis (Fine-Grained)")
    print("=" * 60)
    
    # Test data
    np.random.seed(42)
    test_data = bytes([int(127.5 + 127.5 * np.sin(i * 0.1)) for i in range(500)])
    
    # Fine-grained neuron counts (10 to 600, step 20)
    neuron_counts = list(range(20, 601, 20))
    
    results = []
    
    print(f"\nTesting {len(neuron_counts)} different neuron counts...")
    print("This may take a few minutes...\n")
    
    for i, num_neurons in enumerate(neuron_counts):
        mse = benchmark_mse(test_data, num_neurons)
        results.append({'neurons': num_neurons, 'mse': mse})
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(neuron_counts)} ({num_neurons} neurons, MSE={mse:.1f})")
    
    print("\nAnalysis complete!")
    
    # Find phase transition points
    analyze_transitions(results)
    
    # Create detailed graph
    create_phase_graph(results)
    
    return results


def benchmark_mse(test_data, num_neurons, key_seed=12345):
    """
    Quick MSE benchmark for a given neuron count
    """
    brain = CompryptoReservoir(key_seed, num_neurons=num_neurons)
    
    predictions = []
    last_val = 0
    
    for val in test_data:
        pred = brain.step_predict(last_val)
        predictions.append(pred)
        brain.train(val)
        last_val = val
    
    mse = np.mean([(int(a) - int(b))**2 for a, b in zip(test_data, predictions)])
    return mse


def analyze_transitions(results):
    """
    Analyze the data to find transition points
    """
    print("\n" + "=" * 60)
    print("TRANSITION POINT ANALYSIS")
    print("=" * 60)
    
    neurons = np.array([r['neurons'] for r in results])
    mses = np.array([r['mse'] for r in results])
    
    # Calculate rate of change (derivative)
    mse_diff = np.diff(mses)
    mse_diff_rate = mse_diff / np.diff(neurons)  # Change per neuron
    
    # Find biggest improvement (most negative derivative)
    biggest_drop_idx = np.argmin(mse_diff)
    biggest_drop_neurons = neurons[biggest_drop_idx]
    biggest_drop_rate = mse_diff[biggest_drop_idx]
    
    print(f"\n1. Biggest MSE drop at: {biggest_drop_neurons} -> {neurons[biggest_drop_idx+1]} neurons")
    print(f"   MSE decreased by: {abs(biggest_drop_rate):.1f}")
    
    # Find plateau (where improvement slows down significantly)
    # Define plateau as where improvement rate drops below 10% of max improvement
    max_improvement = abs(np.min(mse_diff))
    plateau_threshold = max_improvement * 0.1
    
    plateau_points = np.where(np.abs(mse_diff) < plateau_threshold)[0]
    if len(plateau_points) > 0:
        first_plateau = neurons[plateau_points[0]]
        print(f"\n2. Diminishing returns start at: ~{first_plateau} neurons")
        print(f"   (Improvement drops below 10% of peak)")
    
    # Calculate efficiency (MSE improvement per neuron)
    efficiency = -mse_diff_rate  # Positive = better
    best_efficiency_idx = np.argmax(efficiency)
    best_efficiency_neurons = neurons[best_efficiency_idx]
    
    print(f"\n3. Best efficiency (MSE drop per neuron): {best_efficiency_neurons} -> {neurons[best_efficiency_idx+1]} neurons")
    print(f"   Efficiency: {efficiency[best_efficiency_idx]:.3f} MSE reduction per neuron")
    
    # Recommendation
    print("\n" + "-" * 60)
    print("RECOMMENDATION:")
    
    # Find point with good balance (80% of min MSE)
    min_mse = np.min(mses)
    target_mse = min_mse * 1.2  # 20% above minimum
    good_enough_idx = np.where(mses <= target_mse)[0][0]
    recommended = neurons[good_enough_idx]
    
    print(f"  For 80% of best accuracy at minimum cost: {recommended} neurons")
    print(f"  (MSE: {mses[good_enough_idx]:.1f} vs best: {min_mse:.1f})")


def create_phase_graph(results):
    """
    Create detailed phase transition visualization
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("SNN Comprypto: Neuron Count Phase Transition Analysis", fontsize=14)
    
    neurons = np.array([r['neurons'] for r in results])
    mses = np.array([r['mse'] for r in results])
    
    # (1) MSE vs Neurons (main curve)
    ax1 = axes[0, 0]
    ax1.plot(neurons, mses, 'b-', linewidth=2, label='MSE')
    ax1.fill_between(neurons, mses, alpha=0.2)
    ax1.set_xlabel("Number of Neurons")
    ax1.set_ylabel("MSE (Prediction Error)")
    ax1.set_title("(1) Learning Curve: Neurons vs MSE")
    ax1.grid(True, alpha=0.3)
    
    # Mark key points
    min_idx = np.argmin(mses)
    ax1.scatter([neurons[min_idx]], [mses[min_idx]], color='green', s=100, zorder=5, label='Best')
    ax1.legend()
    
    # (2) Rate of Change (derivative)
    ax2 = axes[0, 1]
    mse_diff = np.diff(mses)
    mid_neurons = (neurons[:-1] + neurons[1:]) / 2
    ax2.bar(mid_neurons, mse_diff, width=15, alpha=0.7, color='purple')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel("Number of Neurons")
    ax2.set_ylabel("MSE Change")
    ax2.set_title("(2) Rate of Change (Derivative)")
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Highlight biggest drop
    biggest_drop_idx = np.argmin(mse_diff)
    ax2.bar(mid_neurons[biggest_drop_idx], mse_diff[biggest_drop_idx], 
            width=15, color='red', label=f'Max drop: {int(mid_neurons[biggest_drop_idx])} neurons')
    ax2.legend()
    
    # (3) Efficiency (improvement per neuron)
    ax3 = axes[1, 0]
    efficiency = -mse_diff / np.diff(neurons)  # Per neuron improvement
    ax3.plot(mid_neurons, efficiency, 'g-', linewidth=2)
    ax3.fill_between(mid_neurons, efficiency, alpha=0.2, color='green')
    ax3.set_xlabel("Number of Neurons")
    ax3.set_ylabel("Efficiency (MSE reduction per neuron)")
    ax3.set_title("(3) Efficiency: Where to Invest Neurons")
    ax3.grid(True, alpha=0.3)
    
    # (4) Cumulative improvement
    ax4 = axes[1, 1]
    initial_mse = mses[0]
    improvement_pct = (initial_mse - mses) / initial_mse * 100
    ax4.plot(neurons, improvement_pct, 'r-', linewidth=2)
    ax4.fill_between(neurons, improvement_pct, alpha=0.2, color='red')
    ax4.set_xlabel("Number of Neurons")
    ax4.set_ylabel("Improvement (%)")
    ax4.set_title("(4) Total Improvement from Baseline (20 neurons)")
    ax4.grid(True, alpha=0.3)
    
    # Mark 80% of max improvement
    max_improvement = improvement_pct[-1]
    threshold_80 = max_improvement * 0.8
    threshold_idx = np.where(improvement_pct >= threshold_80)[0][0]
    ax4.axhline(y=threshold_80, color='blue', linestyle='--', alpha=0.7, label=f'80% of max ({threshold_80:.1f}%)')
    ax4.axvline(x=neurons[threshold_idx], color='blue', linestyle='--', alpha=0.7)
    ax4.scatter([neurons[threshold_idx]], [improvement_pct[threshold_idx]], color='blue', s=100, zorder=5)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig("phase_transition_analysis.png", dpi=150)
    print(f"\nGraph saved to 'phase_transition_analysis.png'")
    plt.show()


if __name__ == "__main__":
    run_phase_transition_analysis()
