"""
RNG Battle Royale v2 - Neuron Count Comparison
==============================================

Test hypothesis: "More neurons = harder to predict"

Author: roll
Date: 2026-01-17
"""

import numpy as np
import time
from collections import deque
import random
import sys
sys.path.insert(0, '.')
from comprypto_system import CompryptoReservoir


class SNNGenerator:
    def __init__(self, seed=42, num_neurons=100):
        self.reservoir = CompryptoReservoir(seed, num_neurons=num_neurons)
        self.name = f"SNN-{num_neurons}"
        self.num_neurons = num_neurons
        self.last_input = 0
        
    def generate(self):
        self.reservoir.step_predict(self.last_input)
        random_byte = self.reservoir.get_keystream_byte()
        # Expose only 10% of neurons (more realistic)
        exposed_state = np.array([n.v for n in self.reservoir.neurons[:self.num_neurons//10]]).copy()
        self.last_input = random_byte
        return random_byte, exposed_state
    
    def get_state_size(self):
        return max(10, self.num_neurons // 10)


class PythonRandomGenerator:
    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self.name = "Python-random"
        
    def generate(self):
        random_byte = self.rng.randint(0, 255)
        state = self.rng.getstate()
        exposed_state = np.array(state[1][:10], dtype=np.float64)
        return random_byte, exposed_state
    
    def get_state_size(self):
        return 10


class SNNPredictor:
    def __init__(self, state_size, seed=123):
        self.reservoir = CompryptoReservoir(seed, num_neurons=50)
        self.name = "SNN-Pred"
        self.history = deque(maxlen=20)
        
    def predict(self, exposed_state, history):
        if len(exposed_state) > 0:
            state_mean = np.mean(exposed_state)
            self.reservoir.step_predict(int(abs(state_mean) * 127) % 256)
        prediction = self.reservoir.get_keystream_byte()
        return prediction
    
    def update(self, actual, state):
        self.history.append(actual)


class StatisticalPredictor:
    def __init__(self, state_size, seed=456):
        self.name = "Stat-Pred"
        self.history = deque(maxlen=50)
        
    def predict(self, exposed_state, history):
        if len(history) < 3:
            return 128
        recent = list(history)[-5:]
        return int(np.mean(recent)) % 256
    
    def update(self, actual, state):
        self.history.append(actual)


def run_battle(generator, predictors, num_rounds=10000):
    """Run battle with more rounds"""
    results = {p.name: {'correct': 0, 'close_5': 0, 'close_10': 0, 'total': 0} for p in predictors}
    history = deque(maxlen=100)
    
    for _ in range(num_rounds):
        actual_value, exposed_state = generator.generate()
        
        for predictor in predictors:
            prediction = predictor.predict(exposed_state, list(history))
            
            if prediction == actual_value:
                results[predictor.name]['correct'] += 1
            if abs(prediction - actual_value) <= 5:
                results[predictor.name]['close_5'] += 1
            if abs(prediction - actual_value) <= 10:
                results[predictor.name]['close_10'] += 1
            results[predictor.name]['total'] += 1
            
            predictor.update(actual_value, exposed_state)
        
        history.append(actual_value)
    
    return results


def main():
    print("=" * 70)
    print("   RNG BATTLE ROYALE v2 - Neuron Count Comparison")
    print("   Hypothesis: More neurons = harder to predict")
    print("=" * 70)
    
    # Test different neuron counts
    neuron_counts = [50, 100, 200, 300, 500]
    num_rounds = 10000
    
    all_results = {}
    
    # Run SNN generators with different neuron counts
    for n in neuron_counts:
        gen = SNNGenerator(seed=42, num_neurons=n)
        predictors = [
            SNNPredictor(gen.get_state_size(), seed=123),
            StatisticalPredictor(gen.get_state_size(), seed=456),
        ]
        results = run_battle(gen, predictors, num_rounds)
        all_results[gen.name] = results
        print(f"  Completed: {gen.name}")
    
    # Add Python random for comparison
    gen = PythonRandomGenerator(seed=42)
    predictors = [
        SNNPredictor(gen.get_state_size(), seed=123),
        StatisticalPredictor(gen.get_state_size(), seed=456),
    ]
    results = run_battle(gen, predictors, num_rounds)
    all_results[gen.name] = results
    print(f"  Completed: {gen.name}")
    
    # Print results
    print("\n" + "=" * 70)
    print("   RESULTS (10,000 rounds each)")
    print("=" * 70)
    
    print(f"\n{'Generator':<15} | {'Predictor':<12} | {'Exact %':>8} | {'±5':>8} | {'±10':>8}")
    print("-" * 70)
    
    generator_scores = {}
    
    for gen_name, results in all_results.items():
        total_correct = 0
        for pred_name, stats in results.items():
            exact = stats['correct'] / stats['total'] * 100
            close5 = stats['close_5'] / stats['total'] * 100
            close10 = stats['close_10'] / stats['total'] * 100
            total_correct += stats['correct']
            print(f"{gen_name:<15} | {pred_name:<12} | {exact:>7.2f}% | {close5:>7.2f}% | {close10:>7.2f}%")
        
        avg_exact = total_correct / (num_rounds * len(results)) * 100
        generator_scores[gen_name] = avg_exact
    
    # Ranking
    print("\n" + "=" * 70)
    print("   RANKING (by average exact prediction rate)")
    print("   Lower = Better (harder to predict)")
    print("=" * 70)
    
    ranked = sorted(generator_scores.items(), key=lambda x: x[1])
    
    for rank, (name, score) in enumerate(ranked, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        print(f"  {medal} #{rank}: {name:<15} - {score:.3f}%")
    
    # Analysis
    print("\n" + "=" * 70)
    print("   ANALYSIS: Does neuron count matter?")
    print("=" * 70)
    
    snn_scores = [(n, generator_scores[f"SNN-{n}"]) for n in neuron_counts]
    
    print("\n  Neuron Count  |  Prediction Rate")
    print("  " + "-" * 35)
    for n, score in snn_scores:
        trend = "↓" if n > 50 and score < snn_scores[0][1] else "↑" if score > snn_scores[0][1] else "="
        print(f"  {n:>10}     |  {score:.3f}% {trend}")
    
    # Check if hypothesis is true
    if snn_scores[-1][1] < snn_scores[0][1]:
        print("\n  ✅ HYPOTHESIS SUPPORTED: More neurons → harder to predict!")
    else:
        print("\n  ❌ HYPOTHESIS NOT CLEARLY SUPPORTED in this test")
    
    theory = 100 / 256
    print(f"\n  Theory (pure random): {theory:.3f}%")


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"\n  Time: {time.time() - start:.1f}s")
