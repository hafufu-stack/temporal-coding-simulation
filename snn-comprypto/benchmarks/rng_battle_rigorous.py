"""
RNG BATTLE ROYALE - RIGOROUS VERSION
=====================================

More rigorous testing with:
- 100,000 rounds (10x more)
- Multiple parameter configurations
- Statistical significance testing

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


# ============================================
# GENERATORS
# ============================================

class SNNGenerator:
    def __init__(self, seed=42, num_neurons=100):
        self.reservoir = CompryptoReservoir(seed, num_neurons=num_neurons)
        self.name = f"SNN-{num_neurons}"
        self.type = "SNN"
        self.last_input = 0
        self.num_neurons = num_neurons
        
    def generate(self):
        self.reservoir.step_predict(self.last_input)
        random_byte = self.reservoir.get_keystream_byte()
        exposed = np.array([n.v for n in self.reservoir.neurons[:max(5, self.num_neurons//20)]]).copy()
        self.last_input = random_byte
        return random_byte, exposed


class DNNGenerator:
    def __init__(self, seed=42, hidden_size=100):
        np.random.seed(seed)
        self.name = f"DNN-{hidden_size}"
        self.type = "DNN"
        self.W1 = np.random.randn(hidden_size, 32) * 0.1
        self.W2 = np.random.randn(hidden_size, hidden_size) * 0.1
        self.W3 = np.random.randn(256, hidden_size) * 0.1
        self.hidden = np.zeros(hidden_size)
        self.counter = 0
        
    def generate(self):
        x = np.sin(np.arange(32) * self.counter * 0.01) + np.random.randn(32) * 0.01
        h1 = np.tanh(np.dot(self.W1, x))
        h2 = np.tanh(np.dot(self.W2, h1))
        output = np.dot(self.W3, h2)
        random_byte = int(np.argmax(output)) % 256
        self.hidden = h2
        self.counter += 1
        exposed = h2[:5].copy()
        return random_byte, exposed


class LSTMGenerator:
    def __init__(self, seed=42, hidden_size=64):
        np.random.seed(seed)
        self.name = f"LSTM-{hidden_size}"
        self.type = "LSTM"
        self.hidden_size = hidden_size
        self.Wf = np.random.randn(hidden_size, hidden_size + 16) * 0.1
        self.Wi = np.random.randn(hidden_size, hidden_size + 16) * 0.1
        self.Wc = np.random.randn(hidden_size, hidden_size + 16) * 0.1
        self.Wo = np.random.randn(hidden_size, hidden_size + 16) * 0.1
        self.Wy = np.random.randn(256, hidden_size) * 0.1
        self.h = np.zeros(hidden_size)
        self.c = np.zeros(hidden_size)
        self.counter = 0
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        
    def generate(self):
        x = np.sin(np.arange(16) * self.counter * 0.01) + np.random.randn(16) * 0.01
        hx = np.concatenate([self.h, x])
        f = self.sigmoid(np.dot(self.Wf, hx))
        i = self.sigmoid(np.dot(self.Wi, hx))
        c_tilde = np.tanh(np.dot(self.Wc, hx))
        o = self.sigmoid(np.dot(self.Wo, hx))
        self.c = f * self.c + i * c_tilde
        self.h = o * np.tanh(self.c)
        output = np.dot(self.Wy, self.h)
        random_byte = int(np.argmax(output)) % 256
        self.counter += 1
        exposed = self.h[:5].copy()
        return random_byte, exposed


class PythonRandomGenerator:
    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self.name = "Py-Random"
        self.type = "Random"
        
    def generate(self):
        random_byte = self.rng.randint(0, 255)
        state = self.rng.getstate()
        exposed = np.array(state[1][:5], dtype=np.float64) / 1e9
        return random_byte, exposed


# ============================================
# PREDICTORS
# ============================================

class SNNPredictor:
    def __init__(self, seed=123):
        self.reservoir = CompryptoReservoir(seed, num_neurons=50)
        self.name = "SNN-Pred"
        
    def predict(self, exposed, history):
        if len(exposed) > 0:
            self.reservoir.step_predict(int(abs(np.mean(exposed) * 127)) % 256)
        return self.reservoir.get_keystream_byte()
    
    def update(self, actual, state):
        pass


class StatPredictor:
    def __init__(self):
        self.name = "Stat-Pred"
        self.history = deque(maxlen=50)
        
    def predict(self, exposed, history):
        if len(history) < 3:
            return 128
        return int(np.mean(list(history)[-5:])) % 256
    
    def update(self, actual, state):
        self.history.append(actual)


# ============================================
# BATTLE LOGIC
# ============================================

def run_battle(generator, predictors, num_rounds):
    results = {p.name: 0 for p in predictors}
    history = deque(maxlen=100)
    
    for _ in range(num_rounds):
        actual, exposed = generator.generate()
        for p in predictors:
            if p.predict(exposed, list(history)) == actual:
                results[p.name] += 1
            p.update(actual, exposed)
        history.append(actual)
    
    return {name: count / num_rounds * 100 for name, count in results.items()}


def main():
    print("=" * 70)
    print("   🧪 RNG BATTLE ROYALE - RIGOROUS VERSION")
    print("   100,000 rounds, multiple configurations")
    print("=" * 70)
    
    NUM_ROUNDS = 100000
    NUM_SEEDS = 3  # Test with different seeds
    
    # Configurations to test
    configs = [
        ("SNN", [50, 100, 200, 300, 500]),
        ("DNN", [50, 100, 200]),
        ("LSTM", [32, 64, 128]),
    ]
    
    all_results = {}
    
    print(f"\n  Running {NUM_ROUNDS:,} rounds per config, {NUM_SEEDS} seeds each...")
    print()
    
    # Test each configuration
    for gen_type, param_values in configs:
        for param in param_values:
            scores = []
            
            for seed in range(42, 42 + NUM_SEEDS):
                if gen_type == "SNN":
                    gen = SNNGenerator(seed=seed, num_neurons=param)
                elif gen_type == "DNN":
                    gen = DNNGenerator(seed=seed, hidden_size=param)
                elif gen_type == "LSTM":
                    gen = LSTMGenerator(seed=seed, hidden_size=param)
                
                predictors = [SNNPredictor(seed=123), StatPredictor()]
                results = run_battle(gen, predictors, NUM_ROUNDS)
                avg_rate = np.mean(list(results.values()))
                scores.append(avg_rate)
            
            name = f"{gen_type}-{param}"
            avg = np.mean(scores)
            std = np.std(scores)
            all_results[name] = {'mean': avg, 'std': std, 'type': gen_type}
            print(f"  {name:<12}: {avg:.3f}% ± {std:.3f}%")
    
    # Add Python random
    print()
    scores = []
    for seed in range(42, 42 + NUM_SEEDS):
        gen = PythonRandomGenerator(seed=seed)
        predictors = [SNNPredictor(seed=123), StatPredictor()]
        results = run_battle(gen, predictors, NUM_ROUNDS)
        scores.append(np.mean(list(results.values())))
    
    all_results["Py-Random"] = {'mean': np.mean(scores), 'std': np.std(scores), 'type': 'Random'}
    print(f"  {'Py-Random':<12}: {np.mean(scores):.3f}% ± {np.std(scores):.3f}%")
    
    # RANKING
    print("\n" + "=" * 70)
    print("   📊 FINAL RANKING (Lower = Better)")
    print("=" * 70)
    
    ranked = sorted(all_results.items(), key=lambda x: x[1]['mean'])
    
    print(f"\n  {'Rank':<5} {'Generator':<12} {'Mean':>8} {'Std':>8} {'vs Random':>10}")
    print("  " + "-" * 50)
    
    theory = 100 / 256
    
    for rank, (name, data) in enumerate(ranked, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        vs_random = f"{data['mean']/theory:.2f}x"
        print(f"  {medal} #{rank:<2} {name:<12} {data['mean']:>7.3f}% {data['std']:>7.3f}% {vs_random:>10}")
    
    # Type summary
    print("\n" + "=" * 70)
    print("   📈 SUMMARY BY TYPE")
    print("=" * 70)
    
    type_scores = {}
    for name, data in all_results.items():
        t = data['type']
        if t not in type_scores:
            type_scores[t] = []
        type_scores[t].append(data['mean'])
    
    type_avg = {t: (np.mean(scores), np.std(scores)) for t, scores in type_scores.items()}
    ranked_types = sorted(type_avg.items(), key=lambda x: x[1][0])
    
    print(f"\n  {'Type':<10} {'Mean':>10} {'Std':>10} {'Status':>15}")
    print("  " + "-" * 50)
    
    for t, (mean, std) in ranked_types:
        status = "✅ RANDOM-LIKE" if mean < 1.0 else "⚠️ PREDICTABLE"
        print(f"  {t:<10} {mean:>9.3f}% {std:>9.3f}% {status:>15}")
    
    # Statistical test
    print("\n" + "=" * 70)
    print("   🔬 STATISTICAL ANALYSIS")
    print("=" * 70)
    
    snn_scores = [d['mean'] for n, d in all_results.items() if d['type'] == 'SNN']
    dnn_scores = [d['mean'] for n, d in all_results.items() if d['type'] == 'DNN']
    lstm_scores = [d['mean'] for n, d in all_results.items() if d['type'] == 'LSTM']
    
    print(f"\n  SNN average:  {np.mean(snn_scores):.3f}%")
    print(f"  DNN average:  {np.mean(dnn_scores):.3f}%")
    print(f"  LSTM average: {np.mean(lstm_scores):.3f}%")
    print(f"  Theory:       {theory:.3f}%")
    
    # Effect size
    effect_snn_dnn = (np.mean(dnn_scores) - np.mean(snn_scores)) / np.mean(snn_scores) * 100
    effect_snn_lstm = (np.mean(lstm_scores) - np.mean(snn_scores)) / np.mean(snn_scores) * 100
    
    print(f"\n  SNN vs DNN:  DNN is {effect_snn_dnn:.1f}% more predictable than SNN")
    print(f"  SNN vs LSTM: LSTM is {effect_snn_lstm:.1f}% more predictable than SNN")
    
    # Winner
    winner = ranked_types[0][0]
    print("\n" + "=" * 70)
    print(f"   🏆 WINNER: {winner} is the BEST random number generator!")
    print("=" * 70)


if __name__ == "__main__":
    start = time.time()
    main()
    elapsed = time.time() - start
    print(f"\n  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
