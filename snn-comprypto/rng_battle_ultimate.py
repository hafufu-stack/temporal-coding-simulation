"""
RNG BATTLE ROYALE - ULTIMATE EDITION
=====================================

Full tournament with ALL generators AND ALL predictors!
Now with MULTIPROCESSING for maximum CPU utilization!

Author: roll
Date: 2026-01-18
"""

import numpy as np
import time
from collections import deque
import random
import sys
from multiprocessing import Pool, cpu_count
from functools import partial

sys.path.insert(0, '.')
from comprypto_system import CompryptoReservoir


# ============================================
# GENERATORS
# ============================================

class SNNGenerator:
    def __init__(self, seed=42, num_neurons=200):
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
        exposed = h2[:10].copy()
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
        exposed = self.h[:10].copy()
        return random_byte, exposed


class PythonRandomGenerator:
    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self.name = "Py-Random"
        self.type = "Random"
        
    def generate(self):
        random_byte = self.rng.randint(0, 255)
        state = self.rng.getstate()
        exposed = np.array(state[1][:10], dtype=np.float64) / 1e9
        return random_byte, exposed


# ============================================
# PREDICTORS (Now includes DNN and LSTM!)
# ============================================

class SNNPredictor:
    def __init__(self, seed=123):
        self.reservoir = CompryptoReservoir(seed, num_neurons=50)
        self.name = "SNN-Pred"
        self.type = "SNN"
        
    def predict(self, exposed, history):
        if len(exposed) > 0:
            self.reservoir.step_predict(int(abs(np.mean(exposed) * 127)) % 256)
        return self.reservoir.get_keystream_byte()
    
    def update(self, actual, state):
        pass


class DNNPredictor:
    """DNN-based predictor - uses feedforward network to predict"""
    def __init__(self, seed=456):
        np.random.seed(seed)
        self.name = "DNN-Pred"
        self.type = "DNN"
        self.W1 = np.random.randn(64, 30) * 0.1
        self.W2 = np.random.randn(64, 64) * 0.1
        self.W3 = np.random.randn(256, 64) * 0.1
        self.history = deque(maxlen=20)
        
    def predict(self, exposed, history):
        # Combine exposed state + history into input
        x = np.zeros(30)
        if len(exposed) > 0:
            x[:min(10, len(exposed))] = exposed[:10] / 100.0  # Normalize
        if len(history) > 0:
            recent = list(history)[-20:]
            x[10:10+len(recent)] = np.array(recent) / 255.0
        
        # Forward pass
        h1 = np.tanh(np.dot(self.W1, x))
        h2 = np.tanh(np.dot(self.W2, h1))
        output = np.dot(self.W3, h2)
        return int(np.argmax(output)) % 256
    
    def update(self, actual, state):
        self.history.append(actual)


class LSTMPredictor:
    """LSTM-based predictor - uses memory to predict patterns"""
    def __init__(self, seed=789):
        np.random.seed(seed)
        self.name = "LSTM-Pred"
        self.type = "LSTM"
        
        hidden_size = 32
        self.Wf = np.random.randn(hidden_size, hidden_size + 20) * 0.1
        self.Wi = np.random.randn(hidden_size, hidden_size + 20) * 0.1
        self.Wc = np.random.randn(hidden_size, hidden_size + 20) * 0.1
        self.Wo = np.random.randn(hidden_size, hidden_size + 20) * 0.1
        self.Wy = np.random.randn(256, hidden_size) * 0.1
        
        self.h = np.zeros(hidden_size)
        self.c = np.zeros(hidden_size)
        self.history = deque(maxlen=20)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def predict(self, exposed, history):
        # Build input
        x = np.zeros(20)
        if len(exposed) > 0:
            x[:min(10, len(exposed))] = exposed[:10] / 100.0
        if len(history) > 0:
            recent = list(history)[-10:]
            x[10:10+len(recent)] = np.array(recent) / 255.0
        
        # LSTM forward
        hx = np.concatenate([self.h, x])
        f = self.sigmoid(np.dot(self.Wf, hx))
        i = self.sigmoid(np.dot(self.Wi, hx))
        c_tilde = np.tanh(np.dot(self.Wc, hx))
        o = self.sigmoid(np.dot(self.Wo, hx))
        
        self.c = f * self.c + i * c_tilde
        self.h = o * np.tanh(self.c)
        
        output = np.dot(self.Wy, self.h)
        return int(np.argmax(output)) % 256
    
    def update(self, actual, state):
        self.history.append(actual)


class StatPredictor:
    def __init__(self):
        self.name = "Stat-Pred"
        self.type = "Stat"
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

def run_single_battle(args):
    """Run a single generator vs predictor battle (for parallel execution)"""
    gen_type, gen_param, pred_type, num_rounds, seed = args
    
    # Create generator
    if gen_type == "SNN":
        gen = SNNGenerator(seed=seed, num_neurons=gen_param)
    elif gen_type == "DNN":
        gen = DNNGenerator(seed=seed, hidden_size=gen_param)
    elif gen_type == "LSTM":
        gen = LSTMGenerator(seed=seed, hidden_size=gen_param)
    elif gen_type == "Random":
        gen = PythonRandomGenerator(seed=seed)
    
    # Create predictor
    if pred_type == "SNN":
        pred = SNNPredictor(seed=seed+100)
    elif pred_type == "DNN":
        pred = DNNPredictor(seed=seed+200)
    elif pred_type == "LSTM":
        pred = LSTMPredictor(seed=seed+300)
    elif pred_type == "Stat":
        pred = StatPredictor()
    
    # Run battle
    correct = 0
    history = deque(maxlen=100)
    
    for _ in range(num_rounds):
        actual, exposed = gen.generate()
        prediction = pred.predict(exposed, list(history))
        if prediction == actual:
            correct += 1
        pred.update(actual, exposed)
        history.append(actual)
    
    return (gen.name, pred.name, correct / num_rounds * 100)


def main():
    print("=" * 70)
    print("   🏆 RNG BATTLE ROYALE - ULTIMATE EDITION 🏆")
    print("   ALL Generators × ALL Predictors × Multiprocessing!")
    print("=" * 70)
    
    NUM_ROUNDS = 50000  # Per battle
    NUM_SEEDS = 2       # For averaging
    
    # Define all combinations
    generators = [
        ("SNN", 100),
        ("SNN", 300),
        ("DNN", 100),
        ("LSTM", 64),
        ("Random", 0),
    ]
    
    predictors = ["SNN", "DNN", "LSTM", "Stat"]
    
    # Build task list
    tasks = []
    for gen_type, gen_param in generators:
        for pred_type in predictors:
            for seed in range(42, 42 + NUM_SEEDS):
                tasks.append((gen_type, gen_param, pred_type, NUM_ROUNDS, seed))
    
    print(f"\n  Total battles: {len(tasks)}")
    print(f"  Rounds per battle: {NUM_ROUNDS:,}")
    print(f"  Using {cpu_count()} CPU cores")
    print()
    
    # Run in parallel!
    start = time.time()
    
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(run_single_battle, tasks)
    
    elapsed = time.time() - start
    
    # Aggregate results
    aggregated = {}
    for gen_name, pred_name, rate in results:
        key = (gen_name, pred_name)
        if key not in aggregated:
            aggregated[key] = []
        aggregated[key].append(rate)
    
    avg_results = {k: np.mean(v) for k, v in aggregated.items()}
    
    # Print results
    print("\n" + "=" * 70)
    print("   📊 RESULTS: Generator (rows) vs Predictor (columns)")
    print("=" * 70)
    
    # Header
    gen_names = list(set(g for g, p in avg_results.keys()))
    pred_names = list(set(p for g, p in avg_results.keys()))
    
    print(f"\n{'Generator':<15}", end="")
    for pred in sorted(pred_names):
        print(f" {pred:<12}", end="")
    print("  Average")
    print("-" * 70)
    
    generator_scores = {}
    
    for gen in sorted(gen_names):
        print(f"{gen:<15}", end="")
        rates = []
        for pred in sorted(pred_names):
            rate = avg_results.get((gen, pred), 0)
            rates.append(rate)
            print(f" {rate:>10.2f}%", end="")
        avg = np.mean(rates)
        generator_scores[gen] = avg
        print(f"  {avg:>6.2f}%")
    
    # Predictor averages
    print("-" * 70)
    print(f"{'Pred Avg':<15}", end="")
    for pred in sorted(pred_names):
        pred_rates = [avg_results.get((g, pred), 0) for g in gen_names]
        print(f" {np.mean(pred_rates):>10.2f}%", end="")
    print()
    
    # Rankings
    print("\n" + "=" * 70)
    print("   🏅 GENERATOR RANKING (Lower = Better Randomness)")
    print("=" * 70)
    
    theory = 100 / 256
    ranked = sorted(generator_scores.items(), key=lambda x: x[1])
    
    for rank, (name, score) in enumerate(ranked, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        status = "✅ RANDOM-LIKE" if score < 1.0 else "⚠️ PREDICTABLE"
        print(f"  {medal} #{rank}: {name:<15} {score:.3f}%  {status}")
    
    # Predictor ranking
    print("\n" + "=" * 70)
    print("   🎯 PREDICTOR RANKING (Higher = Better at Cracking)")
    print("=" * 70)
    
    predictor_scores = {}
    for pred in pred_names:
        pred_rates = [avg_results.get((g, pred), 0) for g in gen_names]
        predictor_scores[pred] = np.mean(pred_rates)
    
    ranked_pred = sorted(predictor_scores.items(), key=lambda x: -x[1])  # Higher is better for predictor
    
    for rank, (name, score) in enumerate(ranked_pred, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        print(f"  {medal} #{rank}: {name:<12} avg prediction rate: {score:.3f}%")
    
    # Summary
    print("\n" + "=" * 70)
    print("   📈 KEY INSIGHTS")
    print("=" * 70)
    
    best_gen = ranked[0][0]
    best_pred = ranked_pred[0][0]
    
    print(f"\n  Best Generator (hardest to crack): {best_gen}")
    print(f"  Best Predictor (best at cracking): {best_pred}")
    print(f"  Theory (pure random): {theory:.3f}%")
    
    print(f"\n  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Speed: {len(tasks) * NUM_ROUNDS / elapsed:,.0f} predictions/sec")
    
    # Save results to file
    with open("results/ultimate_results.txt", "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("RNG BATTLE ROYALE - ULTIMATE EDITION RESULTS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("GENERATOR RANKING (Lower = Better Randomness)\n")
        f.write("-" * 50 + "\n")
        for rank, (name, score) in enumerate(ranked, 1):
            status = "RANDOM-LIKE" if score < 1.0 else "PREDICTABLE"
            f.write(f"#{rank}: {name:<15} {score:.3f}%  {status}\n")
        
        f.write("\n\nPREDICTOR RANKING (Higher = Better at Cracking)\n")
        f.write("-" * 50 + "\n")
        for rank, (name, score) in enumerate(ranked_pred, 1):
            f.write(f"#{rank}: {name:<12} avg prediction rate: {score:.3f}%\n")
        
        f.write(f"\n\nBest Generator: {best_gen}\n")
        f.write(f"Best Predictor: {best_pred}\n")
        f.write(f"Theory (pure random): {theory:.3f}%\n")
        f.write(f"Time: {elapsed:.1f}s\n")
    
    print("\n  Results saved to: results/ultimate_results.txt")


if __name__ == "__main__":
    main()
