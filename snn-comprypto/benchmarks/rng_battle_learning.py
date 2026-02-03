"""
RNG BATTLE ROYALE - LEARNING EDITION
=====================================

Full tournament with ONLINE LEARNING for all predictors!
- SNN: LMS online learning
- DNN: Online SGD (simplified backprop)
- LSTM: Online BPTT (simplified)
- Stat: Adaptive statistics

Parallel processing + More neurons + Long battles!

Author: roll
Date: 2026-01-18
"""

import numpy as np
import time
from collections import deque
import random
import sys
from multiprocessing import Pool, cpu_count
import hashlib

sys.path.insert(0, '.')


# ============================================
# GENERATORS (Same as before, but with more neurons)
# ============================================

class SNNGenerator:
    def __init__(self, seed=42, num_neurons=500):
        np.random.seed(seed)
        self.name = f"SNN-{num_neurons}"
        self.type = "SNN"
        self.num_neurons = num_neurons
        self.dt = 0.5
        self.tau = 20.0
        
        # Reservoir weights (Edge of Chaos)
        self.W_res = np.random.randn(num_neurons, num_neurons)
        rho = max(abs(np.linalg.eigvals(self.W_res)))
        self.W_res *= 1.4 / rho
        mask = np.random.rand(num_neurons, num_neurons) < 0.1
        self.W_res *= mask
        
        self.W_in = np.random.randn(num_neurons) * 40.0
        self.v = np.full(num_neurons, -65.0)
        self.fire_rate = np.zeros(num_neurons)
        self.last_input = 0
        
    def generate(self):
        u = (self.last_input / 127.5) - 1.0
        I_rec = self.W_res @ self.fire_rate
        I_ext = self.W_in * u
        thermal = np.random.normal(0, 0.5, self.num_neurons)
        I_total = I_rec + I_ext + thermal + 25.0
        
        dv = (-(self.v + 65.0) + I_total) / self.tau * self.dt
        self.v += dv
        
        spikes = (self.v >= -50.0).astype(float)
        self.v[spikes > 0] = -70.0
        self.fire_rate = 0.7 * self.fire_rate + 0.3 * spikes
        
        state_bytes = self.v.tobytes()
        h = hashlib.sha256(state_bytes).digest()
        random_byte = h[0]
        
        exposed = self.v[:max(5, self.num_neurons//20)].copy()
        self.last_input = random_byte
        return random_byte, exposed


class DNNGenerator:
    def __init__(self, seed=42, hidden_size=200):
        np.random.seed(seed)
        self.name = f"DNN-{hidden_size}"
        self.type = "DNN"
        self.W1 = np.random.randn(hidden_size, 64) * 0.1
        self.W2 = np.random.randn(hidden_size, hidden_size) * 0.1
        self.W3 = np.random.randn(256, hidden_size) * 0.1
        self.hidden = np.zeros(hidden_size)
        self.counter = 0
        
    def generate(self):
        x = np.sin(np.arange(64) * self.counter * 0.01) + np.random.randn(64) * 0.01
        h1 = np.tanh(np.dot(self.W1, x))
        h2 = np.tanh(np.dot(self.W2, h1))
        output = np.dot(self.W3, h2)
        random_byte = int(np.argmax(output)) % 256
        self.hidden = h2
        self.counter += 1
        exposed = h2[:15].copy()
        return random_byte, exposed


class LSTMGenerator:
    def __init__(self, seed=42, hidden_size=128):
        np.random.seed(seed)
        self.name = f"LSTM-{hidden_size}"
        self.type = "LSTM"
        self.hidden_size = hidden_size
        input_size = 32
        combined = hidden_size + input_size
        
        self.Wf = np.random.randn(hidden_size, combined) * 0.1
        self.Wi = np.random.randn(hidden_size, combined) * 0.1
        self.Wc = np.random.randn(hidden_size, combined) * 0.1
        self.Wo = np.random.randn(hidden_size, combined) * 0.1
        self.Wy = np.random.randn(256, hidden_size) * 0.1
        self.h = np.zeros(hidden_size)
        self.c = np.zeros(hidden_size)
        self.counter = 0
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        
    def generate(self):
        x = np.sin(np.arange(32) * self.counter * 0.01) + np.random.randn(32) * 0.01
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
        exposed = self.h[:15].copy()
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
# PREDICTORS WITH ONLINE LEARNING!
# ============================================

class SNNPredictorLearning:
    """SNN Predictor with LMS Online Learning"""
    def __init__(self, seed=123, num_neurons=100):
        np.random.seed(seed)
        self.name = f"SNN-Learn-{num_neurons}"
        self.type = "SNN"
        self.num_neurons = num_neurons
        
        self.W_res = np.random.randn(num_neurons, num_neurons) * 0.1
        self.W_in = np.random.randn(num_neurons, 30) * 0.1
        self.W_out = np.zeros(num_neurons)  # Learnable!
        
        self.v = np.full(num_neurons, -65.0)
        self.fire_rate = np.zeros(num_neurons)
        self.alpha = 0.01  # Learning rate
        
    def predict(self, exposed, history):
        # Build input
        x = np.zeros(30)
        if len(exposed) > 0:
            x[:min(15, len(exposed))] = exposed[:15] / 100.0
        if len(history) > 0:
            recent = list(history)[-15:]
            x[15:15+len(recent)] = np.array(recent) / 255.0
        
        # SNN dynamics
        I_in = np.dot(self.W_in, x)
        I_rec = np.dot(self.W_res, self.fire_rate)
        I_total = I_in + I_rec + 25.0
        
        dv = (-(self.v + 65.0) + I_total) / 20.0 * 0.5
        self.v += dv
        spikes = (self.v >= -50.0).astype(float)
        self.v[spikes > 0] = -70.0
        self.fire_rate = 0.7 * self.fire_rate + 0.3 * spikes
        
        # Readout
        y = np.dot(self.W_out, self.fire_rate)
        return int(np.clip((y + 1.0) * 127.5, 0, 255))
    
    def update(self, actual, state):
        # LMS Learning!
        target = (actual / 127.5) - 1.0
        y = np.dot(self.W_out, self.fire_rate)
        error = target - y
        self.W_out += self.alpha * error * self.fire_rate


class DNNPredictorLearning:
    """DNN Predictor with Online SGD (Backpropagation)"""
    def __init__(self, seed=456, hidden_size=100):
        np.random.seed(seed)
        self.name = f"DNN-Learn-{hidden_size}"
        self.type = "DNN"
        
        self.W1 = np.random.randn(hidden_size, 40) * 0.1
        self.W2 = np.random.randn(hidden_size, hidden_size) * 0.1
        self.W3 = np.random.randn(256, hidden_size) * 0.1
        
        self.h1 = None
        self.h2 = None
        self.x = None
        self.alpha = 0.001  # Learning rate
        
    def predict(self, exposed, history):
        self.x = np.zeros(40)
        if len(exposed) > 0:
            self.x[:min(15, len(exposed))] = exposed[:15] / 100.0
        if len(history) > 0:
            recent = list(history)[-25:]
            self.x[15:15+len(recent)] = np.array(recent) / 255.0
        
        # Forward pass
        self.h1 = np.tanh(np.dot(self.W1, self.x))
        self.h2 = np.tanh(np.dot(self.W2, self.h1))
        output = np.dot(self.W3, self.h2)
        
        return int(np.argmax(output)) % 256
    
    def update(self, actual, state):
        if self.x is None:
            return
            
        # Simplified backpropagation
        target = np.zeros(256)
        target[actual] = 1.0
        
        output = np.dot(self.W3, self.h2)
        output_softmax = np.exp(output - np.max(output))
        output_softmax /= output_softmax.sum()
        
        # Output layer gradient
        d3 = output_softmax - target
        dW3 = np.outer(d3, self.h2)
        
        # Hidden layer 2 gradient
        d2 = np.dot(self.W3.T, d3) * (1 - self.h2**2)
        dW2 = np.outer(d2, self.h1)
        
        # Hidden layer 1 gradient
        d1 = np.dot(self.W2.T, d2) * (1 - self.h1**2)
        dW1 = np.outer(d1, self.x)
        
        # Update weights
        self.W3 -= self.alpha * dW3
        self.W2 -= self.alpha * dW2
        self.W1 -= self.alpha * dW1


class LSTMPredictorLearning:
    """LSTM Predictor with Simplified Online Learning"""
    def __init__(self, seed=789, hidden_size=64):
        np.random.seed(seed)
        self.name = f"LSTM-Learn-{hidden_size}"
        self.type = "LSTM"
        
        self.hidden_size = hidden_size
        input_size = 30
        combined = hidden_size + input_size
        
        self.Wf = np.random.randn(hidden_size, combined) * 0.1
        self.Wi = np.random.randn(hidden_size, combined) * 0.1
        self.Wc = np.random.randn(hidden_size, combined) * 0.1
        self.Wo = np.random.randn(hidden_size, combined) * 0.1
        self.Wy = np.random.randn(256, hidden_size) * 0.1
        
        self.h = np.zeros(hidden_size)
        self.c = np.zeros(hidden_size)
        self.alpha = 0.001
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def predict(self, exposed, history):
        x = np.zeros(30)
        if len(exposed) > 0:
            x[:min(15, len(exposed))] = exposed[:15] / 100.0
        if len(history) > 0:
            recent = list(history)[-15:]
            x[15:15+len(recent)] = np.array(recent) / 255.0
        
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
        # Simplified output layer learning only
        target = np.zeros(256)
        target[actual] = 1.0
        
        output = np.dot(self.Wy, self.h)
        output_softmax = np.exp(output - np.max(output))
        output_softmax /= output_softmax.sum()
        
        d = output_softmax - target
        dWy = np.outer(d, self.h)
        self.Wy -= self.alpha * dWy


class StatPredictorLearning:
    """Adaptive Statistical Predictor"""
    def __init__(self):
        self.name = "Stat-Learn"
        self.type = "Stat"
        self.byte_counts = np.ones(256)  # Prior
        self.history = deque(maxlen=100)
        self.transition = np.ones((256, 256))  # Markov model
        self.last_byte = 128
        
    def predict(self, exposed, history):
        if len(self.history) < 5:
            return 128
        
        # Use Markov model for prediction
        pred = np.argmax(self.transition[self.last_byte])
        return int(pred)
    
    def update(self, actual, state):
        # Update Markov transition matrix
        self.transition[self.last_byte, actual] += 1.0
        self.last_byte = actual
        self.history.append(actual)


# ============================================
# BATTLE LOGIC
# ============================================

def run_learning_battle(args):
    """Run a single generator vs learning predictor battle"""
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
    
    # Create LEARNING predictor
    if pred_type == "SNN":
        pred = SNNPredictorLearning(seed=seed+100, num_neurons=100)
    elif pred_type == "DNN":
        pred = DNNPredictorLearning(seed=seed+200, hidden_size=100)
    elif pred_type == "LSTM":
        pred = LSTMPredictorLearning(seed=seed+300, hidden_size=64)
    elif pred_type == "Stat":
        pred = StatPredictorLearning()
    
    # Run battle with learning
    correct = 0
    history = deque(maxlen=100)
    
    # Track learning progress
    window_correct = 0
    window_size = 1000
    early_correct = 0
    late_correct = 0
    
    for i in range(num_rounds):
        actual, exposed = gen.generate()
        prediction = pred.predict(exposed, list(history))
        
        if prediction == actual:
            correct += 1
            window_correct += 1
            if i < num_rounds // 2:
                early_correct += 1
            else:
                late_correct += 1
        
        # LEARNING STEP!
        pred.update(actual, exposed)
        history.append(actual)
    
    early_rate = early_correct / (num_rounds // 2) * 100
    late_rate = late_correct / (num_rounds // 2) * 100
    improvement = late_rate - early_rate
    
    return (gen.name, pred.name, correct / num_rounds * 100, early_rate, late_rate, improvement)


def main():
    print("=" * 70)
    print("   RNG BATTLE ROYALE - LEARNING EDITION")
    print("   All Predictors Learn During Battle!")
    print("=" * 70)
    
    NUM_ROUNDS = 100000  # Long battle for learning
    NUM_SEEDS = 2
    
    # Large-scale generators
    generators = [
        ("SNN", 500),    # 500 neurons
        ("SNN", 300),    # 300 neurons
        ("DNN", 200),    # 200 hidden
        ("LSTM", 128),   # 128 hidden
        ("Random", 0),   # Baseline
    ]
    
    predictors = ["SNN", "DNN", "LSTM", "Stat"]
    
    # Build tasks
    tasks = []
    for gen_type, gen_param in generators:
        for pred_type in predictors:
            for seed in range(42, 42 + NUM_SEEDS):
                tasks.append((gen_type, gen_param, pred_type, NUM_ROUNDS, seed))
    
    print(f"\n  Total battles: {len(tasks)}")
    print(f"  Rounds per battle: {NUM_ROUNDS:,}")
    print(f"  Using {cpu_count()} CPU cores")
    print(f"  Predictors LEARN during battle!")
    print()
    
    start = time.time()
    
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(run_learning_battle, tasks)
    
    elapsed = time.time() - start
    
    # Aggregate results
    aggregated = {}
    learning_data = {}
    
    for gen_name, pred_name, rate, early, late, improvement in results:
        key = (gen_name, pred_name)
        if key not in aggregated:
            aggregated[key] = []
            learning_data[key] = []
        aggregated[key].append(rate)
        learning_data[key].append((early, late, improvement))
    
    # Print results
    print("\n" + "=" * 70)
    print("   RESULTS: Final Prediction Rates")
    print("=" * 70)
    
    gen_names = sorted(set(g for g, p in aggregated.keys()))
    pred_names = sorted(set(p for g, p in aggregated.keys()))
    
    print(f"\n{'Generator':<15}", end="")
    for pred in pred_names:
        print(f" {pred:<14}", end="")
    print("  Average")
    print("-" * 80)
    
    generator_scores = {}
    
    for gen in gen_names:
        print(f"{gen:<15}", end="")
        rates = []
        for pred in pred_names:
            rate = np.mean(aggregated.get((gen, pred), [0]))
            rates.append(rate)
            print(f" {rate:>12.3f}%", end="")
        avg = np.mean(rates)
        generator_scores[gen] = avg
        print(f"  {avg:>6.3f}%")
    
    # Learning Analysis
    print("\n" + "=" * 70)
    print("   LEARNING ANALYSIS: Did Predictors Improve?")
    print("=" * 70)
    
    print(f"\n{'Matchup':<30} {'Early':>10} {'Late':>10} {'Change':>12}")
    print("-" * 70)
    
    for key in sorted(learning_data.keys()):
        gen, pred = key
        data = learning_data[key]
        early_avg = np.mean([d[0] for d in data])
        late_avg = np.mean([d[1] for d in data])
        improvement = np.mean([d[2] for d in data])
        
        trend = "improved!" if improvement > 0.1 else "no change" if abs(improvement) < 0.1 else "worse"
        print(f"{gen} vs {pred:<15} {early_avg:>9.3f}% {late_avg:>9.3f}% {improvement:>+10.3f}% ({trend})")
    
    # Rankings
    print("\n" + "=" * 70)
    print("   GENERATOR RANKING (Lower = Harder to Crack)")
    print("=" * 70)
    
    theory = 100 / 256
    ranked = sorted(generator_scores.items(), key=lambda x: x[1])
    
    for rank, (name, score) in enumerate(ranked, 1):
        status = "RANDOM-LIKE" if score < 1.0 else "PREDICTABLE"
        print(f"  #{rank}: {name:<15} {score:.3f}%  {status}")
    
    print(f"\n  Theory (pure random): {theory:.3f}%")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Total predictions: {len(tasks) * NUM_ROUNDS:,}")
    
    # Save results
    with open("results/learning_battle_results.txt", "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("RNG BATTLE ROYALE - LEARNING EDITION RESULTS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("GENERATOR RANKING (Lower = Better)\n")
        f.write("-" * 50 + "\n")
        for rank, (name, score) in enumerate(ranked, 1):
            status = "RANDOM-LIKE" if score < 1.0 else "PREDICTABLE"
            f.write(f"#{rank}: {name:<15} {score:.3f}%  {status}\n")
        
        f.write("\n\nLEARNING ANALYSIS\n")
        f.write("-" * 50 + "\n")
        for key in sorted(learning_data.keys()):
            gen, pred = key
            data = learning_data[key]
            early_avg = np.mean([d[0] for d in data])
            late_avg = np.mean([d[1] for d in data])
            improvement = np.mean([d[2] for d in data])
            f.write(f"{gen} vs {pred}: Early={early_avg:.3f}%, Late={late_avg:.3f}%, Change={improvement:+.3f}%\n")
        
        f.write(f"\nTime: {elapsed:.1f}s\n")
    
    print("\n  Results saved to: results/learning_battle_results.txt")


if __name__ == "__main__":
    main()
