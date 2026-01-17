"""
RNG BATTLE ROYALE - FULL VERSION
=================================

SNN vs DNN vs CNN vs LSTM vs Random

All generators, all predictors, full tournament!

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
# GENERATORS (produce random numbers)
# ============================================

class SNNGenerator:
    """SNN-based RNG - uses chaotic reservoir dynamics"""
    def __init__(self, seed=42, num_neurons=300):
        self.reservoir = CompryptoReservoir(seed, num_neurons=num_neurons)
        self.name = f"SNN-{num_neurons}"
        self.type = "SNN"
        self.last_input = 0
        self.num_neurons = num_neurons
        
    def generate(self):
        self.reservoir.step_predict(self.last_input)
        random_byte = self.reservoir.get_keystream_byte()
        # Expose 10% of membrane potentials
        exposed = np.array([n.v for n in self.reservoir.neurons[:self.num_neurons//10]]).copy()
        self.last_input = random_byte
        return random_byte, exposed


class DNNGenerator:
    """DNN (MLP) based RNG - feedforward network"""
    def __init__(self, seed=42, hidden_size=100):
        np.random.seed(seed)
        self.name = f"DNN-{hidden_size}"
        self.type = "DNN"
        
        # 3-layer MLP
        self.W1 = np.random.randn(hidden_size, 32) * 0.1
        self.W2 = np.random.randn(hidden_size, hidden_size) * 0.1
        self.W3 = np.random.randn(256, hidden_size) * 0.1
        
        self.hidden = np.zeros(hidden_size)
        self.counter = 0
        
    def generate(self):
        # Input: counter + random noise
        x = np.sin(np.arange(32) * self.counter * 0.01)
        
        # Forward pass
        h1 = np.tanh(np.dot(self.W1, x))
        h2 = np.tanh(np.dot(self.W2, h1))
        output = np.dot(self.W3, h2)
        
        random_byte = int(np.argmax(output)) % 256
        
        # Update state
        self.hidden = h2
        self.counter += 1
        
        # Expose hidden layer
        exposed = h2[:10].copy()
        return random_byte, exposed


class CNNGenerator:
    """CNN-based RNG - uses convolutional patterns"""
    def __init__(self, seed=42, channels=16):
        np.random.seed(seed)
        self.name = f"CNN-{channels}"
        self.type = "CNN"
        
        # 1D convolution filters
        self.filters = np.random.randn(channels, 5) * 0.1
        self.fc = np.random.randn(256, channels * 8) * 0.1
        
        self.buffer = np.zeros(32)
        self.counter = 0
        
    def generate(self):
        # Create input signal
        self.buffer = np.roll(self.buffer, -1)
        self.buffer[-1] = np.sin(self.counter * 0.1) + np.random.randn() * 0.1
        
        # Convolution
        conv_out = []
        for f in self.filters:
            conv = np.convolve(self.buffer, f, mode='valid')
            conv_out.append(np.max(conv))  # Max pooling
        
        features = np.array(conv_out)
        
        # Tile to match fc input size
        features_tiled = np.tile(features, 8)[:self.fc.shape[1]]
        
        # FC layer
        output = np.dot(self.fc, features_tiled)
        random_byte = int(np.argmax(output)) % 256
        
        self.counter += 1
        
        # Expose features
        exposed = features[:10].copy()
        return random_byte, exposed


class LSTMGenerator:
    """LSTM-based RNG - uses recurrent memory"""
    def __init__(self, seed=42, hidden_size=64):
        np.random.seed(seed)
        self.name = f"LSTM-{hidden_size}"
        self.type = "LSTM"
        
        self.hidden_size = hidden_size
        
        # LSTM weights
        self.Wf = np.random.randn(hidden_size, hidden_size + 16) * 0.1  # Forget
        self.Wi = np.random.randn(hidden_size, hidden_size + 16) * 0.1  # Input
        self.Wc = np.random.randn(hidden_size, hidden_size + 16) * 0.1  # Cell
        self.Wo = np.random.randn(hidden_size, hidden_size + 16) * 0.1  # Output
        self.Wy = np.random.randn(256, hidden_size) * 0.1  # Final
        
        self.h = np.zeros(hidden_size)
        self.c = np.zeros(hidden_size)
        self.counter = 0
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        
    def generate(self):
        # Input
        x = np.sin(np.arange(16) * self.counter * 0.01)
        
        # Concatenate h and x
        hx = np.concatenate([self.h, x])
        
        # LSTM gates
        f = self.sigmoid(np.dot(self.Wf, hx))
        i = self.sigmoid(np.dot(self.Wi, hx))
        c_tilde = np.tanh(np.dot(self.Wc, hx))
        o = self.sigmoid(np.dot(self.Wo, hx))
        
        # Update cell and hidden
        self.c = f * self.c + i * c_tilde
        self.h = o * np.tanh(self.c)
        
        # Output
        output = np.dot(self.Wy, self.h)
        random_byte = int(np.argmax(output)) % 256
        
        self.counter += 1
        
        # Expose hidden state
        exposed = self.h[:10].copy()
        return random_byte, exposed


class PythonRandomGenerator:
    """Standard Python random"""
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
# PREDICTORS (try to guess next number)
# ============================================

class SNNPredictor:
    """SNN-based predictor"""
    def __init__(self, seed=123):
        self.reservoir = CompryptoReservoir(seed, num_neurons=50)
        self.name = "SNN-Pred"
        self.history = deque(maxlen=20)
        
    def predict(self, exposed_state, history):
        if len(exposed_state) > 0:
            state_mean = np.mean(exposed_state)
            self.reservoir.step_predict(int(abs(state_mean * 127)) % 256)
        return self.reservoir.get_keystream_byte()
    
    def update(self, actual, state):
        self.history.append(actual)
        
    def reset(self):
        self.reservoir = CompryptoReservoir(123, num_neurons=50)
        self.history.clear()


class DNNPredictor:
    """DNN-based predictor"""
    def __init__(self, seed=456):
        np.random.seed(seed)
        self.name = "DNN-Pred"
        self.W1 = np.random.randn(64, 20) * 0.1
        self.W2 = np.random.randn(256, 64) * 0.1
        self.history = deque(maxlen=20)
        
    def predict(self, exposed_state, history):
        # Combine exposed state with history
        x = np.zeros(20)
        if len(exposed_state) > 0:
            x[:min(10, len(exposed_state))] = exposed_state[:10]
        if len(history) > 0:
            recent = list(history)[-10:]
            x[10:10+len(recent)] = np.array(recent) / 255.0
        
        h = np.tanh(np.dot(self.W1, x))
        output = np.dot(self.W2, h)
        return int(np.argmax(output)) % 256
    
    def update(self, actual, state):
        self.history.append(actual)
        
    def reset(self):
        np.random.seed(456)
        self.W1 = np.random.randn(64, 20) * 0.1
        self.W2 = np.random.randn(256, 64) * 0.1
        self.history.clear()


class StatPredictor:
    """Statistical predictor"""
    def __init__(self, seed=789):
        self.name = "Stat-Pred"
        self.history = deque(maxlen=50)
        
    def predict(self, exposed_state, history):
        if len(history) < 3:
            return 128
        return int(np.mean(list(history)[-5:])) % 256
    
    def update(self, actual, state):
        self.history.append(actual)
        
    def reset(self):
        self.history.clear()


# ============================================
# BATTLE LOGIC
# ============================================

def run_battle(generator, predictors, num_rounds=10000):
    """Run battle between generator and predictors"""
    results = {p.name: {'correct': 0, 'total': 0} for p in predictors}
    history = deque(maxlen=100)
    
    for _ in range(num_rounds):
        actual, exposed = generator.generate()
        
        for p in predictors:
            pred = p.predict(exposed, list(history))
            if pred == actual:
                results[p.name]['correct'] += 1
            results[p.name]['total'] += 1
            p.update(actual, exposed)
        
        history.append(actual)
    
    return results


def main():
    print("=" * 70)
    print("   🏆 RNG BATTLE ROYALE - FULL TOURNAMENT 🏆")
    print("   SNN vs DNN vs CNN vs LSTM vs Random")
    print("=" * 70)
    
    NUM_ROUNDS = 10000
    
    # Create all generators
    generators = [
        SNNGenerator(seed=42, num_neurons=100),
        SNNGenerator(seed=42, num_neurons=300),
        SNNGenerator(seed=42, num_neurons=500),
        DNNGenerator(seed=42, hidden_size=100),
        CNNGenerator(seed=42, channels=16),
        LSTMGenerator(seed=42, hidden_size=64),
        PythonRandomGenerator(seed=42),
    ]
    
    all_results = {}
    
    for gen in generators:
        # Fresh predictors for each generator
        predictors = [
            SNNPredictor(seed=123),
            DNNPredictor(seed=456),
            StatPredictor(seed=789),
        ]
        
        print(f"\n  Testing: {gen.name}...", end=" ", flush=True)
        results = run_battle(gen, predictors, NUM_ROUNDS)
        all_results[gen.name] = results
        print("Done!")
    
    # Results
    print("\n" + "=" * 70)
    print("   📊 RESULTS")
    print("=" * 70)
    
    print(f"\n{'Generator':<12} | {'Predictor':<10} | {'Exact %':>8} | {'vs Random':>10}")
    print("-" * 50)
    
    theory = 100 / 256  # 0.391%
    
    generator_avg = {}
    
    for gen_name, results in all_results.items():
        total_rate = 0
        for pred_name, stats in results.items():
            rate = stats['correct'] / stats['total'] * 100
            total_rate += rate
            vs_random = f"{rate/theory:.2f}x"
            print(f"{gen_name:<12} | {pred_name:<10} | {rate:>7.2f}% | {vs_random:>10}")
        
        generator_avg[gen_name] = total_rate / len(results)
    
    # RANKING
    print("\n" + "=" * 70)
    print("   🏅 GENERATOR RANKING (Lower = Harder to Predict = Better)")
    print("=" * 70)
    
    ranked = sorted(generator_avg.items(), key=lambda x: x[1])
    
    for rank, (name, score) in enumerate(ranked, 1):
        if rank == 1:
            medal = "🥇"
        elif rank == 2:
            medal = "🥈"
        elif rank == 3:
            medal = "🥉"
        else:
            medal = "  "
        
        # Get type
        gen_type = name.split("-")[0]
        status = "✅ RANDOM-LIKE" if score < 0.5 else "⚠️ PREDICTABLE"
        
        print(f"  {medal} #{rank}: {name:<12} | {score:.3f}% | {status}")
    
    # GROUP BY TYPE
    print("\n" + "=" * 70)
    print("   📈 ANALYSIS BY TYPE (Average Prediction Rate)")
    print("=" * 70)
    
    type_scores = {}
    for name, score in generator_avg.items():
        gen_type = name.split("-")[0]
        if gen_type not in type_scores:
            type_scores[gen_type] = []
        type_scores[gen_type].append(score)
    
    type_avg = {t: np.mean(scores) for t, scores in type_scores.items()}
    ranked_types = sorted(type_avg.items(), key=lambda x: x[1])
    
    print(f"\n  {'Type':<10} | {'Avg Prediction Rate':>20} | {'Rank':>5}")
    print("  " + "-" * 45)
    
    for rank, (t, score) in enumerate(ranked_types, 1):
        print(f"  {t:<10} | {score:>19.3f}% | #{rank}")
    
    # WINNER
    print("\n" + "=" * 70)
    winner_type = ranked_types[0][0]
    print(f"   🏆 WINNER: {winner_type} is the BEST random number generator!")
    print(f"   (Hardest to predict by all predictors)")
    print("=" * 70)
    
    return all_results, generator_avg


if __name__ == "__main__":
    start = time.time()
    results, scores = main()
    print(f"\n  Total time: {time.time() - start:.1f}s")
