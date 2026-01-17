"""
RNG Battle Royale - Simplified Version
=======================================

Test the hypothesis: "SNN is the strongest random number generator"

Experiment Design:
- Generator: Produces random numbers (exposes internal state)
- Predictor: Tries to predict the next random number

Generators:
1. SNN (exposes membrane potentials)
2. Python random (exposes internal state via getstate())
3. numpy.random (exposes internal state)

Predictors:
1. SNN-based predictor
2. LSTM-based predictor (simplified)
3. Statistical predictor (autocorrelation)

Evaluation:
- If prediction matches actual: predictor wins
- Lower prediction accuracy = better generator

Author: roll
Date: 2026-01-17
"""

import numpy as np
import time
import hashlib
from collections import deque
import random

# Import our SNN system
import sys
sys.path.insert(0, '.')
from comprypto_system import CompryptoReservoir


class SNNGenerator:
    """SNN-based random number generator (exposes membrane potentials)"""
    
    def __init__(self, seed=42, num_neurons=100):
        self.reservoir = CompryptoReservoir(seed, num_neurons=num_neurons)
        self.name = "SNN"
        self.last_input = 0
        
    def generate(self):
        """Generate a random byte and return (value, exposed_state)"""
        # Step the reservoir with some input
        self.reservoir.step_predict(self.last_input)
        
        # Generate random byte from membrane potentials
        random_byte = self.reservoir.get_keystream_byte()
        
        # Expose the membrane potentials (like attaching a brain scanner!)
        exposed_state = np.array([n.v for n in self.reservoir.neurons]).copy()
        
        self.last_input = random_byte
        return random_byte, exposed_state
    
    def get_state_size(self):
        return len(self.reservoir.neurons)


class PythonRandomGenerator:
    """Python's built-in random (exposes internal state)"""
    
    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self.name = "Python random"
        
    def generate(self):
        """Generate a random byte and return (value, exposed_state)"""
        random_byte = self.rng.randint(0, 255)
        
        # Expose internal state (Mersenne Twister state)
        state = self.rng.getstate()
        # state[1] is a tuple of 625 integers
        exposed_state = np.array(state[1][:100], dtype=np.float64)  # First 100 values
        
        return random_byte, exposed_state
    
    def get_state_size(self):
        return 100


class NumpyRandomGenerator:
    """NumPy's random (exposes internal state)"""
    
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        self.name = "NumPy random"
        
    def generate(self):
        """Generate a random byte and return (value, exposed_state)"""
        random_byte = self.rng.randint(0, 256)
        
        # Expose internal state
        state = self.rng.get_state()
        exposed_state = state[1][:100].astype(np.float64)  # First 100 values
        
        return random_byte, exposed_state
    
    def get_state_size(self):
        return 100


class SNNPredictor:
    """SNN-based predictor that learns from exposed states"""
    
    def __init__(self, state_size, seed=123):
        self.reservoir = CompryptoReservoir(seed, num_neurons=50)
        self.name = "SNN Predictor"
        self.history = deque(maxlen=10)
        self.state_size = state_size
        
        # Simple linear weights for mapping state to prediction
        np.random.seed(seed)
        self.weights = np.random.randn(state_size) * 0.01
        
    def predict(self, exposed_state, history):
        """Predict next random byte based on exposed state"""
        # Use reservoir to process the state
        state_hash = np.mean(exposed_state[:min(len(exposed_state), self.state_size)])
        self.reservoir.step_predict(int(abs(state_hash) * 127) % 256)
        
        # Combine reservoir output with linear prediction
        reservoir_pred = self.reservoir.get_keystream_byte()
        
        # Add some learning from history
        if len(history) > 0:
            # Simple: predict the mode of recent values
            hist_pred = int(np.mean(history)) % 256
            prediction = (reservoir_pred + hist_pred) // 2
        else:
            prediction = reservoir_pred
            
        return prediction % 256
    
    def update(self, actual_value, exposed_state):
        """Update predictor after seeing actual value"""
        self.history.append(actual_value)


class StatisticalPredictor:
    """Statistical predictor using autocorrelation and patterns"""
    
    def __init__(self, state_size, seed=456):
        self.name = "Statistical Predictor"
        self.history = deque(maxlen=100)
        self.state_history = deque(maxlen=10)
        
    def predict(self, exposed_state, history):
        """Predict based on statistical patterns"""
        if len(history) < 3:
            return 128  # Default to middle value
        
        # Try to find patterns
        recent = list(history)[-10:]
        
        # Method 1: Moving average
        ma_pred = int(np.mean(recent))
        
        # Method 2: Linear extrapolation
        if len(recent) >= 2:
            diff = recent[-1] - recent[-2]
            linear_pred = (recent[-1] + diff) % 256
        else:
            linear_pred = ma_pred
        
        # Method 3: Correlation with exposed state
        if len(self.state_history) > 0:
            # Look for correlation between state changes and output
            state_diff = np.mean(exposed_state) - np.mean(self.state_history[-1])
            corr_pred = (recent[-1] + int(state_diff * 10)) % 256
        else:
            corr_pred = ma_pred
        
        self.state_history.append(exposed_state.copy())
        
        # Combine predictions
        prediction = (ma_pred + linear_pred + corr_pred) // 3
        return prediction % 256
    
    def update(self, actual_value, exposed_state):
        """Update predictor after seeing actual value"""
        self.history.append(actual_value)


class LSTMSimplePredictor:
    """Simplified LSTM-like predictor (without deep learning library)"""
    
    def __init__(self, state_size, seed=789):
        self.name = "LSTM-like Predictor"
        np.random.seed(seed)
        
        # Simple recurrent memory
        self.hidden = np.zeros(32)
        self.cell = np.zeros(32)
        
        # Weights
        self.W_input = np.random.randn(32, min(state_size, 32)) * 0.1
        self.W_hidden = np.random.randn(32, 32) * 0.1
        self.W_output = np.random.randn(256, 32) * 0.1
        
        self.history = deque(maxlen=50)
        
    def predict(self, exposed_state, history):
        """Predict using LSTM-like memory"""
        # Prepare input
        x = exposed_state[:min(len(exposed_state), 32)]
        if len(x) < 32:
            x = np.pad(x, (0, 32 - len(x)))
        
        # Simple LSTM-like update
        forget = 1 / (1 + np.exp(-np.dot(self.W_hidden, self.hidden)))
        input_gate = 1 / (1 + np.exp(-np.dot(self.W_input, x[:self.W_input.shape[1]])))
        
        self.cell = forget * self.cell + input_gate * np.tanh(np.dot(self.W_input, x[:self.W_input.shape[1]]))
        self.hidden = np.tanh(self.cell)
        
        # Output
        output = np.dot(self.W_output, self.hidden)
        prediction = int(np.argmax(output))
        
        return prediction % 256
    
    def update(self, actual_value, exposed_state):
        """Update predictor"""
        self.history.append(actual_value)


def run_battle(generator, predictors, num_rounds=1000):
    """Run a battle between one generator and multiple predictors"""
    
    results = {p.name: {'correct': 0, 'close': 0, 'total': 0} for p in predictors}
    history = deque(maxlen=100)
    
    print(f"\n{'='*60}")
    print(f"Generator: {generator.name}")
    print(f"Predictors: {', '.join([p.name for p in predictors])}")
    print(f"Rounds: {num_rounds}")
    print(f"{'='*60}")
    
    for round_num in range(num_rounds):
        # Generator produces a random value
        actual_value, exposed_state = generator.generate()
        
        # Each predictor tries to predict
        for predictor in predictors:
            prediction = predictor.predict(exposed_state, list(history))
            
            # Check accuracy
            if prediction == actual_value:
                results[predictor.name]['correct'] += 1
            if abs(prediction - actual_value) <= 10:  # Within 10 of actual
                results[predictor.name]['close'] += 1
            results[predictor.name]['total'] += 1
            
            # Update predictor with actual value
            predictor.update(actual_value, exposed_state)
        
        history.append(actual_value)
        
        # Progress
        if (round_num + 1) % 200 == 0:
            print(f"  Round {round_num + 1}/{num_rounds}...")
    
    return results


def run_tournament():
    """Run the full tournament"""
    
    print("=" * 70)
    print("   RNG BATTLE ROYALE - Simplified Version")
    print("   Testing: Is SNN the strongest random number generator?")
    print("=" * 70)
    
    # Create generators
    generators = [
        SNNGenerator(seed=42, num_neurons=100),
        PythonRandomGenerator(seed=42),
        NumpyRandomGenerator(seed=42),
    ]
    
    all_results = {}
    
    for gen in generators:
        # Create fresh predictors for each generator
        state_size = gen.get_state_size()
        predictors = [
            SNNPredictor(state_size, seed=123),
            StatisticalPredictor(state_size, seed=456),
            LSTMSimplePredictor(state_size, seed=789),
        ]
        
        results = run_battle(gen, predictors, num_rounds=1000)
        all_results[gen.name] = results
    
    # Print final results
    print("\n")
    print("=" * 70)
    print("   FINAL RESULTS")
    print("=" * 70)
    
    print("\n### Prediction Accuracy (lower = better generator) ###\n")
    
    # Header
    print(f"{'Generator':<20} | {'Predictor':<20} | {'Exact %':>10} | {'Close %':>10}")
    print("-" * 70)
    
    generator_scores = {}
    
    for gen_name, results in all_results.items():
        gen_total_correct = 0
        for pred_name, stats in results.items():
            exact_pct = stats['correct'] / stats['total'] * 100
            close_pct = stats['close'] / stats['total'] * 100
            gen_total_correct += stats['correct']
            print(f"{gen_name:<20} | {pred_name:<20} | {exact_pct:>9.2f}% | {close_pct:>9.2f}%")
        
        # Average prediction rate for this generator
        avg_pred_rate = gen_total_correct / (stats['total'] * len(results)) * 100
        generator_scores[gen_name] = avg_pred_rate
        print()
    
    # Ranking
    print("\n### GENERATOR RANKING (lower prediction rate = better security) ###\n")
    
    ranked = sorted(generator_scores.items(), key=lambda x: x[1])
    
    for rank, (name, score) in enumerate(ranked, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
        random_theory = 100 / 256  # 0.39%
        comparison = "RANDOM-LIKE" if score < 1.0 else f"{score/random_theory:.1f}x above random"
        print(f"  {medal} #{rank}: {name:<20} - Avg prediction rate: {score:.2f}% ({comparison})")
    
    print("\n" + "=" * 70)
    print("   Theory: Perfect random = 0.39% prediction rate")
    print("=" * 70)
    
    return all_results, generator_scores


if __name__ == "__main__":
    start_time = time.time()
    results, scores = run_tournament()
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f} seconds")
