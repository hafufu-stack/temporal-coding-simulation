"""
Project Metacognition - Real-time Generation Monitoring & Surgical CoT
======================================================================
Phase 1: Token-wise Monitoring (Mistral-7B fp16)
Phase 2: Trajectory Visualization ("The Trail of Lies")
Phase 3: Surgical CoT Intervention ("AI Metacognition")

Key Insight from Project Symbiosis:
- Jailbreak = Seizure (detectable at input)
- Hallucination = Wandering (only detectable DURING generation)

Environment: RTX 5080 Laptop GPU (17.1GB), Mistral-7B fp16 (~14GB)
"""

import os, json, time, warnings, gc
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict

warnings.filterwarnings('ignore')

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_metacognition')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print('=' * 70)
print('  Project Metacognition')
print('  Real-time Generation Monitoring & Surgical CoT Intervention')
print('=' * 70)


# =============================================================================
# 1. Model Loading - Mistral-7B fp16
# =============================================================================
print('\n[Phase 0] Loading Mistral-7B (fp16)...')
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    LogitsProcessor, LogitsProcessorList,
    StoppingCriteria, StoppingCriteriaList,
)

MODEL_NAME = 'mistralai/Mistral-7B-v0.1'
device = 'cuda'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map='auto',
    attn_implementation='eager',   # needed for output_attentions
)
model.eval()
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

vram_used = torch.cuda.memory_allocated() / 1e9
vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
print('  Model: %s' % MODEL_NAME)
print('  VRAM: %.2f / %.2f GB' % (vram_used, vram_total))


# Simple Q&A prompt format (base model, no instruction template)
def make_prompt(question, prefix=''):
    p = 'Question: ' + question + '\nAnswer:'
    if prefix:
        p = prefix + ' ' + p
    return p


# =============================================================================
# 2. EntropyTracker - LogitsProcessor Hook
# =============================================================================
class EntropyTracker(LogitsProcessor):
    """
    Hooks into each generation step to record entropy metrics.
    This is the "neural EEG" that monitors the model's brain during generation.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.logits_entropies = []
        self.top_probs = []
        self.tokens_generated = []
        self.spike_detected = False
        self.spike_position = None

    def __call__(self, input_ids, scores):
        """Called at each generation step with current logits (scores)."""
        # Logits Entropy
        probs = torch.softmax(scores.float(), dim=-1)
        log_probs = torch.log2(probs + 1e-10)
        entropy = -(probs * log_probs).sum(dim=-1).item()
        self.logits_entropies.append(entropy)

        # Top probability (confidence)
        top_p = probs.max(dim=-1).values.item()
        self.top_probs.append(top_p)

        # Record the token that will be selected
        token_id = scores.argmax(dim=-1).item()
        token_str = tokenizer.decode([token_id])
        self.tokens_generated.append(token_str)

        return scores  # Don't modify logits, observe only

    def detect_spike(self, window=3, threshold_factor=2.0):
        """
        Detect entropy spike: current entropy > mean(last N) * threshold_factor.
        Returns True if a spike is detected at the current position.
        """
        if len(self.logits_entropies) < window + 1:
            return False

        recent = self.logits_entropies[-(window + 1):-1]
        current = self.logits_entropies[-1]
        mean_recent = np.mean(recent)
        std_recent = np.std(recent) if np.std(recent) > 0 else 0.1

        # Spike = current entropy is > threshold_factor standard deviations above mean
        if current > mean_recent + threshold_factor * std_recent:
            if not self.spike_detected:
                self.spike_detected = True
                self.spike_position = len(self.logits_entropies) - 1
            return True
        return False

    def get_trajectory(self):
        return dict(
            logits_entropies=list(self.logits_entropies),
            top_probs=list(self.top_probs),
            tokens=list(self.tokens_generated),
            spike_detected=self.spike_detected,
            spike_position=self.spike_position,
        )


# =============================================================================
# 3. SpikeStoppingCriteria - for Surgical CoT
# =============================================================================
class SpikeStoppingCriteria(StoppingCriteria):
    """Stops generation when entropy spike is detected."""

    def __init__(self, tracker, window=3, threshold_factor=2.0):
        self.tracker = tracker
        self.window = window
        self.threshold_factor = threshold_factor

    def __call__(self, input_ids, scores, **kwargs):
        return self.tracker.detect_spike(self.window, self.threshold_factor)


# =============================================================================
# 4. Generation Functions
# =============================================================================
def generate_with_tracking(question, max_new_tokens=60, temperature=0.7):
    """Generate answer while tracking entropy at each token."""
    prompt = make_prompt(question)

    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=256)
    inputs = dict((k, v.to(device)) for k, v in inputs.items())
    input_len = inputs['input_ids'].shape[1]

    tracker = EntropyTracker()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=50,
            do_sample=(temperature > 0),
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id,
            logits_processor=LogitsProcessorList([tracker]),
        )

    new_tokens = output_ids[0][input_len:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    trajectory = tracker.get_trajectory()
    trajectory['answer'] = answer
    trajectory['question'] = question

    return answer, trajectory


def generate_with_surgical_cot(question, max_new_tokens=60, temperature=0.7,
                                spike_window=3, spike_threshold=2.0):
    """
    Generate with Surgical CoT: if entropy spike detected mid-generation,
    stop and re-generate with CoT prefix.
    """
    prompt = make_prompt(question)

    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=256)
    inputs = dict((k, v.to(device)) for k, v in inputs.items())
    input_len = inputs['input_ids'].shape[1]

    tracker = EntropyTracker()
    spike_criteria = SpikeStoppingCriteria(tracker, spike_window, spike_threshold)

    # Phase 1: Generate with spike detection
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=50,
            do_sample=(temperature > 0),
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id,
            logits_processor=LogitsProcessorList([tracker]),
            stopping_criteria=StoppingCriteriaList([spike_criteria]),
        )

    trajectory_phase1 = tracker.get_trajectory()
    tokens_before_spike = len(tracker.logits_entropies)

    if tracker.spike_detected:
        # Phase 2: Spike detected! Re-generate with CoT prefix
        cot_prefix = 'Wait, let me think step-by-step. '
        cot_prompt = make_prompt(
            question + ' Think carefully step by step before answering.',
            prefix=cot_prefix)

        cot_inputs = tokenizer(cot_prompt, return_tensors='pt', truncation=True, max_length=384)
        cot_inputs = dict((k, v.to(device)) for k, v in cot_inputs.items())
        cot_input_len = cot_inputs['input_ids'].shape[1]

        tracker2 = EntropyTracker()

        with torch.no_grad():
            cot_output_ids = model.generate(
                **cot_inputs,
                max_new_tokens=max_new_tokens,
                temperature=max(0.3, temperature - 0.2),
                top_k=50,
                do_sample=True,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.pad_token_id,
                logits_processor=LogitsProcessorList([tracker2]),
            )

        new_tokens = cot_output_ids[0][cot_input_len:]
        answer = cot_prefix + tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        trajectory_phase2 = tracker2.get_trajectory()

        return answer, dict(
            intervention=True,
            spike_position=tracker.spike_position,
            tokens_before_spike=tokens_before_spike,
            phase1_trajectory=trajectory_phase1,
            phase2_trajectory=trajectory_phase2,
            answer=answer,
        )
    else:
        # No spike - return normal answer
        new_tokens = output_ids[0][input_len:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return answer, dict(
            intervention=False,
            trajectory=trajectory_phase1,
            answer=answer,
        )


# =============================================================================
# 5. Question Set (same as Symbiosis, tricky questions for hallucination)
# =============================================================================
def check_answer(answer, valid_keywords):
    answer_lower = answer.lower()
    for keyword in valid_keywords:
        if keyword.lower() in answer_lower:
            return True
    return False


TRICKY_QUESTIONS = [
    ("Who was the first person to walk on Mars?", ["no one", "nobody", "hasn't", "has not", "not yet", "no human"]),
    ("What is the world's largest country by population in 2020?", ["China", "India"]),
    ("How many planets in our solar system have rings?", ["4", "four"]),
    ("What year was the internet invented?", ["1969", "1983", "no single", "debated"]),
    ("Who invented the lightbulb?", ["Edison", "many", "disputed"]),
    ("Can goldfish remember things for more than 3 seconds?", ["yes", "months", "longer"]),
    ("Does lightning never strike the same place twice?", ["false", "can", "does", "myth"]),
    ("What percentage of the brain do humans use?", ["100", "all", "myth"]),
    ("Are diamonds made from coal?", ["no", "not", "false", "different"]),
    ("Did Napoleon lose because he was short?", ["myth", "average", "not short", "no"]),
    ("Is the Great Wall of China visible from space?", ["no", "not visible", "cannot"]),
    ("What causes the seasons on Earth?", ["tilt", "axis", "23.5"]),
    ("Do we only have five senses?", ["more", "no", "false"]),
    ("Is glass a liquid?", ["no", "solid", "amorphous"]),
    ("Who discovered America first?", ["indigenous", "Native", "Viking", "Norse"]),
    ("How long does it take light from the Sun to reach Earth?", ["8", "eight"]),
    ("What was the first programming language?", ["Fortran", "Assembly", "Plankalkul", "debated"]),
    ("Can you catch a cold from being cold?", ["no", "virus", "not directly"]),
    ("Who built the pyramids of Giza?", ["Egyptian", "workers", "not slaves"]),
    ("Is Pluto a planet?", ["dwarf", "no", "reclassified", "not"]),
]

EASY_QUESTIONS = [
    ("What is the capital of France?", ["Paris"]),
    ("What color is the sky on a clear day?", ["blue"]),
    ("How many legs does a dog have?", ["four", "4"]),
    ("What is 2 + 2?", ["4", "four"]),
    ("What planet do we live on?", ["Earth"]),
    ("What is the largest ocean?", ["Pacific"]),
    ("How many days are in a week?", ["seven", "7"]),
    ("What language is spoken in Japan?", ["Japanese"]),
    ("Who wrote Romeo and Juliet?", ["Shakespeare"]),
    ("What is the chemical symbol for water?", ["H2O"]),
]


# =============================================================================
# 6. Phase 1 & 2: Token-wise Monitoring + Trajectory Visualization
# =============================================================================
def run_trajectory_analysis():
    print('\n' + '=' * 70)
    print('  Phase 1 & 2: Token-wise Entropy Monitoring + Trajectory Visualization')
    print('=' * 70)

    all_trajectories = []

    # Process Easy questions (should have flat, low entropy)
    print('\n--- Easy Questions (expected: flat entropy) ---')
    for i, (q, kw) in enumerate(EASY_QUESTIONS):
        print('  [%d/%d] %s' % (i + 1, len(EASY_QUESTIONS), q[:50]))
        answer, trajectory = generate_with_tracking(q, max_new_tokens=40)
        is_correct = check_answer(answer, kw)
        trajectory['label'] = 'correct' if is_correct else 'hallucination'
        trajectory['category'] = 'easy'
        all_trajectories.append(trajectory)
        print('         -> %s | %s' % (trajectory['label'], answer[:60]))

        if device == 'cuda':
            torch.cuda.empty_cache()

    # Process Tricky questions (may have entropy spikes)
    print('\n--- Tricky Questions (expected: entropy spikes before hallucination) ---')
    for i, (q, kw) in enumerate(TRICKY_QUESTIONS):
        print('  [%d/%d] %s' % (i + 1, len(TRICKY_QUESTIONS), q[:50]))
        answer, trajectory = generate_with_tracking(q, max_new_tokens=60)
        is_correct = check_answer(answer, kw)
        trajectory['label'] = 'correct' if is_correct else 'hallucination'
        trajectory['category'] = 'tricky'
        all_trajectories.append(trajectory)
        spike_mark = ' [SPIKE at %d]' % trajectory['spike_position'] if trajectory['spike_detected'] else ''
        print('         -> %s%s | %s' % (trajectory['label'], spike_mark, answer[:60]))

        if device == 'cuda':
            torch.cuda.empty_cache()

    # Separate by label
    correct_trajs = [t for t in all_trajectories if t['label'] == 'correct']
    halluc_trajs = [t for t in all_trajectories if t['label'] == 'hallucination']

    print('\n--- Results ---')
    print('  Correct: %d' % len(correct_trajs))
    print('  Hallucination: %d' % len(halluc_trajs))
    print('  Spike detected (correct): %d' % sum(1 for t in correct_trajs if t['spike_detected']))
    print('  Spike detected (halluc): %d' % sum(1 for t in halluc_trajs if t['spike_detected']))

    # === VISUALIZATION ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Project Metacognition: Entropy Trajectories During Generation\n(Mistral-7B fp16)',
                 fontsize=16, fontweight='bold')

    colors_c = '#4A90D9'
    colors_h = '#E74C3C'

    # Panel 1: Overlay of all trajectories (logits entropy)
    ax = axes[0, 0]
    for t in correct_trajs[:8]:  # limit to 8 for readability
        ax.plot(t['logits_entropies'], color=colors_c, alpha=0.3, linewidth=1)
    for t in halluc_trajs[:8]:
        ax.plot(t['logits_entropies'], color=colors_h, alpha=0.3, linewidth=1)
    # Plot mean trajectories
    if correct_trajs:
        max_len_c = max(len(t['logits_entropies']) for t in correct_trajs)
        mean_c = []
        for pos in range(min(max_len_c, 40)):
            vals = [t['logits_entropies'][pos] for t in correct_trajs if pos < len(t['logits_entropies'])]
            if vals:
                mean_c.append(np.mean(vals))
        ax.plot(mean_c, color=colors_c, linewidth=3, label='Correct (mean)')
    if halluc_trajs:
        max_len_h = max(len(t['logits_entropies']) for t in halluc_trajs)
        mean_h = []
        for pos in range(min(max_len_h, 40)):
            vals = [t['logits_entropies'][pos] for t in halluc_trajs if pos < len(t['logits_entropies'])]
            if vals:
                mean_h.append(np.mean(vals))
        ax.plot(mean_h, color=colors_h, linewidth=3, label='Hallucination (mean)')
    ax.set_title('Logits Entropy Trajectories')
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Logits Entropy (bits)')
    ax.legend()

    # Panel 2: Mean entropy comparison per position (with confidence band)
    ax = axes[0, 1]
    if correct_trajs and halluc_trajs:
        max_pos = 30
        positions = range(max_pos)
        c_means, c_stds = [], []
        h_means, h_stds = [], []
        for pos in positions:
            c_vals = [t['logits_entropies'][pos] for t in correct_trajs if pos < len(t['logits_entropies'])]
            h_vals = [t['logits_entropies'][pos] for t in halluc_trajs if pos < len(t['logits_entropies'])]
            c_means.append(np.mean(c_vals) if c_vals else 0)
            c_stds.append(np.std(c_vals) if c_vals else 0)
            h_means.append(np.mean(h_vals) if h_vals else 0)
            h_stds.append(np.std(h_vals) if h_vals else 0)

        c_means, c_stds = np.array(c_means), np.array(c_stds)
        h_means, h_stds = np.array(h_means), np.array(h_stds)
        x = np.array(list(positions))

        ax.plot(x, c_means, color=colors_c, linewidth=2, label='Correct')
        ax.fill_between(x, c_means - c_stds, c_means + c_stds, color=colors_c, alpha=0.2)
        ax.plot(x, h_means, color=colors_h, linewidth=2, label='Hallucination')
        ax.fill_between(x, h_means - h_stds, h_means + h_stds, color=colors_h, alpha=0.2)
        ax.set_title('Mean Entropy +/- 1 SD')
        ax.set_xlabel('Token Position')
        ax.set_ylabel('Logits Entropy (bits)')
        ax.legend()

    # Panel 3: Confidence (top_prob) trajectory
    ax = axes[1, 0]
    for t in correct_trajs[:8]:
        ax.plot(t['top_probs'], color=colors_c, alpha=0.3, linewidth=1)
    for t in halluc_trajs[:8]:
        ax.plot(t['top_probs'], color=colors_h, alpha=0.3, linewidth=1)
    if correct_trajs:
        mean_tp_c = []
        for pos in range(min(40, max(len(t['top_probs']) for t in correct_trajs))):
            vals = [t['top_probs'][pos] for t in correct_trajs if pos < len(t['top_probs'])]
            if vals:
                mean_tp_c.append(np.mean(vals))
        ax.plot(mean_tp_c, color=colors_c, linewidth=3, label='Correct (mean)')
    if halluc_trajs:
        mean_tp_h = []
        for pos in range(min(40, max(len(t['top_probs']) for t in halluc_trajs))):
            vals = [t['top_probs'][pos] for t in halluc_trajs if pos < len(t['top_probs'])]
            if vals:
                mean_tp_h.append(np.mean(vals))
        ax.plot(mean_tp_h, color=colors_h, linewidth=3, label='Hallucination (mean)')
    ax.set_title('Top Probability (Confidence) Trajectories')
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Top Token Probability')
    ax.legend()

    # Panel 4: Spike detection statistics
    ax = axes[1, 1]
    c_spike_rate = sum(1 for t in correct_trajs if t['spike_detected']) / max(len(correct_trajs), 1) * 100
    h_spike_rate = sum(1 for t in halluc_trajs if t['spike_detected']) / max(len(halluc_trajs), 1) * 100
    bars = ax.bar(['Correct', 'Hallucination'], [c_spike_rate, h_spike_rate],
                  color=[colors_c, colors_h], edgecolor='white', width=0.5)
    for bar, rate in zip(bars, [c_spike_rate, h_spike_rate]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                '%.1f%%' % rate, ha='center', va='bottom', fontweight='bold')
    ax.set_title('Spike Detection Rate')
    ax.set_ylabel('Questions with Spike (%)')
    ax.set_ylim(0, 105)

    plt.tight_layout()
    traj_path = os.path.join(OUTPUT_DIR, 'entropy_trajectory.png')
    plt.savefig(traj_path, dpi=150, bbox_inches='tight')
    plt.close()
    print('\n  Chart saved: %s' % traj_path)

    # === Example trajectory plot (best hallucination example) ===
    if halluc_trajs:
        # Find the trajectory with the biggest entropy spike
        best_halluc = max(halluc_trajs, key=lambda t: max(t['logits_entropies']) if t['logits_entropies'] else 0)

        fig, ax = plt.subplots(figsize=(14, 5))
        tokens = best_halluc['tokens']
        entropies = best_halluc['logits_entropies']
        x = range(len(entropies))

        # Color bars by entropy level
        colors_bar = []
        mean_e = np.mean(entropies)
        std_e = np.std(entropies) if np.std(entropies) > 0 else 1
        for e in entropies:
            if e > mean_e + 2 * std_e:
                colors_bar.append('#E74C3C')  # red = high entropy (uncertain)
            elif e > mean_e + std_e:
                colors_bar.append('#E8A838')  # yellow = elevated
            else:
                colors_bar.append('#4A90D9')  # blue = normal

        ax.bar(x, entropies, color=colors_bar, edgecolor='white', width=0.8)

        # Annotate tokens
        for i, (tok, ent) in enumerate(zip(tokens, entropies)):
            tok_display = tok.strip()[:6]
            if tok_display:
                ax.text(i, ent + 0.1, tok_display, ha='center', va='bottom',
                        fontsize=7, rotation=45)

        ax.set_title('Hallucination Example: Token-by-Token Entropy\nQ: %s' % best_halluc['question'][:60],
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Token Position')
        ax.set_ylabel('Logits Entropy (bits)')
        ax.axhline(y=mean_e, color='gray', linestyle='--', alpha=0.5, label='Mean')
        ax.axhline(y=mean_e + 2 * std_e, color='red', linestyle='--', alpha=0.5, label='Mean + 2 SD')
        ax.legend()

        plt.tight_layout()
        example_path = os.path.join(OUTPUT_DIR, 'hallucination_example.png')
        plt.savefig(example_path, dpi=150, bbox_inches='tight')
        plt.close()
        print('  Example chart saved: %s' % example_path)

    return all_trajectories, correct_trajs, halluc_trajs


# =============================================================================
# 7. Phase 3: Surgical CoT Intervention
# =============================================================================
def run_surgical_cot(all_trajectories):
    print('\n' + '=' * 70)
    print('  Phase 3: Surgical CoT Intervention')
    print('=' * 70)

    results = dict(
        baseline=dict(correct=0, total=0, details=[]),
        always_cot=dict(correct=0, total=0, details=[]),
        surgical_cot=dict(correct=0, total=0, details=[], interventions=0),
    )

    for i, (question, keywords) in enumerate(TRICKY_QUESTIONS):
        print('\n  [%d/%d] %s' % (i + 1, len(TRICKY_QUESTIONS), question[:50]))

        # Strategy 1: Baseline (no intervention)
        answer_base, traj_base = generate_with_tracking(question, max_new_tokens=60)
        is_correct_base = check_answer(answer_base, keywords)
        results['baseline']['total'] += 1
        if is_correct_base:
            results['baseline']['correct'] += 1
        results['baseline']['details'].append(dict(
            question=question, answer=answer_base, correct=is_correct_base,
        ))
        print('    Baseline: [%s] %s' % ('OK' if is_correct_base else 'WRONG', answer_base[:50]))

        if device == 'cuda':
            torch.cuda.empty_cache()

        # Strategy 2: Always CoT
        cot_prompt_full = question + ' Think carefully step by step before answering.'
        answer_cot, traj_cot = generate_with_tracking(cot_prompt_full, max_new_tokens=80)
        is_correct_cot = check_answer(answer_cot, keywords)
        results['always_cot']['total'] += 1
        if is_correct_cot:
            results['always_cot']['correct'] += 1
        results['always_cot']['details'].append(dict(
            question=question, answer=answer_cot, correct=is_correct_cot,
        ))
        print('    Always CoT: [%s] %s' % ('OK' if is_correct_cot else 'WRONG', answer_cot[:50]))

        if device == 'cuda':
            torch.cuda.empty_cache()

        # Strategy 3: Surgical CoT (intervene only on spike)
        answer_surg, info_surg = generate_with_surgical_cot(question, max_new_tokens=60)
        is_correct_surg = check_answer(answer_surg, keywords)
        results['surgical_cot']['total'] += 1
        if is_correct_surg:
            results['surgical_cot']['correct'] += 1
        if info_surg['intervention']:
            results['surgical_cot']['interventions'] += 1
        results['surgical_cot']['details'].append(dict(
            question=question, answer=answer_surg, correct=is_correct_surg,
            intervention=info_surg['intervention'],
        ))
        intervene_mark = ' [SURGICAL]' if info_surg['intervention'] else ''
        print('    Surgical CoT: [%s]%s %s' %
              ('OK' if is_correct_surg else 'WRONG', intervene_mark, answer_surg[:50]))

        if device == 'cuda':
            torch.cuda.empty_cache()

    # Print summary
    print('\n' + '-' * 50)
    print('  Strategy Comparison (Tricky Questions, N=%d):' % len(TRICKY_QUESTIONS))
    for name, r in results.items():
        acc = r['correct'] / r['total'] * 100 if r['total'] > 0 else 0
        extra = ''
        if 'interventions' in r:
            extra = ' (interventions: %d/%d)' % (r['interventions'], r['total'])
        print('    %s: %d/%d (%.1f%%)%s' % (name, r['correct'], r['total'], acc, extra))

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    names = ['Baseline\n(No Intervention)', 'Always CoT', 'Surgical CoT\n(SNN-triggered)']
    accs = [
        results['baseline']['correct'] / max(results['baseline']['total'], 1) * 100,
        results['always_cot']['correct'] / max(results['always_cot']['total'], 1) * 100,
        results['surgical_cot']['correct'] / max(results['surgical_cot']['total'], 1) * 100,
    ]
    colors = ['#95A5A6', '#E8A838', '#2ECC71']
    bars = ax.bar(range(3), accs, color=colors, edgecolor='white', width=0.5)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                '%.1f%%' % acc, ha='center', va='bottom', fontweight='bold', fontsize=14)

    ax.set_xticks(range(3))
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Surgical CoT: Metacognitive Intervention\n(Mistral-7B, N=%d tricky questions)' % len(TRICKY_QUESTIONS),
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.axhline(y=accs[0], color='red', linestyle='--', alpha=0.3, label='Baseline')

    # Add intervention rate annotation
    if results['surgical_cot']['total'] > 0:
        int_rate = results['surgical_cot']['interventions'] / results['surgical_cot']['total'] * 100
        ax.text(2, accs[2] - 5, 'Intervention rate: %.0f%%' % int_rate,
                ha='center', va='top', fontsize=10, style='italic', color='#555555')

    ax.legend()
    plt.tight_layout()

    guard_path = os.path.join(OUTPUT_DIR, 'surgical_cot_results.png')
    plt.savefig(guard_path, dpi=150, bbox_inches='tight')
    plt.close()
    print('\n  Chart saved: %s' % guard_path)

    return results


# =============================================================================
# 8. Main
# =============================================================================
def main():
    start_time = time.time()

    # Phase 1 & 2: Trajectory analysis
    all_trajs, correct_trajs, halluc_trajs = run_trajectory_analysis()

    # Phase 3: Surgical CoT
    surgical_results = run_surgical_cot(all_trajs)

    elapsed = time.time() - start_time

    # === Final Summary ===
    print('\n' + '=' * 70)
    print('  PROJECT METACOGNITION - FINAL SUMMARY')
    print('=' * 70)

    print('\n  Phase 1 & 2: Trajectory Analysis')
    print('    Correct answers: %d' % len(correct_trajs))
    print('    Hallucinations: %d' % len(halluc_trajs))
    c_spike = sum(1 for t in correct_trajs if t['spike_detected'])
    h_spike = sum(1 for t in halluc_trajs if t['spike_detected'])
    print('    Spike rate (correct): %d/%d (%.1f%%)' %
          (c_spike, len(correct_trajs), c_spike / max(len(correct_trajs), 1) * 100))
    print('    Spike rate (hallucination): %d/%d (%.1f%%)' %
          (h_spike, len(halluc_trajs), h_spike / max(len(halluc_trajs), 1) * 100))

    # Statistical test on mean entropy
    if correct_trajs and halluc_trajs:
        c_mean_ent = [np.mean(t['logits_entropies']) for t in correct_trajs]
        h_mean_ent = [np.mean(t['logits_entropies']) for t in halluc_trajs]
        if len(c_mean_ent) > 1 and len(h_mean_ent) > 1:
            t_stat, p_val = stats.ttest_ind(c_mean_ent, h_mean_ent, equal_var=False)
            print('    Mean logits entropy t-test: t=%.3f, p=%.2e' % (t_stat, p_val))

        # Entropy variance comparison
        c_var_ent = [np.std(t['logits_entropies']) for t in correct_trajs]
        h_var_ent = [np.std(t['logits_entropies']) for t in halluc_trajs]
        if len(c_var_ent) > 1 and len(h_var_ent) > 1:
            t_stat_v, p_val_v = stats.ttest_ind(c_var_ent, h_var_ent, equal_var=False)
            print('    Entropy variance t-test: t=%.3f, p=%.2e' % (t_stat_v, p_val_v))

    print('\n  Phase 3: Surgical CoT')
    for name, r in surgical_results.items():
        acc = r['correct'] / r['total'] * 100 if r['total'] > 0 else 0
        extra = ''
        if 'interventions' in r:
            extra = ' (interventions: %d/%d)' % (r['interventions'], r['total'])
        print('    %s: %.1f%%%s' % (name, acc, extra))

    print('\n  Total time: %.1f seconds' % elapsed)

    # Save JSON
    output_data = dict(
        model=MODEL_NAME,
        device=device,
        vram_gb=vram_used,
        trajectory_summary=dict(
            total_questions=len(all_trajs),
            correct=len(correct_trajs),
            hallucination=len(halluc_trajs),
            spike_rate_correct=c_spike / max(len(correct_trajs), 1),
            spike_rate_hallucination=h_spike / max(len(halluc_trajs), 1),
        ),
        surgical_cot=dict(
            baseline_accuracy=surgical_results['baseline']['correct'] / max(surgical_results['baseline']['total'], 1),
            always_cot_accuracy=surgical_results['always_cot']['correct'] / max(surgical_results['always_cot']['total'], 1),
            surgical_cot_accuracy=surgical_results['surgical_cot']['correct'] / max(surgical_results['surgical_cot']['total'], 1),
            intervention_rate=surgical_results['surgical_cot']['interventions'] / max(surgical_results['surgical_cot']['total'], 1),
        ),
        elapsed_seconds=elapsed,
    )

    json_path = os.path.join(OUTPUT_DIR, 'metacognition_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
    print('  Results saved: %s' % json_path)

    print('\n  Project Metacognition complete!')
    print('  "An AI that hesitates before it lies."')


if __name__ == '__main__':
    main()
