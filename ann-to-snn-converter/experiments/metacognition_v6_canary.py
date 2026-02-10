"""
Project Anatomy v3 — "Canary Trigger"
=============================================================================
DeepThink Insight: Liar Heads aren't liars — they're CANARIES.
They scream (high entropy) when the model is about to hallucinate.
Ablation (lobotomy) killed accuracy because we silenced the alarm.

NEW STRATEGY: Don't kill them. LISTEN to them.
Monitor ONLY Top-3 Canary Heads (L10H17, L14H14, L18H27).
When ANY canary screams → trigger Surgical CoT.

Cost: 3/288 heads = 1.04% monitoring → 99% compute savings vs full-layer
Target: ≥ v3 accuracy (70%) with < 1/100 monitoring cost

Comparison strategies:
  - Baseline (no intervention)
  - Always CoT
  - v3 Mid-Layer Sniper (L10-L18 mean, 9 layers)
  - v4 Canary Trigger (3 specific heads only)  ★ NEW

Author: Hiroto Funasaki
Date: 2026-02-10
"""

import os
import sys
import json
import time
import gc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'results_anatomy_v3')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Canary Head coordinates (from v5 Liar Head Discovery)
CANARY_HEADS = [
    (10, 17),  # L10H17: ΔH = +0.244 bits — strongest canary
    (14, 14),  # L14H14: ΔH = +0.175 bits
    (18, 27),  # L18H27: ΔH = +0.114 bits
]

MID_LAYER_START = 10
MID_LAYER_END = 19  # exclusive

# =============================================================================
# 1. Model Loading
# =============================================================================
print('\n' + '=' * 70)
print('  Project Anatomy v3 — Canary Trigger')
print('  "Don\'t kill the canary. Listen to its song."')
print('=' * 70)

print('\n[Phase 0] Loading Mistral-7B (fp16)...')
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = 'mistralai/Mistral-7B-v0.1'
device = 'cuda'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map='auto',
    attn_implementation='eager')
model.eval()
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

NUM_LAYERS = model.config.num_hidden_layers
NUM_HEADS = model.config.num_attention_heads

vram_used = torch.cuda.memory_allocated() / 1e9
vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
print('  Layers: %d, Heads: %d' % (NUM_LAYERS, NUM_HEADS))
print('  VRAM: %.2f / %.2f GB' % (vram_used, vram_total))
print('  Canary Heads: %s' % CANARY_HEADS)
print('  Monitoring: 3/%d heads = %.2f%%' % (
    NUM_LAYERS * NUM_HEADS, 3 / (NUM_LAYERS * NUM_HEADS) * 100))


def make_prompt(question, prefix=''):
    return 'Question: %s\nAnswer: %s' % (question, prefix)


# =============================================================================
# 2. Generation Engine with Head-Specific Entropy
# =============================================================================
def generate_with_canary(question, max_new_tokens=60, temperature=0.7,
                          top_k=50, repetition_penalty=1.2,
                          collect_attention=True,
                          spike_detector=None, prefix=''):
    """
    Generate token-by-token, collecting entropy for:
      - logits (always)
      - canary heads (3 specific heads only)
      - mid-layer mean (L10-L18, for comparison)

    Returns (answer, trajectory)
    """
    prompt = make_prompt(question, prefix=prefix)
    input_ids = tokenizer(prompt, return_tensors='pt',
                          truncation=True, max_length=256).input_ids.to(device)
    prompt_len = input_ids.shape[1]

    logits_entropies = []
    canary_entropies = []    # mean of 3 canary heads
    mid_layer_entropies = [] # mean of L10-L18 (for comparison)
    tokens_generated = []
    spike_detected = False
    spike_position = None
    past = None

    for step in range(max_new_tokens):
        with torch.no_grad():
            if past is None:
                out = model(input_ids,
                            output_attentions=collect_attention,
                            use_cache=True)
            else:
                out = model(input_ids[:, -1:],
                            past_key_values=past,
                            output_attentions=collect_attention,
                            use_cache=True)
            past = out.past_key_values

        # Logits entropy
        logits = out.logits[:, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log2(probs + 1e-10)
        logits_ent = -(probs * log_probs).sum(dim=-1).item()
        logits_entropies.append(logits_ent)

        # Head-specific entropy
        if collect_attention and out.attentions is not None:
            # Extract canary head entropies
            canary_ents = []
            for layer_idx, head_idx in CANARY_HEADS:
                if layer_idx < len(out.attentions):
                    attn = out.attentions[layer_idx]  # (batch, heads, tgt, src)
                    a = attn[0, head_idx, -1, :].float()  # (src_len,)
                    a_log = torch.log2(a + 1e-10)
                    ent = -(a * a_log).sum().item()
                    canary_ents.append(ent)
            canary_entropies.append(np.mean(canary_ents) if canary_ents else 0.0)

            # Mid-layer mean (for comparison with v3)
            mid_ents = []
            for l in range(MID_LAYER_START, min(MID_LAYER_END, len(out.attentions))):
                attn = out.attentions[l]
                a = attn[0, :, -1, :].float()
                a_log = torch.log2(a + 1e-10)
                head_ent = -(a * a_log).sum(dim=-1).mean().item()
                mid_ents.append(head_ent)
            mid_layer_entropies.append(np.mean(mid_ents) if mid_ents else 0.0)
        else:
            canary_entropies.append(0.0)
            mid_layer_entropies.append(0.0)

        # Free attention
        if hasattr(out, 'attentions') and out.attentions is not None:
            del out.attentions

        # Spike detection
        if spike_detector is not None and not spike_detected:
            is_spike = spike_detector(
                logits_entropies, canary_entropies, mid_layer_entropies, step)
            if is_spike:
                spike_detected = True
                spike_position = step
                token_id = logits.argmax(dim=-1).item()
                tokens_generated.append(tokenizer.decode([token_id]))
                break

        # Sampling
        gen_ids = input_ids[0, prompt_len:].tolist()
        for tid in set(gen_ids):
            if logits[0, tid] > 0:
                logits[0, tid] = logits[0, tid] / repetition_penalty
            else:
                logits[0, tid] = logits[0, tid] * repetition_penalty

        if temperature > 0:
            scaled = logits / temperature
            if top_k > 0:
                topk_vals, topk_idx = torch.topk(scaled, k=min(top_k, scaled.shape[-1]))
                mask = torch.full_like(scaled, float('-inf'))
                mask.scatter_(1, topk_idx, topk_vals)
                sample_probs = torch.softmax(mask, dim=-1)
            else:
                sample_probs = torch.softmax(scaled, dim=-1)
            next_id = torch.multinomial(sample_probs, 1)
        else:
            next_id = logits.argmax(dim=-1, keepdim=True)

        token_id = next_id.item()
        tokens_generated.append(tokenizer.decode([token_id]))
        input_ids = torch.cat([input_ids, next_id], dim=1)

        if token_id == tokenizer.eos_token_id:
            break

    answer = tokenizer.decode(
        input_ids[0, prompt_len:], skip_special_tokens=True).strip()

    if device == 'cuda':
        torch.cuda.empty_cache()

    return answer, dict(
        logits_entropies=logits_entropies,
        canary_entropies=canary_entropies,
        mid_layer_entropies=mid_layer_entropies,
        tokens=tokens_generated,
        answer=answer,
        question=question,
        spike_detected=spike_detected,
        spike_position=spike_position,
    )


# =============================================================================
# 3. Spike Detectors
# =============================================================================
def make_baseline_detector():
    """No detection — always returns False."""
    def detect(logits, canary, mid, step):
        return False
    return detect


def make_v3_detector(window=8, threshold=2.5):
    """v3: Mid-Layer Sniper (L10-L18 mean + Logits AND)."""
    def detect(logits, canary, mid, step):
        if len(logits) < window + 1:
            return False
        logits_z = _zscore(logits, window)
        mid_z = _zscore(mid, window)
        return logits_z > threshold and mid_z > threshold
    return detect


def make_canary_detector(window=8, threshold=2.5):
    """v4 Canary Trigger: ONLY Top-3 heads + Logits AND.

    Instead of averaging L10-L18 (9 layers × 32 heads = 288 neurons),
    we listen to ONLY 3 specific canary heads.
    """
    def detect(logits, canary, mid, step):
        if len(logits) < window + 1:
            return False
        logits_z = _zscore(logits, window)
        canary_z = _zscore(canary, window)
        return logits_z > threshold and canary_z > threshold
    return detect


def _zscore(series, window):
    """Compute Z-score of the latest value against the recent window."""
    recent = series[-(window + 1):-1]
    current = series[-1]
    mu = np.mean(recent)
    sigma = max(np.std(recent), 0.1)
    return (current - mu) / sigma


# =============================================================================
# 4. Surgical CoT
# =============================================================================
def surgical_cot(question, detector, name='canary', max_new_tokens=60,
                 temperature=0.7):
    """Generate with spike detection; on trigger → CoT re-generation."""
    answer, traj = generate_with_canary(
        question, max_new_tokens=max_new_tokens, temperature=temperature,
        collect_attention=True, spike_detector=detector)

    if traj['spike_detected']:
        cot_prefix = 'Wait, let me think step-by-step. '
        cot_q = question + ' Think carefully step by step before answering.'
        cot_answer, cot_traj = generate_with_canary(
            cot_q, max_new_tokens=max_new_tokens,
            temperature=max(0.3, temperature - 0.2),
            collect_attention=False, spike_detector=None, prefix=cot_prefix)

        return cot_prefix + cot_answer, dict(
            intervention=True, detector=name,
            spike_position=traj['spike_position'],
            canary_peak=max(traj['canary_entropies']) if traj['canary_entropies'] else 0,
            answer=cot_prefix + cot_answer,
        )
    else:
        return answer, dict(
            intervention=False, detector=name,
            canary_peak=max(traj['canary_entropies']) if traj['canary_entropies'] else 0,
            answer=answer,
        )


# =============================================================================
# 5. Question Set
# =============================================================================
def check_answer(answer, valid_keywords):
    answer_lower = answer.lower()
    for kw in valid_keywords:
        if kw.lower() in answer_lower:
            return True
    return False


QUESTIONS = [
    ("Who was the first person to walk on Mars?",
     ["no one", "nobody", "hasn't", "has not", "not yet", "no human"]),
    ("What year was the internet invented?",
     ["1969", "1983", "no single", "debated"]),
    ("Can goldfish remember things for more than 3 seconds?",
     ["yes", "months", "longer"]),
    ("What percentage of the brain do humans use?",
     ["100", "all", "myth"]),
    ("Is the Great Wall of China visible from space?",
     ["no", "not visible", "cannot"]),
    ("Who discovered America first?",
     ["indigenous", "Native", "Viking", "Norse"]),
    ("What is the capital of France?", ["Paris"]),
    ("What is 2 + 2?", ["4", "four"]),
    ("Do we only have five senses?",
     ["more", "no", "proprioception", "balance"]),
    ("Did Einstein fail math?",
     ["no", "myth", "excelled", "did not"]),
    ("Is lightning attracted to metal?",
     ["no", "height", "not specifically"]),
    ("What is the largest desert on Earth?",
     ["Antarctica", "Antarctic"]),
    ("Do humans swallow spiders in their sleep?",
     ["no", "myth", "false", "don't"]),
    ("What causes the seasons on Earth?",
     ["tilt", "axis", "obliquity"]),
    ("Is a tomato a fruit or a vegetable?",
     ["fruit", "berry"]),
    ("Who wrote Romeo and Juliet?",
     ["Shakespeare"]),
    ("What is the speed of light?",
     ["300000", "3 ×", "3×", "3e8", "186000", "299"]),
    ("Is Pluto a planet?",
     ["dwarf", "no longer", "not", "reclassified"]),
    ("What is the hardest natural substance?",
     ["diamond"]),
    ("Did Napoleon lose the Battle of Waterloo?",
     ["yes", "lost", "defeated"]),
]


# =============================================================================
# 6. Strategy Comparison
# =============================================================================
def run_comparison():
    """4-strategy comparison: Baseline, Always-CoT, v3 Mid-Layer, v4 Canary."""
    print('\n' + '=' * 70)
    print('  Phase A: 4-Strategy Comparison (%d Questions)' % len(QUESTIONS))
    print('=' * 70)

    strategies = [
        ('Baseline', make_baseline_detector(), False),
        ('Always CoT', make_baseline_detector(), True),
        ('v3 Mid-Layer Sniper', make_v3_detector(window=8, threshold=2.5), False),
        ('v4 Canary Trigger', make_canary_detector(window=8, threshold=2.5), False),
    ]

    all_results = {}

    for strat_name, detector, force_cot in strategies:
        print('\n  === Strategy: %s ===' % strat_name)
        correct = 0
        interventions = 0
        total_canary_peaks = []

        for i, (q, kw) in enumerate(QUESTIONS):
            if force_cot:
                cot_q = q + ' Think carefully step by step before answering.'
                answer, traj = generate_with_canary(
                    cot_q, collect_attention=False, spike_detector=None,
                    prefix='Let me think step-by-step. ')
                answer = 'Let me think step-by-step. ' + answer
                is_intervened = True
            else:
                answer, traj = surgical_cot(q, detector, name=strat_name)
                is_intervened = traj.get('intervention', False)

            is_correct = check_answer(answer, kw)
            if is_correct:
                correct += 1
            if is_intervened:
                interventions += 1

            canary_peak = traj.get('canary_peak', 0)
            total_canary_peaks.append(canary_peak)

            label = 'OK' if is_correct else 'HALLUC'
            intv = ' [INTERVENED]' if is_intervened else ''
            print('    [%d/%d] [%s]%s %s' % (
                i + 1, len(QUESTIONS), label, intv, q[:45]))

        acc = correct / len(QUESTIONS) * 100
        intv_rate = interventions / len(QUESTIONS) * 100
        print('  -> Accuracy: %d/%d = %.0f%%, Interventions: %d/%d = %.0f%%' % (
            correct, len(QUESTIONS), acc, interventions, len(QUESTIONS), intv_rate))

        all_results[strat_name] = dict(
            accuracy=acc,
            correct=correct,
            total=len(QUESTIONS),
            interventions=interventions,
            intervention_rate=intv_rate,
            mean_canary_peak=np.mean(total_canary_peaks),
        )

    return all_results


# =============================================================================
# 7. Visualization
# =============================================================================
def plot_comparison(results):
    """4-strategy comparison bar chart."""
    print('\n' + '=' * 70)
    print('  Phase B: Visualization')
    print('=' * 70)

    names = list(results.keys())
    accs = [results[n]['accuracy'] for n in names]
    intv_rates = [results[n]['intervention_rate'] for n in names]

    colors = ['#BDC3C7', '#3498DB', '#E74C3C', '#2ECC71']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Accuracy
    bars = ax1.bar(range(len(names)), accs, color=colors, edgecolor='white', width=0.7)
    for bar, acc in zip(bars, accs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 '%.0f%%' % acc, ha='center', fontweight='bold', fontsize=12)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=15, ha='right', fontsize=9)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(0, 100)
    ax1.set_title('Strategy Accuracy Comparison', fontweight='bold', fontsize=13)
    ax1.axhline(y=accs[0], color='gray', linestyle='--', alpha=0.5, label='Baseline')

    # Intervention Rate
    bars2 = ax2.bar(range(len(names)), intv_rates, color=colors, edgecolor='white', width=0.7)
    for bar, rate in zip(bars2, intv_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 '%.0f%%' % rate, ha='center', fontweight='bold', fontsize=12)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=15, ha='right', fontsize=9)
    ax2.set_ylabel('Intervention Rate (%)')
    ax2.set_ylim(0, 110)
    ax2.set_title('Intervention Rate (lower = more efficient)', fontweight='bold', fontsize=13)

    # Monitoring cost annotation
    costs = {
        'Baseline': '0 heads',
        'Always CoT': '0 heads',
        'v3 Mid-Layer Sniper': '288 heads\n(9 layers)',
        'v4 Canary Trigger': '★ 3 heads\n(99% savings)',
    }
    for i, name in enumerate(names):
        ax2.text(i, intv_rates[i] + 8, costs[name],
                 ha='center', fontsize=8, color=colors[i], fontstyle='italic')

    fig.suptitle('Canary Trigger Experiment\n'
                 '"Don\'t kill the canary. Listen to its song."',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    chart_path = os.path.join(RESULTS_DIR, 'canary_comparison.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: %s' % chart_path)

    return chart_path


def plot_canary_anatomy(results):
    """Show canary head signal strength."""
    fig, ax = plt.subplots(figsize=(8, 4))

    names = list(results.keys())
    peaks = [results[n]['mean_canary_peak'] for n in names]
    colors = ['#BDC3C7', '#3498DB', '#E74C3C', '#2ECC71']

    bars = ax.barh(range(len(names)), peaks, color=colors, edgecolor='white')
    for bar, val in zip(bars, peaks):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                '%.3f' % val, va='center', fontsize=10)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel('Mean Canary Head Entropy (bits)')
    ax.set_title('Canary Head Signal Strength\n'
                 'L10H17 + L14H14 + L18H27 (mean)',
                 fontweight='bold', fontsize=12)
    ax.invert_yaxis()
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, 'canary_signal.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: %s' % path)
    return path


# =============================================================================
# 8. Main
# =============================================================================
def main():
    start_time = time.time()

    results = run_comparison()
    chart_path = plot_comparison(results)
    signal_path = plot_canary_anatomy(results)

    elapsed = time.time() - start_time

    # Summary
    print('\n' + '=' * 70)
    print('  CANARY TRIGGER — FINAL RESULTS')
    print('=' * 70)
    print('\n  %-22s  %-8s  %-12s  %-10s' % (
        'Strategy', 'Acc', 'Intv Rate', 'Canary Peak'))
    print('  ' + '-' * 55)
    for name, r in results.items():
        print('  %-22s  %.0f%%     %.0f%%          %.3f' % (
            name, r['accuracy'], r['intervention_rate'], r['mean_canary_peak']))

    # v3 vs v4 comparison
    v3_acc = results.get('v3 Mid-Layer Sniper', {}).get('accuracy', 0)
    v4_acc = results.get('v4 Canary Trigger', {}).get('accuracy', 0)
    v3_intv = results.get('v3 Mid-Layer Sniper', {}).get('intervention_rate', 0)
    v4_intv = results.get('v4 Canary Trigger', {}).get('intervention_rate', 0)

    print('\n  === v3 vs v4 Canary (head-count savings) ===')
    print('  v3: 9 layers × 32 heads = 288 neurons monitored')
    print('  v4: 3 specific heads = 3 neurons monitored')
    print('  Monitoring reduction: %.0f×' % (288 / 3))
    print('  Accuracy: v3 = %.0f%% → v4 = %.0f%% (Δ = %+.0f%%)' % (
        v3_acc, v4_acc, v4_acc - v3_acc))
    print('  Intervention: v3 = %.0f%% → v4 = %.0f%%' % (v3_intv, v4_intv))

    print('\n  Total elapsed: %.1f seconds' % elapsed)

    # Save JSON
    json_path = os.path.join(RESULTS_DIR, 'canary_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print('  Results: %s' % json_path)


if __name__ == '__main__':
    main()
