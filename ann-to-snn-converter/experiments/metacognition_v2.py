"""
Project Metacognition v2 — Dynamic & Anatomy
=============================================================================
v1 findings: Baseline 85%, Surgical CoT 70% (intervention rate 95%)
Problem: Spike detector too sensitive (Logits-only, window=3, threshold=2.0)

v2 upgrades:
  1. Hybrid Dynamic Threshold: Attention + Logits AND condition, Z-score, window=8
  2. Layer-wise Anatomy: 32-layer attention entropy heatmap
  3. Manual token-by-token generation for full instrumentation

Author: Hiroto Funasaki
Date: 2026-02-09
"""

import os
import sys
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'results_metacognition_v2')
os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================================================================
# 1. Model Loading — Mistral-7B fp16
# =============================================================================
print('\n' + '=' * 70)
print('  Project Metacognition v2 — Dynamic & Anatomy')
print('  "The surgeon who only cuts when truly needed."')
print('=' * 70)

print('\n[Phase 0] Loading Mistral-7B (fp16)...')
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = 'mistralai/Mistral-7B-v0.1'
device = 'cuda'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map='auto',
    attn_implementation='eager',   # required for output_attentions
)
model.eval()
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

NUM_LAYERS = model.config.num_hidden_layers  # 32 for Mistral-7B
NUM_HEADS = model.config.num_attention_heads  # 32

vram_used = torch.cuda.memory_allocated() / 1e9
vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
print('  Model: %s (%d layers, %d heads)' % (MODEL_NAME, NUM_LAYERS, NUM_HEADS))
print('  VRAM: %.2f / %.2f GB' % (vram_used, vram_total))


def make_prompt(question, prefix=''):
    return 'Question: %s\nAnswer: %s' % (question, prefix)


# =============================================================================
# 2. Manual Generation Engine — Full Instrumentation
# =============================================================================
def generate_manual(question, max_new_tokens=60, temperature=0.7,
                    top_k=50, repetition_penalty=1.2,
                    collect_layer_attention=True,
                    spike_detector=None, prefix=''):
    """
    Manual token-by-token generation with full instrumentation.

    Returns (answer, trajectory) where trajectory contains:
      - logits_entropies: [float] per token
      - attn_entropies: [float] per token (mean across layers)
      - layer_entropies: [[float]*NUM_LAYERS] per token (for heatmap)
      - top_probs: [float] per token
      - tokens: [str] per token
      - spike_detected: bool
      - spike_position: int or None
    """
    prompt = make_prompt(question, prefix=prefix)
    input_ids = tokenizer(prompt, return_tensors='pt',
                          truncation=True, max_length=256).input_ids.to(device)
    prompt_len = input_ids.shape[1]

    # Trajectory storage
    logits_entropies = []
    attn_entropies = []       # mean attention entropy across all layers
    layer_entropies = []      # [step][layer] = entropy
    top_probs = []
    tokens_generated = []
    spike_detected = False
    spike_position = None

    past = None

    for step in range(max_new_tokens):
        with torch.no_grad():
            if past is None:
                out = model(input_ids,
                            output_attentions=collect_layer_attention,
                            use_cache=True)
            else:
                out = model(input_ids[:, -1:],
                            past_key_values=past,
                            output_attentions=collect_layer_attention,
                            use_cache=True)
            past = out.past_key_values

        # --- Logits Entropy ---
        logits = out.logits[:, -1, :].float()  # (1, vocab)
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log2(probs + 1e-10)
        logits_ent = -(probs * log_probs).sum(dim=-1).item()
        logits_entropies.append(logits_ent)

        # Top probability
        top_p = probs.max(dim=-1).values.item()
        top_probs.append(top_p)

        # --- Layer-wise Attention Entropy ---
        step_layer_ents = []
        if collect_layer_attention and out.attentions is not None:
            for layer_attn in out.attentions:
                # layer_attn: (batch, heads, tgt_len, src_len)
                # Get last generated token's attention distribution
                a = layer_attn[0, :, -1, :]  # (heads, src_len)
                a_log = torch.log2(a.float() + 1e-10)
                head_ent = -(a.float() * a_log).sum(dim=-1)  # (heads,)
                step_layer_ents.append(head_ent.mean().item())
            layer_entropies.append(step_layer_ents)
            attn_entropies.append(np.mean(step_layer_ents))
        else:
            layer_entropies.append([0.0] * NUM_LAYERS)
            attn_entropies.append(0.0)

        # Free attention tensors from GPU immediately
        if hasattr(out, 'attentions') and out.attentions is not None:
            del out.attentions

        # --- Spike Detection ---
        if spike_detector is not None and not spike_detected:
            is_spike = spike_detector(
                logits_entropies, attn_entropies, step)
            if is_spike:
                spike_detected = True
                spike_position = step
                # Decode partial answer before stopping
                token_id = logits.argmax(dim=-1).item()
                tokens_generated.append(tokenizer.decode([token_id]))
                break

        # --- Sampling ---
        # Apply repetition penalty
        gen_ids = input_ids[0, prompt_len:].tolist()
        for tid in set(gen_ids):
            if logits[0, tid] > 0:
                logits[0, tid] = logits[0, tid] / repetition_penalty
            else:
                logits[0, tid] = logits[0, tid] * repetition_penalty

        # Temperature + Top-k sampling
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
        token_str = tokenizer.decode([token_id])
        tokens_generated.append(token_str)

        input_ids = torch.cat([input_ids, next_id], dim=1)

        # Stop on EOS
        if token_id == tokenizer.eos_token_id:
            break

    # Decode answer
    answer_ids = input_ids[0, prompt_len:]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

    return answer, dict(
        logits_entropies=logits_entropies,
        attn_entropies=attn_entropies,
        layer_entropies=layer_entropies,
        top_probs=top_probs,
        tokens=tokens_generated,
        spike_detected=spike_detected,
        spike_position=spike_position,
        answer=answer,
        question=question,
    )


# =============================================================================
# 3. Spike Detectors — v1 (Logits-only) vs v2 (Hybrid AND)
# =============================================================================
def make_v1_detector(window=3, threshold=2.0):
    """v1: Logits-only spike detection (the 'overprotective mom')."""
    def detect(logits_ents, attn_ents, step):
        if len(logits_ents) < window + 1:
            return False
        recent = logits_ents[-(window + 1):-1]
        current = logits_ents[-1]
        mu = np.mean(recent)
        sigma = max(np.std(recent), 0.1)
        return current > mu + threshold * sigma
    return detect


def make_v2_detector(window=8, threshold=2.5):
    """v2: Hybrid AND detector (Attention + Logits Z-score)."""
    def detect(logits_ents, attn_ents, step):
        if len(logits_ents) < window + 1:
            return False

        # Logits Z-score
        logits_recent = logits_ents[-(window + 1):-1]
        logits_current = logits_ents[-1]
        logits_mu = np.mean(logits_recent)
        logits_sigma = max(np.std(logits_recent), 0.1)
        logits_z = (logits_current - logits_mu) / logits_sigma

        # Attention Z-score (mean across layers)
        attn_recent = attn_ents[-(window + 1):-1]
        attn_current = attn_ents[-1]
        attn_mu = np.mean(attn_recent)
        attn_sigma = max(np.std(attn_recent), 0.1)
        attn_z = (attn_current - attn_mu) / attn_sigma

        # AND condition: BOTH must spike
        return logits_z > threshold and attn_z > threshold
    return detect


# =============================================================================
# 4. Surgical CoT — v1 vs v2
# =============================================================================
def generate_surgical_cot(question, detector, detector_name='v2',
                          max_new_tokens=60, temperature=0.7):
    """
    Surgical CoT with configurable spike detector.
    Phase 1: Generate with spike monitoring.
    Phase 2: If spike, re-generate with CoT prefix.
    """
    answer, traj = generate_manual(
        question, max_new_tokens=max_new_tokens, temperature=temperature,
        collect_layer_attention=True, spike_detector=detector)

    if traj['spike_detected']:
        # Phase 2: CoT re-generation
        cot_prefix = 'Wait, let me think step-by-step. '
        cot_question = question + ' Think carefully step by step before answering.'
        cot_answer, cot_traj = generate_manual(
            cot_question, max_new_tokens=max_new_tokens,
            temperature=max(0.3, temperature - 0.2),
            collect_layer_attention=False,  # save VRAM on re-gen
            spike_detector=None, prefix=cot_prefix)

        return cot_prefix + cot_answer, dict(
            intervention=True,
            detector=detector_name,
            spike_position=traj['spike_position'],
            phase1_logits_ents=traj['logits_entropies'],
            phase1_attn_ents=traj['attn_entropies'],
            phase2_answer=cot_answer,
            answer=cot_prefix + cot_answer,
        )
    else:
        return answer, dict(
            intervention=False,
            detector=detector_name,
            trajectory=traj,
            answer=answer,
        )


# =============================================================================
# 5. Question Set (identical to v1)
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


# =============================================================================
# 6. Phase A: Layer-wise Anatomy — "Where does the lie begin?"
# =============================================================================
def run_layer_anatomy():
    """
    Generate answers for selected questions with full layer-wise tracking.
    Produce heatmaps: Layer (0-31) x Token Position, colored by attention entropy.
    """
    print('\n' + '=' * 70)
    print('  Phase A: Layer-wise Anatomy — Where does the lie begin?')
    print('=' * 70)

    # Select representative questions:
    # Mix of easy-correct, tricky-correct, and likely-hallucination
    anatomy_questions = [
        ("What is the capital of France?", ["Paris"], "easy"),
        ("What color is the sky on a clear day?", ["blue"], "easy"),
        ("Who was the first person to walk on Mars?", ["no one", "nobody", "hasn't", "has not", "not yet", "no human"], "tricky-correct"),
        ("Did Napoleon lose because he was short?", ["myth", "average", "not short", "no"], "tricky-correct"),
        ("What percentage of the brain do humans use?", ["100", "all", "myth"], "tricky-correct"),
        ("Who discovered America first?", ["indigenous", "Native", "Viking", "Norse"], "tricky-halluc"),
        ("What was the first programming language?", ["Fortran", "Assembly", "Plankalkul", "debated"], "tricky-halluc"),
        ("Is Pluto a planet?", ["dwarf", "no", "reclassified", "not"], "tricky-correct"),
    ]

    anatomy_results = []

    for i, (q, kw, category) in enumerate(anatomy_questions):
        print('\n  [%d/%d] %s' % (i + 1, len(anatomy_questions), q[:55]))
        answer, traj = generate_manual(q, max_new_tokens=50,
                                       collect_layer_attention=True)
        is_correct = check_answer(answer, kw)
        traj['label'] = 'correct' if is_correct else 'hallucination'
        traj['category'] = category
        anatomy_results.append(traj)
        print('         -> %s | %s' % (traj['label'], answer[:60]))

        if device == 'cuda':
            torch.cuda.empty_cache()

    # === Visualization: Layer-wise Heatmaps ===
    print('\n  Generating layer-wise heatmaps...')

    n_plots = len(anatomy_results)
    fig, axes = plt.subplots(n_plots, 1, figsize=(16, 4 * n_plots))
    if n_plots == 1:
        axes = [axes]

    fig.suptitle('Layer-wise Anatomy: Attention Entropy per Layer per Token\n'
                 '(Mistral-7B, 32 layers × N tokens)',
                 fontsize=16, fontweight='bold', y=0.995)

    for idx, traj in enumerate(anatomy_results):
        ax = axes[idx]
        # Build 2D matrix: [layer][token_pos]
        le = traj['layer_entropies']  # [token][layer]
        n_tokens = len(le)
        if n_tokens == 0:
            continue

        matrix = np.array(le).T  # [layer][token] — shape (32, n_tokens)

        # Plot heatmap
        label_icon = '✅' if traj['label'] == 'correct' else '❌'
        im = ax.imshow(matrix, aspect='auto', interpolation='nearest',
                       cmap='inferno', origin='lower')
        ax.set_ylabel('Layer')
        ax.set_xlabel('Token Position')
        title = '%s %s: "%s"' % (label_icon, traj['category'],
                                  traj['question'][:50])
        ax.set_title(title, fontsize=11, fontweight='bold', loc='left')

        # Token labels on x-axis (sparse)
        tokens = traj['tokens'][:n_tokens]
        tick_step = max(1, n_tokens // 15)
        tick_pos = list(range(0, n_tokens, tick_step))
        tick_labels = [tokens[i][:6] if i < len(tokens) else '' for i in tick_pos]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, fontsize=7, rotation=45)

        # Layer ticks
        ax.set_yticks([0, 7, 15, 23, 31])
        ax.set_yticklabels(['L0', 'L7', 'L15', 'L23', 'L31'])

        # Colorbar
        cb = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.01)
        cb.set_label('Attn Entropy (bits)', fontsize=8)

        # Mark answer text
        ax.text(1.0, -0.08, 'Answer: %s' % traj['answer'][:70],
                transform=ax.transAxes, fontsize=8, ha='right',
                fontstyle='italic', color='#666666')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(RESULTS_DIR, 'layer_anatomy_heatmap.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: %s' % out_path)

    # === Summary: Mean entropy by layer group (shallow vs deep) ===
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
    fig2.suptitle('Layer-wise Anatomy: Shallow vs Deep Layers\n'
                  'Hypothesis: Hallucinations arise in deep layers',
                  fontsize=14, fontweight='bold')

    correct_trajs = [t for t in anatomy_results if t['label'] == 'correct']
    halluc_trajs = [t for t in anatomy_results if t['label'] == 'hallucination']

    for ax, trajs, label, color in [
            (axes2[0], correct_trajs, 'Correct', '#4A90D9'),
            (axes2[1], halluc_trajs, 'Hallucination', '#E74C3C')]:
        if not trajs:
            ax.text(0.5, 0.5, 'No samples', transform=ax.transAxes,
                    ha='center', fontsize=14)
            ax.set_title(label)
            continue

        # Average entropy per layer across all tokens and all questions
        all_layer_means = np.zeros(NUM_LAYERS)
        count = 0
        for t in trajs:
            le = np.array(t['layer_entropies'])  # [tokens, layers]
            if le.shape[0] > 0:
                all_layer_means += le.mean(axis=0)
                count += 1
        if count > 0:
            all_layer_means /= count

        bars = ax.bar(range(NUM_LAYERS), all_layer_means, color=color, alpha=0.8)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Mean Attention Entropy (bits)')
        ax.set_title('%s (N=%d)' % (label, len(trajs)), fontweight='bold')
        ax.set_xticks([0, 7, 15, 23, 31])
        ax.set_xticklabels(['L0', 'L7', 'L15', 'L23', 'L31'])

        # Highlight deep vs shallow
        shallow_mean = all_layer_means[:10].mean()
        deep_mean = all_layer_means[22:].mean()
        ax.axhline(shallow_mean, color='green', linestyle='--', alpha=0.5,
                   label='Shallow (L0-9): %.2f' % shallow_mean)
        ax.axhline(deep_mean, color='orange', linestyle='--', alpha=0.5,
                   label='Deep (L22-31): %.2f' % deep_mean)
        ax.legend(fontsize=9)

    plt.tight_layout()
    out_path2 = os.path.join(RESULTS_DIR, 'layer_anatomy_summary.png')
    plt.savefig(out_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: %s' % out_path2)

    return anatomy_results


# =============================================================================
# 7. Phase B: 4-Strategy Comparison
# =============================================================================
def run_comparison(anatomy_results=None):
    """
    Compare 4 strategies on all 20 tricky questions:
      1. Baseline (no intervention)
      2. Always CoT
      3. Surgical CoT v1 (Logits-only, window=3, threshold=2.0)
      4. Surgical CoT v2 (Hybrid AND, window=8, threshold=2.5)
    """
    print('\n' + '=' * 70)
    print('  Phase B: 4-Strategy Comparison')
    print('  "From overprotective mom to calm surgeon"')
    print('=' * 70)

    v1_detector = make_v1_detector(window=3, threshold=2.0)
    v2_detector = make_v2_detector(window=8, threshold=2.5)

    strategies = [
        ('Baseline', None),
        ('Always CoT', 'always'),
        ('Surgical v1 (Logits)', v1_detector),
        ('Surgical v2 (Hybrid)', v2_detector),
    ]

    results = {name: {'correct': 0, 'total': 0, 'interventions': 0,
                       'details': []}
               for name, _ in strategies}

    for qi, (question, keywords) in enumerate(TRICKY_QUESTIONS):
        print('\n  [%d/%d] %s' % (qi + 1, len(TRICKY_QUESTIONS), question[:55]))

        for strategy_name, detector in strategies:
            if detector is None:
                # Baseline: normal generation
                answer, traj = generate_manual(
                    question, collect_layer_attention=False)
                intervened = False
            elif detector == 'always':
                # Always CoT
                cot_prefix = 'Wait, let me think step-by-step. '
                cot_q = question + ' Think carefully step by step before answering.'
                answer, traj = generate_manual(
                    cot_q, collect_layer_attention=False, prefix=cot_prefix)
                answer = cot_prefix + answer
                intervened = True
            else:
                # Surgical CoT (v1 or v2)
                detector_name = strategy_name
                answer, traj = generate_surgical_cot(
                    question, detector, detector_name)
                intervened = traj.get('intervention', False)

            is_correct = check_answer(answer, keywords)
            tag = '[OK]' if is_correct else '[WRONG]'
            extra = ' [SURGICAL]' if intervened and 'Surgical' in strategy_name else ''
            print('    %s: %s%s %s' % (strategy_name, tag, extra,
                                        answer[:55]))

            results[strategy_name]['total'] += 1
            if is_correct:
                results[strategy_name]['correct'] += 1
            if intervened:
                results[strategy_name]['interventions'] += 1

            results[strategy_name]['details'].append(dict(
                question=question,
                answer=answer[:200],
                correct=is_correct,
                intervened=intervened,
            ))

            if device == 'cuda':
                torch.cuda.empty_cache()

    # === Print Summary ===
    print('\n' + '=' * 70)
    print('  Phase B Results Summary')
    print('=' * 70)
    for name, data in results.items():
        acc = data['correct'] / max(data['total'], 1) * 100
        intv = data['interventions'] / max(data['total'], 1) * 100
        print('  %-25s  Accuracy: %5.1f%%  Interventions: %d/%d (%.0f%%)'
              % (name, acc, data['interventions'], data['total'], intv))

    # === Visualization: 4-Strategy Bar Chart ===
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    fig.suptitle('Surgical CoT v2: Metacognitive Intervention\n'
                 '(Mistral-7B, N=20 tricky questions)',
                 fontsize=16, fontweight='bold')

    names = list(results.keys())
    accuracies = [results[n]['correct'] / max(results[n]['total'], 1) * 100
                  for n in names]
    intv_rates = [results[n]['interventions'] / max(results[n]['total'], 1) * 100
                  for n in names]

    colors = ['#888888', '#E8A838', '#E74C3C', '#27AE60']
    bars = ax.bar(names, accuracies, color=colors, edgecolor='white',
                  linewidth=1.5, width=0.6)

    # Add value labels
    for bar, acc, intv in zip(bars, accuracies, intv_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                '%.1f%%' % acc, ha='center', fontsize=16, fontweight='bold')
        if intv > 0 and bar.get_x() > 0:  # skip baseline
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() / 2,
                    'Intervention: %.0f%%' % intv,
                    ha='center', fontsize=10, fontstyle='italic',
                    color='white', alpha=0.9)

    # Baseline reference line
    baseline_acc = accuracies[0]
    ax.axhline(baseline_acc, color='red', linestyle='--', alpha=0.5,
               label='Baseline (%.1f%%)' % baseline_acc)
    ax.set_ylabel('Accuracy (%)', fontsize=13)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=11)

    # Labels
    labels_map = {
        'Baseline': 'Baseline\n(No Intervention)',
        'Always CoT': 'Always CoT\n(Always Think)',
        'Surgical v1 (Logits)': 'Surgical v1\n(Logits-only)',
        'Surgical v2 (Hybrid)': 'Surgical v2\n(Attn+Logits AND)',
    }
    ax.set_xticklabels([labels_map.get(n, n) for n in names], fontsize=11)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, 'v2_comparison_chart.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: %s' % out_path)

    return results


# =============================================================================
# 8. Main
# =============================================================================
def main():
    t0 = time.time()

    # Phase A: Layer-wise Anatomy
    anatomy_results = run_layer_anatomy()

    # Phase B: 4-Strategy Comparison
    comparison_results = run_comparison(anatomy_results)

    elapsed = time.time() - t0

    # === Save JSON Summary ===
    summary = dict(
        model=MODEL_NAME,
        device=device,
        vram_gb=torch.cuda.memory_allocated() / 1e9,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        phase_a=dict(
            questions_analyzed=len(anatomy_results) if anatomy_results else 0,
        ),
        phase_b=dict(
            strategies={},
        ),
        elapsed_seconds=elapsed,
    )

    for name, data in comparison_results.items():
        acc = data['correct'] / max(data['total'], 1)
        intv = data['interventions'] / max(data['total'], 1)
        summary['phase_b']['strategies'][name] = dict(
            accuracy=round(acc, 4),
            intervention_rate=round(intv, 4),
            correct=data['correct'],
            total=data['total'],
        )

    json_path = os.path.join(RESULTS_DIR, 'metacognition_v2_results.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print('\n  Saved: %s' % json_path)

    # === Final Banner ===
    print('\n' + '=' * 70)
    print('  Project Metacognition v2 complete!')
    print('  "The surgeon who only cuts when truly needed."')
    print('  Elapsed: %.0f seconds (%.1f min)' % (elapsed, elapsed / 60))
    print('=' * 70)

    # Print comparison table
    print('\n  Strategy Comparison:')
    print('  %-25s  Acc    Intv' % '')
    print('  ' + '-' * 50)
    for name, data in comparison_results.items():
        acc = data['correct'] / max(data['total'], 1) * 100
        intv = data['interventions'] / max(data['total'], 1) * 100
        marker = ' <-- TARGET' if 'v2' in name else ''
        print('  %-25s  %5.1f%%  %5.1f%%%s' % (name, acc, intv, marker))


if __name__ == '__main__':
    main()
