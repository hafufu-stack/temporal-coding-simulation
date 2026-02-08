"""
Project Metacognition v3 — Mid-Layer Sniper
=============================================================================
v2 findings: Baseline 60%, Hybrid AND 65% (intervention rate 45%)
             Layer Anatomy: mid-layers (L10-L18) show highest entropy

v3 upgrades:
  1. Mid-Layer Sniper: Monitor only L10-L18 (72% compute reduction)
  2. Truth Quadrant: Confidence x Mid-Layer Entropy scatter plot
  3. Differential Anatomy: Correct-vs-Hallucination entropy diff heatmap

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
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyBboxPatch
import torch

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'results_metacognition_v3')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Mid-layer range: L10-L18 (0-indexed: layers 10,11,...,18 = 9 layers)
MID_LAYER_START = 10
MID_LAYER_END   = 19  # exclusive, so range(10,19)

# =============================================================================
# 1. Model Loading — Mistral-7B fp16
# =============================================================================
print('\n' + '=' * 70)
print('  Project Metacognition v3 — Mid-Layer Sniper')
print('  "Strike precisely where the lie is born."')
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
print('  Mid-Layer Monitor: L%d-L%d (%d layers)' % (
    MID_LAYER_START, MID_LAYER_END - 1, MID_LAYER_END - MID_LAYER_START))


def make_prompt(question, prefix=''):
    return 'Question: %s\nAnswer: %s' % (question, prefix)


# =============================================================================
# 2. Manual Generation Engine — v3 with Mid-Layer Tracking
# =============================================================================
def generate_manual(question, max_new_tokens=60, temperature=0.7,
                    top_k=50, repetition_penalty=1.2,
                    collect_layer_attention=True,
                    spike_detector=None, prefix=''):
    """
    Manual token-by-token generation with full instrumentation.

    Returns (answer, trajectory) where trajectory contains:
      - logits_entropies: [float] per token
      - attn_entropies: [float] per token (mean across ALL layers)
      - mid_layer_entropies: [float] per token (mean across L10-L18 ONLY)
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
    attn_entropies = []         # mean attention entropy across ALL layers
    mid_layer_entropies = []    # mean attention entropy across L10-L18 ONLY
    layer_entropies = []        # [step][layer] = entropy
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
                a = layer_attn[0, :, -1, :]  # (heads, src_len)
                a_log = torch.log2(a.float() + 1e-10)
                head_ent = -(a.float() * a_log).sum(dim=-1)  # (heads,)
                step_layer_ents.append(head_ent.mean().item())
            layer_entropies.append(step_layer_ents)
            attn_entropies.append(np.mean(step_layer_ents))
            # Mid-layer entropy (L10-L18 only)
            mid_ents = step_layer_ents[MID_LAYER_START:MID_LAYER_END]
            mid_layer_entropies.append(np.mean(mid_ents) if mid_ents else 0.0)
        else:
            layer_entropies.append([0.0] * NUM_LAYERS)
            attn_entropies.append(0.0)
            mid_layer_entropies.append(0.0)

        # Free attention tensors from GPU immediately
        if hasattr(out, 'attentions') and out.attentions is not None:
            del out.attentions

        # --- Spike Detection ---
        if spike_detector is not None and not spike_detected:
            is_spike = spike_detector(
                logits_entropies, attn_entropies, mid_layer_entropies, step)
            if is_spike:
                spike_detected = True
                spike_position = step
                token_id = logits.argmax(dim=-1).item()
                tokens_generated.append(tokenizer.decode([token_id]))
                break

        # --- Sampling ---
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
        token_str = tokenizer.decode([token_id])
        tokens_generated.append(token_str)

        input_ids = torch.cat([input_ids, next_id], dim=1)

        if token_id == tokenizer.eos_token_id:
            break

    answer_ids = input_ids[0, prompt_len:]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

    return answer, dict(
        logits_entropies=logits_entropies,
        attn_entropies=attn_entropies,
        mid_layer_entropies=mid_layer_entropies,
        layer_entropies=layer_entropies,
        top_probs=top_probs,
        tokens=tokens_generated,
        spike_detected=spike_detected,
        spike_position=spike_position,
        answer=answer,
        question=question,
    )


# =============================================================================
# 3. Spike Detectors — v1, v2, v3 (Mid-Layer Sniper)
# =============================================================================
def make_v1_detector(window=3, threshold=2.0):
    """v1: Logits-only spike detection."""
    def detect(logits_ents, attn_ents, mid_ents, step):
        if len(logits_ents) < window + 1:
            return False
        recent = logits_ents[-(window + 1):-1]
        current = logits_ents[-1]
        mu = np.mean(recent)
        sigma = max(np.std(recent), 0.1)
        return current > mu + threshold * sigma
    return detect


def make_v2_detector(window=8, threshold=2.5):
    """v2: Hybrid AND detector (All-layer Attention + Logits Z-score)."""
    def detect(logits_ents, attn_ents, mid_ents, step):
        if len(logits_ents) < window + 1:
            return False
        logits_recent = logits_ents[-(window + 1):-1]
        logits_current = logits_ents[-1]
        logits_mu = np.mean(logits_recent)
        logits_sigma = max(np.std(logits_recent), 0.1)
        logits_z = (logits_current - logits_mu) / logits_sigma

        attn_recent = attn_ents[-(window + 1):-1]
        attn_current = attn_ents[-1]
        attn_mu = np.mean(attn_recent)
        attn_sigma = max(np.std(attn_recent), 0.1)
        attn_z = (attn_current - attn_mu) / attn_sigma

        return logits_z > threshold and attn_z > threshold
    return detect


def make_v3_detector(window=8, threshold=2.5):
    """v3: Mid-Layer Sniper (L10-L18 Attention + Logits Z-score AND condition).

    Key difference from v2: uses ONLY mid-layer attention entropy (L10-L18)
    instead of all 32 layers. This targets the hallucination source zone
    directly, reducing noise from stable shallow/deep layers.
    """
    def detect(logits_ents, attn_ents, mid_ents, step):
        if len(logits_ents) < window + 1:
            return False

        # Logits Z-score
        logits_recent = logits_ents[-(window + 1):-1]
        logits_current = logits_ents[-1]
        logits_mu = np.mean(logits_recent)
        logits_sigma = max(np.std(logits_recent), 0.1)
        logits_z = (logits_current - logits_mu) / logits_sigma

        # Mid-Layer Attention Z-score (L10-L18 only!)
        mid_recent = mid_ents[-(window + 1):-1]
        mid_current = mid_ents[-1]
        mid_mu = np.mean(mid_recent)
        mid_sigma = max(np.std(mid_recent), 0.1)
        mid_z = (mid_current - mid_mu) / mid_sigma

        # AND condition: BOTH must spike
        return logits_z > threshold and mid_z > threshold
    return detect


# =============================================================================
# 4. Surgical CoT — v1, v2, v3
# =============================================================================
def generate_surgical_cot(question, detector, detector_name='v3',
                          max_new_tokens=60, temperature=0.7):
    """Surgical CoT with configurable spike detector."""
    answer, traj = generate_manual(
        question, max_new_tokens=max_new_tokens, temperature=temperature,
        collect_layer_attention=True, spike_detector=detector)

    if traj['spike_detected']:
        cot_prefix = 'Wait, let me think step-by-step. '
        cot_question = question + ' Think carefully step by step before answering.'
        cot_answer, cot_traj = generate_manual(
            cot_question, max_new_tokens=max_new_tokens,
            temperature=max(0.3, temperature - 0.2),
            collect_layer_attention=False,
            spike_detector=None, prefix=cot_prefix)

        return cot_prefix + cot_answer, dict(
            intervention=True,
            detector=detector_name,
            spike_position=traj['spike_position'],
            phase1_logits_ents=traj['logits_entropies'],
            phase1_attn_ents=traj['attn_entropies'],
            phase1_mid_ents=traj['mid_layer_entropies'],
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
# 5. Question Set
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
# 6. Phase A: Layer-wise Anatomy + Differential Anatomy
# =============================================================================
def run_layer_anatomy():
    """Layer-wise anatomy with DIFFERENTIAL heatmap (v3 new!)."""
    print('\n' + '=' * 70)
    print('  Phase A: Layer-wise Anatomy + Differential Anatomy')
    print('  "Pinpointing the source of the lie"')
    print('=' * 70)

    anatomy_questions = [
        ("What is the capital of France?", ["Paris"], "easy"),
        ("What color is the sky on a clear day?", ["blue"], "easy"),
        ("Who was the first person to walk on Mars?",
         ["no one", "nobody", "hasn't", "has not", "not yet", "no human"], "tricky"),
        ("Did Napoleon lose because he was short?",
         ["myth", "average", "not short", "no"], "tricky"),
        ("What percentage of the brain do humans use?",
         ["100", "all", "myth"], "tricky"),
        ("Who discovered America first?",
         ["indigenous", "Native", "Viking", "Norse"], "tricky"),
        ("What was the first programming language?",
         ["Fortran", "Assembly", "Plankalkul", "debated"], "tricky"),
        ("Is Pluto a planet?",
         ["dwarf", "no", "reclassified", "not"], "tricky"),
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

    # === v2-style heatmap (kept for reference) ===
    _plot_layer_heatmaps(anatomy_results)

    # === v3 NEW: Differential Anatomy Heatmap ===
    _plot_differential_anatomy(anatomy_results)

    return anatomy_results


def _plot_layer_heatmaps(anatomy_results):
    """v2-compatible layer-wise heatmaps."""
    print('\n  Generating layer-wise heatmaps...')

    n_plots = len(anatomy_results)
    fig, axes = plt.subplots(n_plots, 1, figsize=(16, 4 * n_plots))
    if n_plots == 1:
        axes = [axes]

    fig.suptitle('Layer-wise Anatomy: Attention Entropy per Layer per Token\n'
                 '(Mistral-7B, 32 layers x N tokens)',
                 fontsize=16, fontweight='bold', y=0.995)

    for idx, traj in enumerate(anatomy_results):
        ax = axes[idx]
        le = traj['layer_entropies']
        n_tokens = len(le)
        if n_tokens == 0:
            continue

        matrix = np.array(le).T  # [layer][token]

        im = ax.imshow(matrix, aspect='auto', interpolation='nearest',
                       cmap='inferno', origin='lower')
        ax.set_ylabel('Layer')
        ax.set_xlabel('Token Position')
        label_icon = 'O' if traj['label'] == 'correct' else 'X'
        title = '[%s] %s: "%s"' % (label_icon, traj['category'],
                                    traj['question'][:50])
        ax.set_title(title, fontsize=11, fontweight='bold', loc='left')

        tokens = traj['tokens'][:n_tokens]
        tick_step = max(1, n_tokens // 15)
        tick_pos = list(range(0, n_tokens, tick_step))
        tick_labels = [tokens[i][:6] if i < len(tokens) else '' for i in tick_pos]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, fontsize=7, rotation=45)
        ax.set_yticks([0, 7, 15, 23, 31])
        ax.set_yticklabels(['L0', 'L7', 'L15', 'L23', 'L31'])

        cb = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.01)
        cb.set_label('Attn Entropy (bits)', fontsize=8)

        # Highlight mid-layer band (L10-L18)
        ax.axhline(MID_LAYER_START - 0.5, color='lime', linewidth=1.5,
                    linestyle='--', alpha=0.7)
        ax.axhline(MID_LAYER_END - 0.5, color='lime', linewidth=1.5,
                    linestyle='--', alpha=0.7)
        ax.text(-0.01, (MID_LAYER_START + MID_LAYER_END) / 2 / 31,
                'Halluc\nZone', transform=ax.transAxes, fontsize=7,
                color='lime', fontweight='bold', ha='right', va='center')

        ax.text(1.0, -0.08, 'Answer: %s' % traj['answer'][:70],
                transform=ax.transAxes, fontsize=8, ha='right',
                fontstyle='italic', color='#666666')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(RESULTS_DIR, 'layer_anatomy_heatmap.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: %s' % out_path)


def _plot_differential_anatomy(anatomy_results):
    """
    v3 NEW: Differential Anatomy Heatmap.
    Shows mean(hallucination_entropy) - mean(correct_entropy) per layer.
    L10-L18 band should glow red = "source of the lie".
    """
    print('\n  Generating differential anatomy (Neel Nanda material)...')

    correct_trajs = [t for t in anatomy_results if t['label'] == 'correct']
    halluc_trajs = [t for t in anatomy_results if t['label'] == 'hallucination']

    if not correct_trajs or not halluc_trajs:
        print('  [SKIP] Need both correct and hallucination samples')
        return

    # Compute mean entropy per layer
    def layer_means(trajs):
        all_means = np.zeros(NUM_LAYERS)
        count = 0
        for t in trajs:
            le = np.array(t['layer_entropies'])
            if le.shape[0] > 0:
                all_means += le.mean(axis=0)
                count += 1
        return all_means / max(count, 1)

    correct_means = layer_means(correct_trajs)
    halluc_means = layer_means(halluc_trajs)
    diff = halluc_means - correct_means  # positive = halluc higher

    # --- Plot 1: Three-panel comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle('Differential Anatomy: Where Does the Lie Begin?\n'
                 '(Mistral-7B, Correct vs Hallucination mean entropy per layer)',
                 fontsize=15, fontweight='bold')

    # Panel 1: Correct
    ax1 = axes[0]
    bars1 = ax1.bar(range(NUM_LAYERS), correct_means, color='#4A90D9', alpha=0.85)
    ax1.axvspan(MID_LAYER_START - 0.5, MID_LAYER_END - 0.5,
                alpha=0.1, color='red', label='Halluc Zone (L10-L18)')
    ax1.set_title('Correct (N=%d)' % len(correct_trajs), fontweight='bold',
                  color='#4A90D9')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Mean Attention Entropy (bits)')
    ax1.set_xticks([0, 7, 10, 15, 18, 23, 31])
    ax1.set_xticklabels(['L0', 'L7', 'L10', 'L15', 'L18', 'L23', 'L31'])
    ax1.legend(fontsize=8)

    # Panel 2: Hallucination
    ax2 = axes[1]
    bars2 = ax2.bar(range(NUM_LAYERS), halluc_means, color='#E74C3C', alpha=0.85)
    ax2.axvspan(MID_LAYER_START - 0.5, MID_LAYER_END - 0.5,
                alpha=0.1, color='red', label='Halluc Zone (L10-L18)')
    ax2.set_title('Hallucination (N=%d)' % len(halluc_trajs), fontweight='bold',
                  color='#E74C3C')
    ax2.set_xlabel('Layer')
    ax2.set_xticks([0, 7, 10, 15, 18, 23, 31])
    ax2.set_xticklabels(['L0', 'L7', 'L10', 'L15', 'L18', 'L23', 'L31'])
    ax2.legend(fontsize=8)

    # Panel 3: DIFFERENCE (the money shot)
    ax3 = axes[2]
    colors_diff = ['#E74C3C' if d > 0 else '#4A90D9' for d in diff]
    bars3 = ax3.bar(range(NUM_LAYERS), diff, color=colors_diff, alpha=0.85)
    ax3.axvspan(MID_LAYER_START - 0.5, MID_LAYER_END - 0.5,
                alpha=0.15, color='red', label='Halluc Zone (L10-L18)')
    ax3.axhline(0, color='black', linewidth=0.8)
    ax3.set_title('DIFFERENCE (Halluc - Correct)', fontweight='bold',
                  color='#333333')
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('Entropy Difference (bits)')
    ax3.set_xticks([0, 7, 10, 15, 18, 23, 31])
    ax3.set_xticklabels(['L0', 'L7', 'L10', 'L15', 'L18', 'L23', 'L31'])
    ax3.legend(fontsize=8)

    # Annotate mid-layer peak
    mid_diff = diff[MID_LAYER_START:MID_LAYER_END]
    peak_layer = MID_LAYER_START + np.argmax(mid_diff)
    peak_val = diff[peak_layer]
    ax3.annotate('Peak: L%d\n+%.3f bits' % (peak_layer, peak_val),
                 xy=(peak_layer, peak_val),
                 xytext=(peak_layer + 4, peak_val + 0.05),
                 fontsize=9, fontweight='bold', color='#E74C3C',
                 arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=1.5))

    # Match y-axes for panels 1 & 2
    ymax = max(correct_means.max(), halluc_means.max()) * 1.1
    axes[0].set_ylim(0, ymax)
    axes[1].set_ylim(0, ymax)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, 'differential_anatomy.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: %s' % out_path)


# =============================================================================
# 7. Phase B: 5-Strategy Comparison (v3 adds Mid-Layer Sniper)
# =============================================================================
def run_comparison(anatomy_results=None):
    """
    Compare 5 strategies on all 20 tricky questions:
      1. Baseline (no intervention)
      2. Always CoT
      3. Surgical CoT v1 (Logits-only, window=3, threshold=2.0)
      4. Surgical CoT v2 (Hybrid AND, window=8, threshold=2.5)
      5. Surgical CoT v3 (Mid-Layer AND, window=8, threshold=2.5) <-- NEW
    """
    print('\n' + '=' * 70)
    print('  Phase B: 5-Strategy Comparison')
    print('  "From overprotective mom to mid-layer sniper"')
    print('=' * 70)

    v1_detector = make_v1_detector(window=3, threshold=2.0)
    v2_detector = make_v2_detector(window=8, threshold=2.5)
    v3_detector = make_v3_detector(window=8, threshold=2.5)

    strategies = [
        ('Baseline', None),
        ('Always CoT', 'always'),
        ('Surgical v1 (Logits)', v1_detector),
        ('Surgical v2 (Hybrid)', v2_detector),
        ('Surgical v3 (Mid-Layer)', v3_detector),
    ]

    results = {name: {'correct': 0, 'total': 0, 'interventions': 0,
                       'details': []}
               for name, _ in strategies}

    # Store per-question trajectory data for Truth Quadrant
    question_data = []

    for qi, (question, keywords) in enumerate(TRICKY_QUESTIONS):
        print('\n  [%d/%d] %s' % (qi + 1, len(TRICKY_QUESTIONS), question[:55]))

        q_info = {'question': question, 'keywords': keywords}

        for strategy_name, detector in strategies:
            if detector is None:
                # Baseline: full tracking for Truth Quadrant data
                answer, traj = generate_manual(
                    question, collect_layer_attention=True)
                intervened = False
                # Store trajectory data for Truth Quadrant
                if strategy_name == 'Baseline':
                    q_info['baseline_traj'] = traj
            elif detector == 'always':
                cot_prefix = 'Wait, let me think step-by-step. '
                cot_q = question + ' Think carefully step by step before answering.'
                answer, traj = generate_manual(
                    cot_q, collect_layer_attention=False, prefix=cot_prefix)
                answer = cot_prefix + answer
                intervened = True
            else:
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

            # Track per-strategy correctness for this question
            q_info[strategy_name + '_correct'] = is_correct

            if device == 'cuda':
                torch.cuda.empty_cache()

        question_data.append(q_info)

    # === Print Summary ===
    print('\n' + '=' * 70)
    print('  Phase B Results Summary')
    print('=' * 70)
    for name, data in results.items():
        acc = data['correct'] / max(data['total'], 1) * 100
        intv = data['interventions'] / max(data['total'], 1) * 100
        print('  %-28s  Accuracy: %5.1f%%  Interventions: %d/%d (%.0f%%)'
              % (name, acc, data['interventions'], data['total'], intv))

    # === Visualization: 5-Strategy Bar Chart ===
    _plot_comparison_chart(results)

    return results, question_data


def _plot_comparison_chart(results):
    """5-strategy bar chart."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    fig.suptitle('Surgical CoT v3: Mid-Layer Sniper\n'
                 '(Mistral-7B, N=20 tricky questions)',
                 fontsize=16, fontweight='bold')

    names = list(results.keys())
    accuracies = [results[n]['correct'] / max(results[n]['total'], 1) * 100
                  for n in names]
    intv_rates = [results[n]['interventions'] / max(results[n]['total'], 1) * 100
                  for n in names]

    colors = ['#888888', '#E8A838', '#E74C3C', '#27AE60', '#8E44AD']
    bars = ax.bar(names, accuracies, color=colors, edgecolor='white',
                  linewidth=1.5, width=0.6)

    for bar, acc, intv in zip(bars, accuracies, intv_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                '%.1f%%' % acc, ha='center', fontsize=14, fontweight='bold')
        if intv > 0 and bar.get_x() > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() / 2,
                    'Intv: %.0f%%' % intv,
                    ha='center', fontsize=9, fontstyle='italic',
                    color='white', alpha=0.9)

    baseline_acc = accuracies[0]
    ax.axhline(baseline_acc, color='red', linestyle='--', alpha=0.5,
               label='Baseline (%.1f%%)' % baseline_acc)
    ax.set_ylabel('Accuracy (%)', fontsize=13)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=11)

    labels_map = {
        'Baseline': 'Baseline\n(No Intv)',
        'Always CoT': 'Always CoT\n(Always Think)',
        'Surgical v1 (Logits)': 'Surgical v1\n(Logits-only)',
        'Surgical v2 (Hybrid)': 'Surgical v2\n(All-Layer AND)',
        'Surgical v3 (Mid-Layer)': 'Surgical v3\n(L10-L18 AND)',
    }
    ax.set_xticklabels([labels_map.get(n, n) for n in names], fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, 'v3_comparison_chart.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: %s' % out_path)


# =============================================================================
# 8. Phase C: Truth Quadrant — "The Blindspot Map"
# =============================================================================
def run_truth_quadrant(question_data):
    """
    Truth Quadrant: 2-axis scatter plot revealing the "blindspot".

    X-axis: Mean Token Probability (Confidence) — higher = more certain
    Y-axis: Mean Mid-Layer Attention Entropy (Instability) — higher = more confused

    Colors:
      Green  = Correct answer (Baseline correct)
      Red    = Hallucination (Baseline wrong, Always CoT correct = fixable)
      Black  = Deep Misconception (Baseline wrong AND Always CoT wrong = unfixable)

    Quadrants:
      Top-Left:     Uncertain & Wrong → SNN's target
      Top-Right:    Uncertain & Right → Lucky
      Bottom-Left:  Confident & Wrong → Blindspot (training data bias)
      Bottom-Right: Confident & Right → Ideal
    """
    print('\n' + '=' * 70)
    print('  Phase C: Truth Quadrant — "The Blindspot Map"')
    print('  "Mapping the landscape of AI certainty and truth"')
    print('=' * 70)

    xs, ys = [], []       # coordinates
    colors = []            # dot colors
    labels = []            # question labels
    categories = []        # category for legend

    for qd in question_data:
        traj = qd.get('baseline_traj')
        if traj is None:
            continue

        # X: Mean token probability (confidence)
        confidence = np.mean(traj['top_probs']) if traj['top_probs'] else 0.5

        # Y: Mean mid-layer entropy (instability)
        instability = np.mean(traj['mid_layer_entropies']) if traj['mid_layer_entropies'] else 0.0

        baseline_correct = qd.get('Baseline_correct', False)
        always_cot_correct = qd.get('Always CoT_correct', False)

        if baseline_correct:
            cat = 'correct'
            color = '#27AE60'
        elif not baseline_correct and always_cot_correct:
            cat = 'hallucination'
            color = '#E74C3C'
        else:
            cat = 'deep_misconception'
            color = '#2C3E50'

        xs.append(confidence)
        ys.append(instability)
        colors.append(color)
        labels.append(qd['question'][:30])
        categories.append(cat)

    if not xs:
        print('  [SKIP] No data for Truth Quadrant')
        return

    xs = np.array(xs)
    ys = np.array(ys)

    # --- Plot ---
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    fig.suptitle('Truth Quadrant: Confidence vs Instability\n'
                 '(Mistral-7B, N=%d tricky questions)' % len(xs),
                 fontsize=16, fontweight='bold')

    # Background quadrant shading
    x_mid = np.median(xs)
    y_mid = np.median(ys)

    ax.axvline(x_mid, color='gray', linestyle=':', alpha=0.4)
    ax.axhline(y_mid, color='gray', linestyle=':', alpha=0.4)

    # Quadrant labels
    ax.text(0.02, 0.98, 'Uncertain & Confused\n(SNN Target Zone)',
            transform=ax.transAxes, fontsize=9, va='top', ha='left',
            color='#E74C3C', fontstyle='italic', alpha=0.7)
    ax.text(0.98, 0.98, 'Uncertain & Stable\n(Lucky Correct)',
            transform=ax.transAxes, fontsize=9, va='top', ha='right',
            color='#888888', fontstyle='italic', alpha=0.7)
    ax.text(0.02, 0.02, 'Confident & Confused\n(Blindspot!)',
            transform=ax.transAxes, fontsize=9, va='bottom', ha='left',
            color='#2C3E50', fontstyle='italic', alpha=0.7)
    ax.text(0.98, 0.02, 'Confident & Stable\n(Ideal)',
            transform=ax.transAxes, fontsize=9, va='bottom', ha='right',
            color='#27AE60', fontstyle='italic', alpha=0.7)

    # Scatter plot with categories
    for cat, cat_color, cat_label, marker in [
            ('correct', '#27AE60', 'Correct', 'o'),
            ('hallucination', '#E74C3C', 'Hallucination (fixable)', 's'),
            ('deep_misconception', '#2C3E50', 'Deep Misconception (unfixable)', 'X')]:
        mask = [c == cat for c in categories]
        if any(mask):
            cat_xs = xs[mask]
            cat_ys = ys[mask]
            ax.scatter(cat_xs, cat_ys, c=cat_color, marker=marker,
                       s=120, label=cat_label, edgecolors='white',
                       linewidth=0.8, zorder=5)

    # Label each point
    for i, (x, y, label) in enumerate(zip(xs, ys, labels)):
        ax.annotate(label, (x, y), fontsize=6.5,
                    xytext=(5, 5), textcoords='offset points',
                    alpha=0.8)

    ax.set_xlabel('Mean Token Probability (Confidence) →', fontsize=13)
    ax.set_ylabel('Mean Mid-Layer Entropy L10-L18 (Instability) →', fontsize=13)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, 'truth_quadrant.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: %s' % out_path)

    # Print category counts
    n_correct = sum(1 for c in categories if c == 'correct')
    n_halluc = sum(1 for c in categories if c == 'hallucination')
    n_misconc = sum(1 for c in categories if c == 'deep_misconception')
    print('  Categories: %d correct, %d hallucination, %d deep misconception'
          % (n_correct, n_halluc, n_misconc))


# =============================================================================
# 9. Main
# =============================================================================
def main():
    t0 = time.time()

    # Phase A: Layer-wise Anatomy + Differential Anatomy
    anatomy_results = run_layer_anatomy()

    # Phase B: 5-Strategy Comparison
    comparison_results, question_data = run_comparison(anatomy_results)

    # Phase C: Truth Quadrant
    run_truth_quadrant(question_data)

    elapsed = time.time() - t0

    # === Save JSON Summary ===
    summary = dict(
        model=MODEL_NAME,
        version='v3',
        device=device,
        vram_gb=torch.cuda.memory_allocated() / 1e9,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        mid_layer_range='L%d-L%d' % (MID_LAYER_START, MID_LAYER_END - 1),
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

    json_path = os.path.join(RESULTS_DIR, 'metacognition_v3_results.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print('\n  Saved: %s' % json_path)

    # === Final Banner ===
    print('\n' + '=' * 70)
    print('  Project Metacognition v3 complete!')
    print('  "Strike precisely where the lie is born."')
    print('  Elapsed: %.0f seconds (%.1f min)' % (elapsed, elapsed / 60))
    print('=' * 70)

    print('\n  Strategy Comparison:')
    print('  %-28s  Acc    Intv' % '')
    print('  ' + '-' * 55)
    for name, data in comparison_results.items():
        acc = data['correct'] / max(data['total'], 1) * 100
        intv = data['interventions'] / max(data['total'], 1) * 100
        marker = ' <-- TARGET' if 'v3' in name else ''
        print('  %-28s  %5.1f%%  %5.1f%%%s' % (name, acc, intv, marker))


if __name__ == '__main__':
    main()
