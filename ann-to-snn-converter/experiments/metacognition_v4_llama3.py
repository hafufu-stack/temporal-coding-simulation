"""
Project Metacognition v4 — Phase 3: Cross-Model Validation
=============================================================================
Test the Mid-Layer hypothesis on Llama-3.2-3B to prove universality.
If mid-layers (proportional to model depth) show the same pattern,
it's a universal LLM property, not Mistral-specific.

Llama-3.2-3B: 28 layers → mid-layers ~L8-L16
Mistral-7B:   32 layers → mid-layers L10-L18

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
import torch

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'results_metacognition_v4')
os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================================================================
# 1. Model Loading — Llama-3.2-3B
# =============================================================================
print('\n' + '=' * 70)
print('  Project Metacognition v4 — Phase 3: Cross-Model Validation')
print('  "Is the mid-layer hallucination pattern universal?"')
print('=' * 70)

print('\n[Phase 0] Loading Llama-3.2-3B (fp16)...')
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = 'meta-llama/Llama-3.2-3B'
device = 'cuda'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map='auto',
    attn_implementation='eager',
)
model.eval()
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

NUM_LAYERS = model.config.num_hidden_layers  # 28 for Llama-3.2-3B
NUM_HEADS = model.config.num_attention_heads

# Proportional mid-layer range: ~30-55% of depth
MID_LAYER_START = int(NUM_LAYERS * 0.30)   # ~L8
MID_LAYER_END   = int(NUM_LAYERS * 0.58)   # ~L16
MID_RANGE = range(MID_LAYER_START, MID_LAYER_END)

vram_used = torch.cuda.memory_allocated() / 1e9
vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
print('  Model: %s (%d layers, %d heads)' % (MODEL_NAME, NUM_LAYERS, NUM_HEADS))
print('  VRAM: %.2f / %.2f GB' % (vram_used, vram_total))
print('  Mid-Layer Range: L%d-L%d (%d layers, proportional to Mistral L10-L18)' % (
    MID_LAYER_START, MID_LAYER_END - 1, MID_LAYER_END - MID_LAYER_START))


def make_prompt(question, prefix=''):
    return 'Question: %s\nAnswer: %s' % (question, prefix)


# =============================================================================
# 2. Manual Generation Engine
# =============================================================================
def generate_manual(question, max_new_tokens=50, temperature=0.7,
                    top_k=50, repetition_penalty=1.2,
                    collect_layer_attention=True, prefix=''):
    """Manual token-by-token generation with full layer instrumentation."""
    prompt = make_prompt(question, prefix=prefix)
    input_ids = tokenizer(prompt, return_tensors='pt',
                          truncation=True, max_length=256).input_ids.to(device)
    prompt_len = input_ids.shape[1]

    logits_entropies = []
    attn_entropies = []
    mid_layer_entropies = []
    layer_entropies = []
    top_probs = []
    tokens_generated = []

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

        logits = out.logits[:, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log2(probs + 1e-10)
        logits_ent = -(probs * log_probs).sum(dim=-1).item()
        logits_entropies.append(logits_ent)

        top_p = probs.max(dim=-1).values.item()
        top_probs.append(top_p)

        step_layer_ents = []
        if collect_layer_attention and out.attentions is not None:
            for layer_attn in out.attentions:
                a = layer_attn[0, :, -1, :]
                a_log = torch.log2(a.float() + 1e-10)
                head_ent = -(a.float() * a_log).sum(dim=-1)
                step_layer_ents.append(head_ent.mean().item())
            layer_entropies.append(step_layer_ents)
            attn_entropies.append(np.mean(step_layer_ents))
            mid_ents = step_layer_ents[MID_LAYER_START:MID_LAYER_END]
            mid_layer_entropies.append(np.mean(mid_ents) if mid_ents else 0.0)
        else:
            layer_entropies.append([0.0] * NUM_LAYERS)
            attn_entropies.append(0.0)
            mid_layer_entropies.append(0.0)

        if hasattr(out, 'attentions') and out.attentions is not None:
            del out.attentions

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
        answer=answer,
        question=question,
    )


def check_answer(answer, valid_keywords):
    answer_lower = answer.lower()
    for keyword in valid_keywords:
        if keyword.lower() in answer_lower:
            return True
    return False


# =============================================================================
# 3. Layer Anatomy + Differential Anatomy
# =============================================================================
def run_anatomy():
    """Run anatomy experiment with same questions as Mistral v3."""
    print('\n' + '=' * 70)
    print('  Phase 3A: Layer Anatomy — Llama-3.2-3B')
    print('  "Does a different brain lie in the same place?"')
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

    results = []

    for i, (q, kw, cat) in enumerate(anatomy_questions):
        print('\n  [%d/%d] %s' % (i + 1, len(anatomy_questions), q[:55]))
        answer, traj = generate_manual(q, max_new_tokens=50,
                                       collect_layer_attention=True)
        is_correct = check_answer(answer, kw)
        traj['label'] = 'correct' if is_correct else 'hallucination'
        traj['category'] = cat
        results.append(traj)
        print('         -> %s | %s' % (traj['label'], answer[:60]))

        if device == 'cuda':
            torch.cuda.empty_cache()

    # === Heatmap ===
    _plot_heatmaps(results)

    # === Differential Anatomy ===
    _plot_differential(results)

    return results


def _plot_heatmaps(results):
    """Layer-wise heatmaps for Llama-3.2-3B."""
    print('\n  Generating layer-wise heatmaps...')

    n_plots = len(results)
    fig, axes = plt.subplots(n_plots, 1, figsize=(16, 4 * n_plots))
    if n_plots == 1:
        axes = [axes]

    fig.suptitle('Layer Anatomy: Llama-3.2-3B (%d layers)\n'
                 'Attention Entropy per Layer per Token' % NUM_LAYERS,
                 fontsize=16, fontweight='bold', y=0.995)

    for idx, traj in enumerate(results):
        ax = axes[idx]
        le = traj['layer_entropies']
        n_tokens = len(le)
        if n_tokens == 0:
            continue

        matrix = np.array(le).T

        im = ax.imshow(matrix, aspect='auto', interpolation='nearest',
                       cmap='inferno', origin='lower')
        ax.set_ylabel('Layer')
        ax.set_xlabel('Token Position')
        label_icon = 'O' if traj['label'] == 'correct' else 'X'
        title = '[%s] %s: "%s"' % (label_icon, traj['category'],
                                    traj['question'][:50])
        ax.set_title(title, fontsize=11, fontweight='bold', loc='left')

        # Mid-layer band
        ax.axhline(MID_LAYER_START - 0.5, color='lime', linewidth=1.5,
                   linestyle='--', alpha=0.7)
        ax.axhline(MID_LAYER_END - 0.5, color='lime', linewidth=1.5,
                   linestyle='--', alpha=0.7)

        tokens = traj['tokens'][:n_tokens]
        tick_step = max(1, n_tokens // 15)
        tick_pos = list(range(0, n_tokens, tick_step))
        tick_labels = [tokens[i][:6] if i < len(tokens) else '' for i in tick_pos]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, fontsize=7, rotation=45)

        cb = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.01)
        cb.set_label('Attn Entropy (bits)', fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(RESULTS_DIR, 'llama3_layer_anatomy.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: %s' % out_path)


def _plot_differential(results):
    """Differential Anatomy for Llama-3.2-3B."""
    print('\n  Generating Llama-3.2-3B differential anatomy...')

    correct_trajs = [t for t in results if t['label'] == 'correct']
    halluc_trajs = [t for t in results if t['label'] == 'hallucination']

    if not correct_trajs or not halluc_trajs:
        print('  [SKIP] Need both correct and hallucination samples')
        print('  Correct: %d, Hallucination: %d' % (
            len(correct_trajs), len(halluc_trajs)))
        return

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
    diff = halluc_means - correct_means

    # === Three-panel plot ===
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle('Differential Anatomy: Llama-3.2-3B (%d layers)\n'
                 'Correct vs Hallucination — Where Does the Lie Begin?' % NUM_LAYERS,
                 fontsize=15, fontweight='bold')

    # Panel 1: Correct
    ax1 = axes[0]
    ax1.bar(range(NUM_LAYERS), correct_means, color='#4A90D9', alpha=0.85)
    ax1.axvspan(MID_LAYER_START - 0.5, MID_LAYER_END - 0.5,
                alpha=0.1, color='red', label='Halluc Zone (L%d-L%d)' % (
                    MID_LAYER_START, MID_LAYER_END - 1))
    ax1.set_title('Correct (N=%d)' % len(correct_trajs), fontweight='bold',
                  color='#4A90D9')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Mean Attention Entropy (bits)')
    ax1.legend(fontsize=8)

    # Panel 2: Hallucination
    ax2 = axes[1]
    ax2.bar(range(NUM_LAYERS), halluc_means, color='#E74C3C', alpha=0.85)
    ax2.axvspan(MID_LAYER_START - 0.5, MID_LAYER_END - 0.5,
                alpha=0.1, color='red', label='Halluc Zone (L%d-L%d)' % (
                    MID_LAYER_START, MID_LAYER_END - 1))
    ax2.set_title('Hallucination (N=%d)' % len(halluc_trajs), fontweight='bold',
                  color='#E74C3C')
    ax2.set_xlabel('Layer')
    ax2.legend(fontsize=8)

    # Panel 3: DIFFERENCE
    ax3 = axes[2]
    colors_diff = ['#E74C3C' if d > 0 else '#4A90D9' for d in diff]
    ax3.bar(range(NUM_LAYERS), diff, color=colors_diff, alpha=0.85)
    ax3.axvspan(MID_LAYER_START - 0.5, MID_LAYER_END - 0.5,
                alpha=0.15, color='red', label='Halluc Zone (L%d-L%d)' % (
                    MID_LAYER_START, MID_LAYER_END - 1))
    ax3.axhline(0, color='black', linewidth=0.8)
    ax3.set_title('DIFFERENCE (Halluc - Correct)', fontweight='bold',
                  color='#333333')
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('Entropy Difference (bits)')
    ax3.legend(fontsize=8)

    # Annotate peak in mid-layer range
    mid_diff = diff[MID_LAYER_START:MID_LAYER_END]
    if len(mid_diff) > 0:
        peak_idx = np.argmax(np.abs(mid_diff))
        peak_layer = MID_LAYER_START + peak_idx
        peak_val = diff[peak_layer]
        ax3.annotate('Peak: L%d\n%+.3f bits' % (peak_layer, peak_val),
                     xy=(peak_layer, peak_val),
                     xytext=(peak_layer + 3, peak_val + 0.03),
                     fontsize=9, fontweight='bold', color='#E74C3C',
                     arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=1.5))

    ymax = max(correct_means.max(), halluc_means.max()) * 1.1
    axes[0].set_ylim(0, ymax)
    axes[1].set_ylim(0, ymax)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, 'llama3_differential_anatomy.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: %s' % out_path)

    # === Cross-Model Comparison (load Mistral v3 data if available) ===
    _plot_cross_model_comparison(correct_means, halluc_means, diff)


def _plot_cross_model_comparison(llama_correct, llama_halluc, llama_diff):
    """Side-by-side: Mistral-7B vs Llama-3.2-3B differential anatomy."""
    print('\n  Generating cross-model comparison...')

    # Load Mistral v3 results if available
    mistral_json = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'results_metacognition_v3',
                                'metacognition_v3_results.json')

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor('white')
    fig.suptitle('Cross-Model Universality: Where Do LLMs Lie?\n'
                 'Differential Anatomy (Hallucination - Correct entropy per layer)',
                 fontsize=15, fontweight='bold')

    # Panel 1: Llama-3.2-3B
    ax1 = axes[0]
    llama_colors = ['#E74C3C' if d > 0 else '#4A90D9' for d in llama_diff]
    ax1.bar(range(len(llama_diff)), llama_diff, color=llama_colors, alpha=0.85)
    ax1.axvspan(MID_LAYER_START - 0.5, MID_LAYER_END - 0.5,
                alpha=0.15, color='red', label='Halluc Zone')
    ax1.axhline(0, color='black', linewidth=0.8)
    ax1.set_title('Llama-3.2-3B (28 layers)', fontweight='bold',
                  fontsize=13, color='#8E44AD')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Entropy Difference (bits)')
    ax1.legend(fontsize=9)

    # Panel 2: Normalized overlay comparison
    ax2 = axes[1]

    # Normalize both to [0, 1] range on x-axis for comparison
    llama_x = np.linspace(0, 1, len(llama_diff))
    ax2.plot(llama_x, llama_diff, 'o-', color='#8E44AD', linewidth=2,
             markersize=4, label='Llama-3.2-3B (%d layers)' % len(llama_diff),
             alpha=0.8)

    # Mark the mid-layer zones
    llama_mid_start = MID_LAYER_START / len(llama_diff)
    llama_mid_end = MID_LAYER_END / len(llama_diff)
    ax2.axvspan(llama_mid_start, llama_mid_end,
                alpha=0.1, color='purple', label='Llama Mid-Zone')

    # Mistral zone (proportional)
    mistral_mid_start = 10 / 32
    mistral_mid_end = 19 / 32
    ax2.axvspan(mistral_mid_start, mistral_mid_end,
                alpha=0.1, color='green', label='Mistral Mid-Zone (from v3)')

    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_title('Normalized Layer Position Comparison', fontweight='bold',
                  fontsize=13)
    ax2.set_xlabel('Normalized Layer Position (0=shallow, 1=deep)')
    ax2.set_ylabel('Entropy Difference (bits)')
    ax2.legend(fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, 'cross_model_comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: %s' % out_path)


# =============================================================================
# 4. Main
# =============================================================================
if __name__ == '__main__':
    t0 = time.time()

    results = run_anatomy()

    elapsed = time.time() - t0

    print('\n' + '=' * 70)
    print('  Phase 3 Complete — Llama-3.2-3B Cross-Model Validation')
    print('  Elapsed: %.0f seconds (%.1f min)' % (elapsed, elapsed / 60))
    print('=' * 70)

    # Summary
    correct = sum(1 for r in results if r['label'] == 'correct')
    halluc = sum(1 for r in results if r['label'] == 'hallucination')
    print('\n  Llama-3.2-3B: %d correct, %d hallucination (out of %d)' % (
        correct, halluc, len(results)))
    print('  Mid-layer range: L%d-L%d (%d layers)' % (
        MID_LAYER_START, MID_LAYER_END - 1, MID_LAYER_END - MID_LAYER_START))

    # Save results
    json_data = {
        'model': MODEL_NAME,
        'num_layers': NUM_LAYERS,
        'mid_layer_range': [MID_LAYER_START, MID_LAYER_END - 1],
        'results_count': {'correct': correct, 'hallucination': halluc},
        'elapsed_seconds': elapsed,
    }
    json_path = os.path.join(RESULTS_DIR, 'llama3_anatomy.json')
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print('  Saved: %s' % json_path)
