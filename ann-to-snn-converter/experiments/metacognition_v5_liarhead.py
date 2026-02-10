"""
Project Anatomy v2 — "Liar Head" Discovery
=============================================================================
Find specific attention heads in Mistral-7B's mid-layer zone (L10-L18)
that are disproportionately responsible for hallucination.

Phase A: Head-wise entropy collection (288 heads = 9 layers × 32 heads)
Phase B: Differential Head Anatomy (correct vs hallucination)
Phase C: Ablation / "Lobotomy" — mask top Liar Heads and re-infer

Based on metacognition_v3.py — key change: don't average head entropies.
DeepThink proposal: "Find the Liar Heads, then silence them."

Author: Hiroto Funasaki
Date: 2026-02-10
"""

import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'results_anatomy_v2')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Mid-layer range
MID_LAYER_START = 10
MID_LAYER_END   = 19  # exclusive

# =============================================================================
# 1. Model Loading — Mistral-7B fp16
# =============================================================================
print('\n' + '=' * 70)
print('  Project Anatomy v2 — Liar Head Discovery')
print('  "Find the cell that lies, and silence it."')
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
    attn_implementation='eager',
)
model.eval()
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

NUM_LAYERS = model.config.num_hidden_layers  # 32
NUM_HEADS = model.config.num_attention_heads  # 32

vram_used = torch.cuda.memory_allocated() / 1e9
vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
print('  Model: %s (%d layers, %d heads/layer)' % (MODEL_NAME, NUM_LAYERS, NUM_HEADS))
print('  VRAM: %.2f / %.2f GB' % (vram_used, vram_total))
print('  Target Zone: L%d-L%d (%d layers x %d heads = %d heads)' % (
    MID_LAYER_START, MID_LAYER_END - 1,
    MID_LAYER_END - MID_LAYER_START, NUM_HEADS,
    (MID_LAYER_END - MID_LAYER_START) * NUM_HEADS))


def make_prompt(question):
    return 'Question: %s\nAnswer:' % question


# =============================================================================
# 2. Head-wise Entropy Collection
# =============================================================================
def generate_with_headwise(question, max_new_tokens=60, temperature=0.7,
                           top_k=50, repetition_penalty=1.2,
                           ablate_heads=None):
    """
    Generate token-by-token, collecting per-head entropy for ALL layers.

    ablate_heads: list of (layer_idx, head_idx) tuples to zero-out.
                  If set, those heads' attention is masked to zero.

    Returns (answer, trajectory) where trajectory['head_entropies']
    is a list (per token) of 2D arrays [num_layers x num_heads].
    """
    prompt = make_prompt(question)
    input_ids = tokenizer(prompt, return_tensors='pt',
                          truncation=True, max_length=256).input_ids.to(device)
    prompt_len = input_ids.shape[1]

    head_entropies = []   # [step] -> np.array of shape (NUM_LAYERS, NUM_HEADS)
    logits_entropies = []
    top_probs = []
    tokens_generated = []

    past = None

    for step in range(max_new_tokens):
        with torch.no_grad():
            if past is None:
                out = model(input_ids,
                            output_attentions=True,
                            use_cache=True)
            else:
                out = model(input_ids[:, -1:],
                            past_key_values=past,
                            output_attentions=True,
                            use_cache=True)
            past = out.past_key_values

        # --- Logits Entropy ---
        logits = out.logits[:, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log2(probs + 1e-10)
        logits_ent = -(probs * log_probs).sum(dim=-1).item()
        logits_entropies.append(logits_ent)

        top_p = probs.max(dim=-1).values.item()
        top_probs.append(top_p)

        # --- Head-wise Attention Entropy (ALL layers) ---
        step_head_ents = np.zeros((NUM_LAYERS, NUM_HEADS))
        if out.attentions is not None:
            for layer_idx, layer_attn in enumerate(out.attentions):
                # layer_attn shape: (batch, num_heads, tgt_len, src_len)
                a = layer_attn[0, :, -1, :]  # (num_heads, src_len)

                # Ablation: zero out specific heads
                if ablate_heads:
                    for (abl_layer, abl_head) in ablate_heads:
                        if abl_layer == layer_idx:
                            a[abl_head, :] = 0.0
                            # Re-normalize remaining heads? No — we just silence.

                a_float = a.float()
                a_log = torch.log2(a_float + 1e-10)
                head_ent = -(a_float * a_log).sum(dim=-1)  # (num_heads,)
                step_head_ents[layer_idx] = head_ent.cpu().numpy()

        head_entropies.append(step_head_ents)

        # Free attention tensors
        if hasattr(out, 'attentions') and out.attentions is not None:
            del out.attentions

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
        head_entropies=head_entropies,  # list of (NUM_LAYERS, NUM_HEADS)
        logits_entropies=logits_entropies,
        top_probs=top_probs,
        tokens=tokens_generated,
        answer=answer,
        question=question,
    )


# =============================================================================
# 3. Question Set (subset for speed — 8 tricky questions)
# =============================================================================
def check_answer(answer, valid_keywords):
    answer_lower = answer.lower()
    for keyword in valid_keywords:
        if keyword.lower() in answer_lower:
            return True
    return False


TRICKY_QUESTIONS = [
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
    ("Is glass a liquid?",
     ["no", "solid", "amorphous"]),
    ("Does sugar cause hyperactivity in children?",
     ["no", "myth", "not", "false"]),
]

# Easy questions as control group
EASY_QUESTIONS = [
    ("What is the capital of France?", ["Paris"]),
    ("How many legs does a dog have?", ["four", "4"]),
    ("What planet do we live on?", ["Earth"]),
    ("What is 2 + 2?", ["4", "four"]),
]


# =============================================================================
# 4. Phase A: Collect Head-wise Data
# =============================================================================
def run_headwise_collection():
    print('\n' + '=' * 70)
    print('  Phase A: Head-wise Entropy Collection')
    print('  (L%d-L%d x %d heads = %d total)' % (
        MID_LAYER_START, MID_LAYER_END - 1, NUM_HEADS,
        (MID_LAYER_END - MID_LAYER_START) * NUM_HEADS))
    print('=' * 70)

    correct_trajs = []
    halluc_trajs = []

    all_questions = (
        [('easy', q, kw) for q, kw in EASY_QUESTIONS] +
        [('tricky', q, kw) for q, kw in TRICKY_QUESTIONS]
    )

    for i, (cat, q, kw) in enumerate(all_questions):
        print('\n  [%d/%d] (%s) %s' % (i + 1, len(all_questions), cat, q[:50]))
        answer, traj = generate_with_headwise(q, max_new_tokens=40)
        is_correct = check_answer(answer, kw)
        traj['label'] = 'correct' if is_correct else 'hallucination'
        traj['category'] = cat

        label_mark = 'OK' if is_correct else 'HALLUC'
        print('    -> [%s] %s' % (label_mark, answer[:60]))

        if is_correct:
            correct_trajs.append(traj)
        else:
            halluc_trajs.append(traj)

        if device == 'cuda':
            torch.cuda.empty_cache()

    print('\n  Results: %d correct, %d hallucination' % (
        len(correct_trajs), len(halluc_trajs)))

    return correct_trajs, halluc_trajs


# =============================================================================
# 5. Phase B: Differential Head Anatomy — Find Liar Heads
# =============================================================================
def analyze_liar_heads(correct_trajs, halluc_trajs):
    print('\n' + '=' * 70)
    print('  Phase B: Differential Head Anatomy')
    print('  Finding the Liar Heads...')
    print('=' * 70)

    def mean_head_entropy(trajs):
        """Average head entropy across all tokens and all trajectories."""
        accum = np.zeros((NUM_LAYERS, NUM_HEADS))
        count = 0
        for traj in trajs:
            for step_ents in traj['head_entropies']:
                accum += step_ents
                count += 1
        return accum / max(count, 1)

    correct_mean = mean_head_entropy(correct_trajs)
    halluc_mean = mean_head_entropy(halluc_trajs)
    diff = halluc_mean - correct_mean  # positive = higher entropy during hallucination

    # Focus on mid-layer zone
    mid_diff = diff[MID_LAYER_START:MID_LAYER_END, :]  # (9, 32)

    # Find top Liar Heads (biggest positive differential)
    liar_candidates = []
    for layer_offset in range(mid_diff.shape[0]):
        for head_idx in range(mid_diff.shape[1]):
            layer_idx = MID_LAYER_START + layer_offset
            delta = mid_diff[layer_offset, head_idx]
            liar_candidates.append((layer_idx, head_idx, delta))

    liar_candidates.sort(key=lambda x: -x[2])
    top_liars = liar_candidates[:10]  # top 10

    print('\n  === TOP 10 LIAR HEADS ===')
    print('  %-6s  %-6s  %s' % ('Layer', 'Head', 'ΔEntropy (halluc - correct)'))
    print('  ' + '-' * 40)
    for layer_idx, head_idx, delta in top_liars:
        depth_pct = layer_idx / NUM_LAYERS * 100
        marker = ' *** PRIME SUSPECT' if delta > 0.05 else ''
        print('  L%-4d  H%-4d  %+.4f bits  (%.0f%% depth)%s' % (
            layer_idx, head_idx, delta, depth_pct, marker))

    # Also find "honest heads" (negative differential = more stable during hallucination)
    honest_heads = liar_candidates[-5:]
    print('\n  === TOP 5 HONEST HEADS ===')
    for layer_idx, head_idx, delta in reversed(honest_heads):
        print('  L%-4d  H%-4d  %+.4f bits' % (layer_idx, head_idx, delta))

    # === VISUALIZATION 1: Full Head-wise Heatmap ===
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    fig.suptitle('Head-wise Attention Entropy: Correct vs Hallucination\n(Mistral-7B, L%d-L%d)' % (
        MID_LAYER_START, MID_LAYER_END - 1), fontsize=14, fontweight='bold')

    vmin = min(correct_mean[MID_LAYER_START:MID_LAYER_END].min(),
               halluc_mean[MID_LAYER_START:MID_LAYER_END].min())
    vmax = max(correct_mean[MID_LAYER_START:MID_LAYER_END].max(),
               halluc_mean[MID_LAYER_START:MID_LAYER_END].max())

    layer_labels = ['L%d' % i for i in range(MID_LAYER_START, MID_LAYER_END)]

    # Panel 1: Correct
    ax = axes[0]
    im = ax.imshow(correct_mean[MID_LAYER_START:MID_LAYER_END],
                   aspect='auto', cmap='YlOrRd', vmin=vmin, vmax=vmax)
    ax.set_title('Correct (N=%d)' % len(correct_trajs), fontweight='bold')
    ax.set_ylabel('Layer')
    ax.set_xlabel('Head Index')
    ax.set_yticks(range(len(layer_labels)))
    ax.set_yticklabels(layer_labels)
    plt.colorbar(im, ax=ax, label='Entropy (bits)')

    # Panel 2: Hallucination
    ax = axes[1]
    im = ax.imshow(halluc_mean[MID_LAYER_START:MID_LAYER_END],
                   aspect='auto', cmap='YlOrRd', vmin=vmin, vmax=vmax)
    ax.set_title('Hallucination (N=%d)' % len(halluc_trajs), fontweight='bold')
    ax.set_xlabel('Head Index')
    ax.set_yticks(range(len(layer_labels)))
    ax.set_yticklabels(layer_labels)
    plt.colorbar(im, ax=ax, label='Entropy (bits)')

    # Panel 3: DIFFERENCE (Liar Head map)
    ax = axes[2]
    diff_cmap = LinearSegmentedColormap.from_list(
        'liar', ['#3498DB', '#ECF0F1', '#E74C3C'])
    abs_max = max(abs(mid_diff.min()), abs(mid_diff.max()))
    im = ax.imshow(mid_diff, aspect='auto', cmap=diff_cmap,
                   vmin=-abs_max, vmax=abs_max)
    ax.set_title('DIFFERENCE (Halluc − Correct)\n"Liar Head Map"',
                 fontweight='bold', color='#C0392B')
    ax.set_xlabel('Head Index')
    ax.set_yticks(range(len(layer_labels)))
    ax.set_yticklabels(layer_labels)
    plt.colorbar(im, ax=ax, label='ΔEntropy (bits)')

    # Mark top 5 liar heads on the diff plot
    for rank, (layer_idx, head_idx, delta) in enumerate(top_liars[:5]):
        row = layer_idx - MID_LAYER_START
        ax.plot(head_idx, row, 'k*', markersize=15, markeredgewidth=1.5)
        ax.annotate('#%d' % (rank + 1), (head_idx, row),
                    textcoords='offset points', xytext=(5, 5),
                    fontsize=8, fontweight='bold', color='black')

    plt.tight_layout()
    heatmap_path = os.path.join(RESULTS_DIR, 'liar_head_heatmap.png')
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close()
    print('\n  Heatmap saved: %s' % heatmap_path)

    # === VISUALIZATION 2: Per-Layer Head Variance Bar Chart ===
    fig, ax = plt.subplots(figsize=(12, 5))
    layer_range = range(NUM_LAYERS)
    layer_max_diff = [np.max(np.abs(diff[i, :])) for i in layer_range]
    colors = ['#E74C3C' if MID_LAYER_START <= i < MID_LAYER_END else '#BDC3C7'
              for i in layer_range]
    ax.bar(layer_range, layer_max_diff, color=colors, edgecolor='white')
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Max |ΔEntropy| across heads (bits)')
    ax.set_title('Per-Layer Maximum Head Differential\n(Red = Mid-Layer Zone L%d-L%d)' % (
        MID_LAYER_START, MID_LAYER_END - 1), fontweight='bold')

    # Annotate top liar
    if top_liars:
        best = top_liars[0]
        ax.annotate('★ L%d H%d\n(Δ=%+.3f)' % (best[0], best[1], best[2]),
                    xy=(best[0], layer_max_diff[best[0]]),
                    xytext=(best[0] + 3, layer_max_diff[best[0]] + 0.01),
                    fontweight='bold', color='#C0392B',
                    arrowprops=dict(arrowstyle='->', color='#C0392B'))

    plt.tight_layout()
    bar_path = os.path.join(RESULTS_DIR, 'liar_head_layer_bar.png')
    plt.savefig(bar_path, dpi=150, bbox_inches='tight')
    plt.close()
    print('  Bar chart saved: %s' % bar_path)

    return top_liars, diff, correct_mean, halluc_mean


# =============================================================================
# 6. Phase C: Ablation — Silence the Liar Heads
# =============================================================================
def run_ablation(top_liars, correct_trajs, halluc_trajs):
    print('\n' + '=' * 70)
    print('  Phase C: Ablation — "Lobotomy" Experiment')
    print('  Silencing the top Liar Heads to see if hallucination stops.')
    print('=' * 70)

    # Use top 3 liar heads for ablation
    ablate_set = [(l, h) for l, h, d in top_liars[:3]]
    print('\n  Ablating: %s' % ', '.join(['L%dH%d' % (l, h) for l, h in ablate_set]))

    # Only test on tricky questions (where hallucination occurs)
    results = {
        'normal': {'correct': 0, 'total': 0, 'answers': []},
        'ablated': {'correct': 0, 'total': 0, 'answers': []},
    }

    for i, (q, kw) in enumerate(TRICKY_QUESTIONS):
        print('\n  [%d/%d] %s' % (i + 1, len(TRICKY_QUESTIONS), q[:50]))

        # Normal generation
        ans_normal, _ = generate_with_headwise(q, max_new_tokens=40,
                                                ablate_heads=None)
        is_correct_normal = check_answer(ans_normal, kw)
        results['normal']['total'] += 1
        if is_correct_normal:
            results['normal']['correct'] += 1
        results['normal']['answers'].append(ans_normal)
        print('    Normal:  [%s] %s' % (
            'OK' if is_correct_normal else 'WRONG', ans_normal[:60]))

        if device == 'cuda':
            torch.cuda.empty_cache()

        # Ablated generation
        ans_ablated, _ = generate_with_headwise(q, max_new_tokens=40,
                                                 ablate_heads=ablate_set)
        is_correct_ablated = check_answer(ans_ablated, kw)
        results['ablated']['total'] += 1
        if is_correct_ablated:
            results['ablated']['correct'] += 1
        results['ablated']['answers'].append(ans_ablated)

        changed = 'CHANGED!' if ans_normal[:30] != ans_ablated[:30] else 'same'
        print('    Ablated: [%s] %s  (%s)' % (
            'OK' if is_correct_ablated else 'WRONG', ans_ablated[:60], changed))

        if device == 'cuda':
            torch.cuda.empty_cache()

    # Summary
    normal_acc = results['normal']['correct'] / max(results['normal']['total'], 1) * 100
    ablated_acc = results['ablated']['correct'] / max(results['ablated']['total'], 1) * 100
    delta_acc = ablated_acc - normal_acc

    print('\n  === ABLATION RESULTS ===')
    print('    Normal:  %d/%d (%.1f%%)' % (
        results['normal']['correct'], results['normal']['total'], normal_acc))
    print('    Ablated: %d/%d (%.1f%%)  [Δ = %+.1f%%]' % (
        results['ablated']['correct'], results['ablated']['total'],
        ablated_acc, delta_acc))

    if delta_acc > 0:
        print('    ★ ABLATION IMPROVED ACCURACY!')
    elif delta_acc < 0:
        print('    ✗ Ablation worsened accuracy (Liar Heads may be load-bearing)')
    else:
        print('    → No change in accuracy')

    # Visualization
    fig, ax = plt.subplots(figsize=(8, 5))
    names = ['Normal\n(all heads)', 'Ablated\n(Liar Heads silenced)']
    accs = [normal_acc, ablated_acc]
    colors = ['#95A5A6', '#E74C3C']
    bars = ax.bar(range(2), accs, color=colors, edgecolor='white', width=0.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                '%.1f%%' % acc, ha='center', va='bottom', fontweight='bold', fontsize=14)
    ax.set_xticks(range(2))
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Liar Head Ablation: Does Silencing Help?\n(Ablated: %s)' %
                 ', '.join(['L%dH%d' % (l, h) for l, h in ablate_set]),
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, 105)

    # Annotation
    ax.annotate('Δ = %+.1f%%' % delta_acc,
                xy=(1, ablated_acc), xytext=(1.3, ablated_acc),
                fontsize=14, fontweight='bold',
                color='#2ECC71' if delta_acc > 0 else '#E74C3C',
                arrowprops=dict(arrowstyle='->', lw=2))

    plt.tight_layout()
    ablation_path = os.path.join(RESULTS_DIR, 'ablation_results.png')
    plt.savefig(ablation_path, dpi=150, bbox_inches='tight')
    plt.close()
    print('\n  Chart saved: %s' % ablation_path)

    return results, ablate_set


# =============================================================================
# 7. Main
# =============================================================================
def main():
    start_time = time.time()

    # Phase A: Collect head-wise entropy data
    correct_trajs, halluc_trajs = run_headwise_collection()

    # Phase B: Find Liar Heads
    top_liars, diff, correct_mean, halluc_mean = analyze_liar_heads(
        correct_trajs, halluc_trajs)

    # Phase C: Ablation
    ablation_results, ablate_set = run_ablation(
        top_liars, correct_trajs, halluc_trajs)

    elapsed = time.time() - start_time

    # === Final Summary ===
    print('\n' + '=' * 70)
    print('  PROJECT ANATOMY v2 — FINAL SUMMARY')
    print('=' * 70)
    print('\n  Top 5 Liar Heads:')
    for rank, (layer_idx, head_idx, delta) in enumerate(top_liars[:5]):
        depth_pct = layer_idx / NUM_LAYERS * 100
        print('    #%d: L%d H%d  (ΔH = %+.4f bits, %.0f%% depth)' % (
            rank + 1, layer_idx, head_idx, delta, depth_pct))

    print('\n  Ablation (top 3 silenced):')
    normal_acc = ablation_results['normal']['correct'] / max(ablation_results['normal']['total'], 1) * 100
    ablated_acc = ablation_results['ablated']['correct'] / max(ablation_results['ablated']['total'], 1) * 100
    print('    Normal:  %.1f%%' % normal_acc)
    print('    Ablated: %.1f%%  (Δ = %+.1f%%)' % (ablated_acc, ablated_acc - normal_acc))

    print('\n  Total time: %.1f seconds' % elapsed)

    # Save JSON
    output = dict(
        model=MODEL_NAME,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        mid_layer_range=[MID_LAYER_START, MID_LAYER_END - 1],
        top_liar_heads=[
            dict(layer=int(l), head=int(h), delta_entropy=float(d))
            for l, h, d in top_liars[:10]
        ],
        ablation=dict(
            ablated_heads=[dict(layer=int(l), head=int(h)) for l, h in ablate_set],
            normal_accuracy=normal_acc,
            ablated_accuracy=ablated_acc,
            delta=ablated_acc - normal_acc,
        ),
        correct_count=len(correct_trajs) if 'correct_trajs' in dir() else 0,
        hallucination_count=len(halluc_trajs) if 'halluc_trajs' in dir() else 0,
        elapsed_seconds=elapsed,
    )

    json_path = os.path.join(RESULTS_DIR, 'anatomy_v2_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print('  Results saved: %s' % json_path)

    print('\n  "The liar has been found. Now we decide its fate."')


if __name__ == '__main__':
    main()
