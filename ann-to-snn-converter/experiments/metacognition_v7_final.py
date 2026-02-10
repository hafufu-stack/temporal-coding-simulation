"""
Project Anatomy v4 — FIXED VERSION
=============================================================================
Fixes:
1. Qwen GQA NaN: filter NaN heads with torch.where before entropy calc
2. Phi-2 pad_token_id: manually set config attribute before model init
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
                           'results_anatomy_v4')
os.makedirs(RESULTS_DIR, exist_ok=True)

QUESTIONS_CORRECT = [
    "What is the capital of France?",
    "What is 2 + 2?",
    "Who wrote Romeo and Juliet?",
    "What is the speed of light in km/s?",
]
QUESTIONS_HALLUC = [
    "Who was the first person to walk on Mars?",
    "What year was the internet invented?",
    "What percentage of the brain do humans use?",
    "What is the largest desert on Earth?",
]


def compute_layer_entropy(model, tokenizer, questions, device, model_label):
    """Compute per-layer attention entropy with NaN-safe handling."""
    num_layers = model.config.num_hidden_layers
    layer_entropies = []

    for q in questions:
        prompt = 'Question: %s\nAnswer:' % q
        inputs = tokenizer(prompt, return_tensors='pt',
                           truncation=True, max_length=256)
        input_ids = inputs['input_ids'].to(device)

        with torch.no_grad():
            out = model(input_ids, output_attentions=True, use_cache=False)

        step_ents = []
        for l_idx, attn in enumerate(out.attentions):
            a = attn[0, :, -1, :].float()  # (heads, src_len)

            # NaN-safe entropy: replace NaN with 0, then filter
            a = torch.where(torch.isnan(a), torch.zeros_like(a), a)
            a = a.clamp(min=1e-10)
            a_log = torch.log2(a)
            head_ent = -(a * a_log).sum(dim=-1)  # (heads,)

            # Filter out NaN/Inf entropies (from degenerate heads)
            valid = ~(torch.isnan(head_ent) | torch.isinf(head_ent))
            if valid.any():
                mean_ent = head_ent[valid].mean().item()
            else:
                mean_ent = 0.0

            step_ents.append(mean_ent)

        layer_entropies.append(step_ents)

        del out
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()

    return np.array(layer_entropies)  # (N_questions, N_layers)


def analyze_result(model, correct_ents, halluc_ents, model_label, device_label):
    """Compute peak, zone, etc from entropy arrays."""
    correct_mean = correct_ents.mean(axis=0)
    halluc_mean = halluc_ents.mean(axis=0)
    diff = halluc_mean - correct_mean

    # Handle all-NaN case
    if np.all(np.isnan(diff)):
        print('  WARNING: All diff values are NaN for %s' % model_label)
        return None

    # Replace NaN in diff with 0
    diff = np.nan_to_num(diff, nan=0.0)

    peak_layer = int(np.argmax(diff))
    num_layers = model.config.num_hidden_layers
    peak_depth = (peak_layer + 1) / num_layers * 100

    threshold = np.max(diff) * 0.5
    zone_layers = np.where(diff > threshold)[0]
    zone_start = int(zone_layers[0]) if len(zone_layers) > 0 else peak_layer
    zone_end = int(zone_layers[-1]) if len(zone_layers) > 0 else peak_layer

    kv = getattr(model.config, 'num_key_value_heads',
                 model.config.num_attention_heads)

    result = {
        'model': model_label,
        'device': device_label,
        'num_layers': num_layers,
        'num_heads': model.config.num_attention_heads,
        'kv_heads': kv,
        'correct_mean': [round(x, 6) for x in correct_mean.tolist()],
        'halluc_mean': [round(x, 6) for x in halluc_mean.tolist()],
        'diff': [round(x, 6) for x in diff.tolist()],
        'peak_layer': peak_layer,
        'peak_depth_pct': round(peak_depth, 1),
        'peak_delta': round(float(np.max(diff)), 6),
        'mid_zone': [zone_start, zone_end],
        'mid_zone_depth': [
            round((zone_start + 1) / num_layers * 100, 1),
            round((zone_end + 1) / num_layers * 100, 1),
        ],
    }

    print('\n  ★ Peak Layer: L%d (%.0f%% depth)' % (peak_layer, peak_depth))
    print('  ★ Zone: L%d-L%d (%.0f-%.0f%%)' % (
        zone_start, zone_end,
        result['mid_zone_depth'][0], result['mid_zone_depth'][1]))
    print('  ★ Peak ΔH: %.4f bits' % np.max(diff))

    return result


# =============================================================================
# EXP 1: Qwen2.5-1.5B GPU
# =============================================================================
def run_qwen_gpu():
    print('\n' + '=' * 70)
    print('  EXP 1: Qwen2.5-1.5B GPU (eager, NaN-safe)')
    print('=' * 70)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            'Qwen/Qwen2.5-1.5B', trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            'Qwen/Qwen2.5-1.5B',
            torch_dtype=torch.float16, device_map='auto',
            trust_remote_code=True, attn_implementation='eager')
        model.eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        device = next(model.parameters()).device
        print('  Loaded on: %s, Layers: %d, Heads: %d (KV: %d)' % (
            device, model.config.num_hidden_layers,
            model.config.num_attention_heads,
            model.config.num_key_value_heads))

        correct_ents = compute_layer_entropy(
            model, tokenizer, QUESTIONS_CORRECT, device, 'Qwen2.5-GPU')
        halluc_ents = compute_layer_entropy(
            model, tokenizer, QUESTIONS_HALLUC, device, 'Qwen2.5-GPU')

        result = analyze_result(model, correct_ents, halluc_ents,
                                'Qwen2.5-1.5B (GPU)', 'GPU (eager)')

        del model, tokenizer
        gc.collect(); torch.cuda.empty_cache()
        return result

    except Exception as e:
        print('  FAILED: %s' % e)
        import traceback; traceback.print_exc()
        return None


# =============================================================================
# EXP 2: Phi-2 (2.7B)
# =============================================================================
def run_phi2():
    print('\n' + '=' * 70)
    print('  EXP 2: Microsoft Phi-2 (2.7B)')
    print('=' * 70)

    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            'microsoft/phi-2', trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Workaround: load config first and set pad_token_id
        config = AutoConfig.from_pretrained(
            'microsoft/phi-2', trust_remote_code=True)
        if not hasattr(config, 'pad_token_id') or config.pad_token_id is None:
            config.pad_token_id = tokenizer.pad_token_id
            print('  Fixed: set pad_token_id=%d' % config.pad_token_id)

        model = AutoModelForCausalLM.from_pretrained(
            'microsoft/phi-2',
            config=config,
            torch_dtype=torch.float16, device_map='auto',
            trust_remote_code=True, attn_implementation='eager')
        model.eval()

        device = next(model.parameters()).device
        num_layers = model.config.num_hidden_layers
        num_heads = model.config.num_attention_heads
        kv = getattr(model.config, 'num_key_value_heads', num_heads)
        print('  Loaded on: %s, Layers: %d, Heads: %d (KV: %d)' % (
            device, num_layers, num_heads, kv))

        correct_ents = compute_layer_entropy(
            model, tokenizer, QUESTIONS_CORRECT, device, 'Phi-2')
        halluc_ents = compute_layer_entropy(
            model, tokenizer, QUESTIONS_HALLUC, device, 'Phi-2')

        result = analyze_result(model, correct_ents, halluc_ents,
                                'Phi-2 (2.7B)', 'GPU')

        del model, tokenizer
        gc.collect(); torch.cuda.empty_cache()
        return result

    except Exception as e:
        print('  FAILED: %s' % e)
        import traceback; traceback.print_exc()
        return None


# =============================================================================
# UPDATED DEPTH SCALING LAW (5 models)
# =============================================================================
def plot_depth_scaling(qwen_result, phi2_result):
    models = [
        {'name': 'GPT-2\n(124M)', 'params_B': 0.124, 'peak_depth': 17,
         'arch': 'MHA', 'color': '#9B59B6'},
    ]
    if qwen_result:
        models.append({
            'name': 'Qwen2.5\n(1.5B GPU)',
            'params_B': 1.5,
            'peak_depth': qwen_result['peak_depth_pct'],
            'arch': 'GQA', 'color': '#2ECC71'})
    if phi2_result:
        models.append({
            'name': 'Phi-2\n(2.7B)',
            'params_B': 2.7,
            'peak_depth': phi2_result['peak_depth_pct'],
            'arch': 'Dense', 'color': '#F39C12'})
    models.extend([
        {'name': 'Llama-3.2\n(3B)', 'params_B': 3.2, 'peak_depth': 43,
         'arch': 'GQA', 'color': '#3498DB'},
        {'name': 'Mistral\n(7B)', 'params_B': 7.2, 'peak_depth': 44,
         'arch': 'SWA', 'color': '#E74C3C'},
    ])

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('#0D1117')
    ax.set_facecolor('#161B22')

    ax.axhspan(30, 55, alpha=0.15, color='#2ECC71', label='Universal Zone (30-55%)')
    ax.axhline(42.5, color='#2ECC71', alpha=0.3, linestyle='--')

    for m in models:
        marker = 'D' if m['arch'] == 'GQA' else ('s' if m['arch'] == 'Dense' else 'o')
        edge = 'gold' if m['arch'] == 'GQA' else ('orange' if m['arch'] == 'Dense' else 'white')
        ax.scatter(m['params_B'], m['peak_depth'], c=m['color'], s=280, zorder=5,
                   marker=marker, edgecolors=edge, linewidth=2.5)

        # Smart label offset
        offset_x = m['params_B'] * 0.25
        offset_y = 4
        if m['params_B'] < 0.2:
            offset_x = m['params_B'] * 1.5
        elif m['params_B'] > 5:
            offset_x = m['params_B'] * 0.35

        ax.annotate(m['name'],
                    (m['params_B'], m['peak_depth']),
                    xytext=(m['params_B'] + offset_x, m['peak_depth'] + offset_y),
                    fontsize=11, fontweight='bold', color=m['color'],
                    arrowprops=dict(arrowstyle='->', color=m['color'], alpha=0.6))

    # Legend for markers
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='MHA',
               markerfacecolor='#9B59B6', markersize=10, markeredgecolor='white'),
        Line2D([0], [0], marker='D', color='w', label='GQA',
               markerfacecolor='#2ECC71', markersize=10, markeredgecolor='gold'),
        Line2D([0], [0], marker='s', color='w', label='Dense',
               markerfacecolor='#F39C12', markersize=10, markeredgecolor='orange'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10,
              facecolor='#161B22', edgecolor='#30363D', labelcolor='white')

    ax.set_xscale('log')
    ax.set_xlabel('Model Size (Billion Parameters)', fontsize=14, color='white')
    ax.set_ylabel('Peak Hallucination Depth (%)', fontsize=14, color='white')

    n = len(models)
    ax.set_title(f'Depth Scaling Law ({n} Architectures)\n'
                 'Where Do Models Hallucinate?',
                 fontsize=17, fontweight='bold', color='white', pad=15)

    ax.set_xlim(0.08, 15)
    ax.set_ylim(0, 100)
    ax.set_xticks([0.1, 0.5, 1, 2.7, 3, 7])
    ax.set_xticklabels(['100M', '500M', '1B', '2.7B', '3B', '7B'],
                        fontsize=11, color='#BDC3C7')
    ax.tick_params(colors='#BDC3C7')
    for spine in ax.spines.values():
        spine.set_color('#30363D')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.1, color='white')

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'depth_scaling_v2.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0D1117')
    plt.close()
    print('  Saved: %s' % path)
    return path


# =============================================================================
# MAIN
# =============================================================================
def main():
    start = time.time()

    # Exp 1: Qwen GPU
    qwen_result = run_qwen_gpu()

    # Exp 2: Phi-2
    phi2_result = run_phi2()

    # Depth Scaling plot
    plot_depth_scaling(qwen_result, phi2_result)

    elapsed = time.time() - start

    # Save results
    save_data = {
        'qwen_gpu': qwen_result,
        'phi2': phi2_result,
        'elapsed_seconds': round(elapsed, 1),
    }
    json_path = os.path.join(RESULTS_DIR, 'final_phase_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print('\n' + '=' * 70)
    print('  RESULTS SUMMARY')
    print('=' * 70)
    if qwen_result:
        print('  Qwen2.5 GPU: Peak L%d (%.0f%%) ΔH=%.4f' % (
            qwen_result['peak_layer'], qwen_result['peak_depth_pct'],
            qwen_result['peak_delta']))
    else:
        print('  Qwen2.5: FAILED')
    if phi2_result:
        print('  Phi-2: Peak L%d (%.0f%%) ΔH=%.4f' % (
            phi2_result['peak_layer'], phi2_result['peak_depth_pct'],
            phi2_result['peak_delta']))
    else:
        print('  Phi-2: FAILED')
    print('  Elapsed: %.0fs' % elapsed)
    print('  JSON: %s' % json_path)


if __name__ == '__main__':
    main()
