"""
v10 EXP1: Canary Genealogy
===========================
Does L10H17 survive instruction tuning?
- H0: L10H17 preserved (canary is innate)
- H1: Position shifts but stays in 30-55% zone
- H2: Completely different (fine-tuning erases canary)
"""

import torch
import numpy as np
import json
import os
import time
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

RESULTS_DIR = "results_v10_canary"
os.makedirs(RESULTS_DIR, exist_ok=True)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def compute_headwise_entropy(model, tokenizer, questions, device, label=""):
    """Compute per-layer, per-head attention entropy."""
    all_entropies = []
    for qi, q in enumerate(questions):
        prompt = f"Question: {q}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=256)
        input_ids = inputs['input_ids'].to(device)
        
        with torch.no_grad():
            out = model(input_ids, output_attentions=True, use_cache=False)
        
        step_ents = []
        for l_idx, attn in enumerate(out.attentions):
            a = attn[0, :, -1, :].float()
            a = torch.where(torch.isnan(a), torch.zeros_like(a), a)
            a = a.clamp(min=1e-10)
            a_log = torch.log2(a)
            head_ent = -(a * a_log).sum(dim=-1)
            head_ent = torch.where(torch.isnan(head_ent) | torch.isinf(head_ent),
                                   torch.zeros_like(head_ent), head_ent)
            step_ents.append(head_ent.cpu().numpy())
        
        all_entropies.append(step_ents)
        del out
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()
        
        print(f"  [{label}] Q{qi+1}/{len(questions)}: {q[:40]}...")
    
    return np.array(all_entropies)


def find_canary_heads(correct_ents, halluc_ents, num_layers, num_heads):
    """Find top canary heads by absolute entropy differential."""
    correct_mean = correct_ents.mean(axis=0)
    halluc_mean = halluc_ents.mean(axis=0)
    diff = halluc_mean - correct_mean
    
    flat_diff = diff.flatten()
    flat_abs = np.abs(flat_diff)
    top_indices = np.argsort(flat_abs)[::-1][:10]
    
    canaries = []
    for idx in top_indices:
        l = idx // num_heads
        h = idx % num_heads
        canaries.append({
            "rank": len(canaries) + 1,
            "layer": int(l),
            "head": int(h),
            "label": f"L{l}H{h}",
            "depth_pct": round(l / num_layers * 100, 1),
            "delta_H": round(float(diff[l, h]), 4),
            "abs_delta_H": round(float(flat_abs[idx]), 4),
            "correct_H": round(float(correct_mean[l, h]), 4),
            "halluc_H": round(float(halluc_mean[l, h]), 4),
        })
    
    return canaries, diff, correct_mean, halluc_mean


def run_genealogy():
    print("="*70)
    print("v10 EXP1: CANARY GENEALOGY — Does L10H17 survive instruction tuning?")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- Questions ---
    QUESTIONS_CORRECT = [
        "What is the capital of France?",
        "What is 2 + 2?",
        "Who wrote Romeo and Juliet?",
        "What is the boiling point of water in Celsius?",
        "What planet is closest to the Sun?",
        "What is the largest mammal?",
        "How many continents are there?",
        "What color is the sky on a clear day?",
        "What gas do plants absorb?",
        "What is the speed of light in km/s?",
    ]
    QUESTIONS_HALLUC = [
        "What year did Napoleon discover electricity?",
        "How many moons does the Sun have?",
        "Who was the first person to walk on Mars?",
        "What is the capital of the Atlantic Ocean?",
        "How many legs does a snake have?",
        "What country won World War 3?",
        "Who invented the Internet in 1823?",
        "What is the 8th color of the rainbow?",
        "When did Einstein win the Nobel Prize for gravity?",
        "What is the square root of negative one hundred?",
    ]
    
    # --- Load Instruct model ---
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    print(f"\nLoading {model_name}...")
    t0 = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16,
        attn_implementation="eager",
        device_map=device
    )
    model.eval()
    load_time = time.time() - t0
    print(f"Loaded in {load_time:.1f}s")
    
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    print(f"Layers: {num_layers}, Heads: {num_heads}")
    
    # --- Compute entropies ---
    print("\n--- Computing correct question entropies ---")
    correct_ents = compute_headwise_entropy(model, tokenizer, QUESTIONS_CORRECT, device, "correct")
    
    print("\n--- Computing hallucination question entropies ---")
    halluc_ents = compute_headwise_entropy(model, tokenizer, QUESTIONS_HALLUC, device, "halluc")
    
    # --- Find canaries ---
    canaries, diff, correct_mean, halluc_mean = find_canary_heads(
        correct_ents, halluc_ents, num_layers, num_heads)
    
    print(f"\n{'='*70}")
    print(f"CANARY GENEALOGY RESULTS — {model_name}")
    print(f"{'='*70}")
    print(f"\nTop 10 Canary Heads (Instruct):")
    for c in canaries:
        zone_tag = " [IN ZONE 30-55%]" if 30 <= c['depth_pct'] <= 55 else ""
        base_tag = " <<< BASE CANARY!" if (c['layer'] == 10 and c['head'] == 17) else ""
        print(f"  #{c['rank']:2d}: {c['label']:6s} (depth {c['depth_pct']:5.1f}%) "
              f"dH={c['delta_H']:+.4f} "
              f"(correct={c['correct_H']:.4f}, halluc={c['halluc_H']:.4f})"
              f"{zone_tag}{base_tag}")
    
    # --- Check L10H17 survival ---
    base_canary_rank = None
    base_canary_data = None
    for c in canaries:
        if c['layer'] == 10 and c['head'] == 17:
            base_canary_rank = c['rank']
            base_canary_data = c
            break
    
    # If not in top 10, check raw diff
    if base_canary_rank is None and num_layers > 10 and num_heads > 17:
        base_delta = float(diff[10, 17])
        base_abs = abs(base_delta)
        # Find actual rank
        flat_abs = np.abs(diff.flatten())
        sorted_abs = np.sort(flat_abs)[::-1]
        rank_list = list(sorted_abs)
        try:
            base_canary_rank = rank_list.index(base_abs) + 1
        except ValueError:
            base_canary_rank = None
        base_canary_data = {
            "label": "L10H17",
            "delta_H": round(base_delta, 4),
            "depth_pct": round(10 / num_layers * 100, 1),
        }
    
    # --- Verdict ---
    top_canary = canaries[0]
    top_in_zone = 30 <= top_canary['depth_pct'] <= 55
    
    print(f"\n{'='*70}")
    print(f"VERDICT")
    print(f"{'='*70}")
    print(f"  ACE Canary (Instruct): {top_canary['label']} "
          f"(depth {top_canary['depth_pct']}%, dH={top_canary['delta_H']:+.4f})")
    print(f"  In 30-55% zone: {'YES' if top_in_zone else 'NO'}")
    print(f"  Base canary L10H17 rank: #{base_canary_rank}")
    
    preserved = base_canary_rank is not None and base_canary_rank <= 3
    
    if preserved:
        hypothesis = "H0"
        verdict = "L10H17 PRESERVED! Canary is INNATE to the architecture!"
    elif top_in_zone:
        hypothesis = "H1"
        verdict = "Depth zone preserved! Canary position shifted but stays at 30-55%."
    else:
        hypothesis = "H2"
        verdict = "Canary completely changed. Fine-tuning erases canary identity."
    
    print(f"  Hypothesis: {hypothesis}")
    print(f"  Verdict: {verdict}")
    
    # --- Save results ---
    result = {
        "model": model_name,
        "num_layers": int(num_layers),
        "num_heads": int(num_heads),
        "ace_canary": top_canary,
        "top_10_canaries": canaries,
        "base_canary_L10H17_rank": base_canary_rank,
        "base_canary_preserved": bool(preserved),
        "ace_in_depth_zone": bool(top_in_zone),
        "hypothesis": hypothesis,
        "verdict": verdict,
    }
    
    results_path = os.path.join(RESULTS_DIR, "genealogy_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    print(f"\nResults saved: {results_path}")
    
    # --- Plot: Base vs Instruct canary comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Canary Genealogy: Mistral-7B Base vs Instruct\n"
                 "Does L10H17 survive instruction tuning?", fontsize=14, fontweight='bold')
    
    # Panel 1: Differential heatmap
    ax1 = axes[0]
    im = ax1.imshow(diff.T, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Head")
    ax1.set_title(f"Entropy Diff (Halluc - Correct)\n{model_name.split('/')[-1]}")
    plt.colorbar(im, ax=ax1, shrink=0.8, label="Delta-H (bits)")
    # Mark top canary
    ax1.scatter(top_canary['layer'], top_canary['head'], s=200, marker='*',
               color='gold', edgecolors='black', linewidth=2, zorder=5,
               label=f"ACE: {top_canary['label']}")
    # Mark L10H17
    ax1.scatter(10, 17, s=100, marker='D', color='lime', edgecolors='black',
               linewidth=2, zorder=5, label="L10H17 (Base)")
    ax1.legend(fontsize=8, loc='upper right')
    
    # Panel 2: Top 10 canaries bar chart
    ax2 = axes[1]
    labels = [c['label'] for c in canaries]
    deltas = [c['delta_H'] for c in canaries]
    colors = []
    for c in canaries:
        if c['layer'] == 10 and c['head'] == 17:
            colors.append('lime')
        elif 30 <= c['depth_pct'] <= 55:
            colors.append('steelblue')
        else:
            colors.append('gray')
    
    ax2.barh(range(len(canaries)), [abs(d) for d in deltas], color=colors, alpha=0.8)
    ax2.set_yticks(range(len(canaries)))
    ax2.set_yticklabels(labels)
    ax2.invert_yaxis()
    ax2.set_xlabel("|Delta-H| (bits)")
    ax2.set_title("Top 10 Canary Heads (Instruct)\n"
                  "Blue=In Zone, Green=L10H17, Gray=Outside")
    
    # Panel 3: Verdict text
    ax3 = axes[2]
    ax3.axis('off')
    verdict_text = f"CANARY GENEALOGY VERDICT\n{'='*35}\n\n"
    verdict_text += f"Model: Mistral-7B-Instruct-v0.2\n\n"
    verdict_text += f"ACE Canary: {top_canary['label']}\n"
    verdict_text += f"  Depth: {top_canary['depth_pct']}%\n"
    verdict_text += f"  Delta-H: {top_canary['delta_H']:+.4f}\n\n"
    verdict_text += f"Base L10H17:\n"
    verdict_text += f"  Rank: #{base_canary_rank}\n"
    if base_canary_data:
        verdict_text += f"  Delta-H: {base_canary_data['delta_H']:+.4f}\n"
    verdict_text += f"\n{'-'*35}\n\n"
    verdict_text += f"Hypothesis: {hypothesis}\n\n"
    verdict_text += f"{verdict}\n"
    
    bg = 'lightgreen' if hypothesis in ['H0', 'H1'] else 'lightyellow'
    ax3.text(0.05, 0.95, verdict_text, transform=ax3.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor=bg, alpha=0.5))
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    chart_path = os.path.join(RESULTS_DIR, "genealogy_chart.png")
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Chart saved: {chart_path}")
    
    # Cleanup
    del model, tokenizer
    torch.cuda.empty_cache()
    
    return result


if __name__ == '__main__':
    result = run_genealogy()
    print(f"\n{'='*70}")
    print(f"FINAL: {result['hypothesis']} — {result['verdict']}")
    print(f"{'='*70}")
