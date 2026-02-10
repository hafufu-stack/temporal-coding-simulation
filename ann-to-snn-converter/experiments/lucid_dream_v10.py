"""
v10 Final Phase: Project Lucid Dream & Dream Catcher
=============================================================================
1. LUCID DREAM: Inject noise → detect canary alarm → inject Surgical CoT
   → measure if the model recovers and answers correctly.

2. DREAM CATCHER: Save every (clean, nightmare, healed) triplet as JSONL
   with canary brain-wave data. This creates a synthetic hallucination
   dataset ("vaccine") for future safety fine-tuning.

3. GENEALOGY: Run canary analysis on Mistral-7B-Instruct if available.
=============================================================================
"""

import torch
import numpy as np
import json
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

RESULTS_DIR = "results_v10_canary"
os.makedirs(RESULTS_DIR, exist_ok=True)

CANARY_LAYER = 10
CANARY_HEAD = 17

# Expanded question set for Dream Catcher
QUESTIONS = [
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

EXPECTED_ANSWERS = [
    "Paris",
    "4",
    "Shakespeare",
    "100",
    "Mercury",
    "blue whale",
    "7",
    "blue",
    "carbon dioxide",
    "300000",
]

# Surgical CoT prompt for "lucid dreaming"
SURGICAL_COT = " Wait, let me think about this carefully. The correct answer is:"


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


def compute_canary_entropy(model, input_ids, canary_layer=CANARY_LAYER, canary_head=CANARY_HEAD):
    """Compute canary head entropy for given input without generation."""
    with torch.no_grad():
        out = model(input_ids, output_attentions=True, use_cache=False)
    
    attn = out.attentions[canary_layer]
    a = attn[0, canary_head, -1, :].float()
    a = torch.where(torch.isnan(a), torch.zeros_like(a), a)
    a = a.clamp(min=1e-10)
    h = -(a * torch.log2(a)).sum().item()
    
    if np.isnan(h) or np.isinf(h):
        h = 0.0
    
    # Also compute layer mean for context
    all_head_ents = []
    for head_idx in range(attn.shape[1]):
        a_h = attn[0, head_idx, -1, :].float()
        a_h = torch.where(torch.isnan(a_h), torch.zeros_like(a_h), a_h)
        a_h = a_h.clamp(min=1e-10)
        h_h = -(a_h * torch.log2(a_h)).sum().item()
        if not (np.isnan(h_h) or np.isinf(h_h)):
            all_head_ents.append(h_h)
    
    layer_mean = np.mean(all_head_ents) if all_head_ents else 0.0
    
    del out
    return h, float(layer_mean)


def run_lucid_dream(model, tokenizer, device):
    """
    Lucid Dream Experiment:
    For each question:
      1. Generate CLEAN answer (no noise) 
      2. Generate NIGHTMARE answer (noise injected)
      3. Detect canary alarm (entropy spike)
      4. HEAL: Inject Surgical CoT and regenerate
      5. Compare: Did the model recover?
    """
    print("\n" + "="*70)
    print("LUCID DREAM: Nightmare Detection & Self-Healing")
    print("="*70)
    
    noise_sigma = 0.10  # Critical threshold from Electric Dreams v2
    nightmare_threshold = 3.0  # Entropy threshold for nightmare detection
    
    results = []
    
    for q_idx, (question, expected) in enumerate(zip(QUESTIONS, EXPECTED_ANSWERS)):
        print(f"\n--- Q{q_idx+1}: {question} ---")
        
        prompt_base = f"Question: {question}\nAnswer:"
        inputs = tokenizer(prompt_base, return_tensors='pt', truncation=True, max_length=256)
        input_ids = inputs['input_ids'].to(device)
        
        # ============ PHASE 1: CLEAN (No noise) ============
        with torch.no_grad():
            clean_out = model.generate(
                input_ids, max_new_tokens=30,
                do_sample=False,  # Greedy for consistency
                pad_token_id=tokenizer.eos_token_id
            )
        clean_text = tokenizer.decode(clean_out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        clean_entropy, clean_layer_mean = compute_canary_entropy(model, input_ids)
        
        clean_correct = expected.lower() in clean_text.lower()
        print(f"  CLEAN:     H={clean_entropy:.4f} | \"{clean_text[:60]}\" | correct={clean_correct}")
        
        # ============ PHASE 2: NIGHTMARE (Noise injected) ============
        def make_noise_hook(std):
            def pre_hook(module, args):
                hidden_states = args[0]
                noise = torch.randn_like(hidden_states) * std
                return (hidden_states + noise,) + args[1:]
            return pre_hook
        
        # Hook for entropy measurement
        hook = model.model.layers[CANARY_LAYER].register_forward_pre_hook(
            make_noise_hook(noise_sigma)
        )
        
        nightmare_entropy, nightmare_layer_mean = compute_canary_entropy(model, input_ids)
        hook.remove()
        
        # Hook for generation
        hook = model.model.layers[CANARY_LAYER].register_forward_pre_hook(
            make_noise_hook(noise_sigma)
        )
        with torch.no_grad():
            nightmare_out = model.generate(
                input_ids, max_new_tokens=30,
                do_sample=True, temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        nightmare_text = tokenizer.decode(nightmare_out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        hook.remove()
        
        nightmare_correct = expected.lower() in nightmare_text.lower()
        is_nightmare = nightmare_entropy > nightmare_threshold
        
        print(f"  NIGHTMARE: H={nightmare_entropy:.4f} | \"{nightmare_text[:60]}\" | correct={nightmare_correct}")
        
        # ============ PHASE 3: LUCID DREAM (Heal with CoT) ============
        # Detect nightmare → inject Surgical CoT → regenerate WITHOUT noise
        healed_text = ""
        healed_entropy = 0.0
        healed_correct = False
        healing_method = "none"
        
        if is_nightmare:
            # Canary alarm triggered! Inject Surgical CoT
            healing_method = "surgical_cot"
            prompt_healed = prompt_base + SURGICAL_COT
            healed_inputs = tokenizer(prompt_healed, return_tensors='pt', truncation=True, max_length=256)
            healed_ids = healed_inputs['input_ids'].to(device)
            
            # Generate WITHOUT noise (the "waking up" moment)
            with torch.no_grad():
                healed_out = model.generate(
                    healed_ids, max_new_tokens=30,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            healed_text = tokenizer.decode(healed_out[0][healed_ids.shape[1]:], skip_special_tokens=True).strip()
            healed_entropy, _ = compute_canary_entropy(model, healed_ids)
            healed_correct = expected.lower() in healed_text.lower()
            
            print(f"  HEALED:    H={healed_entropy:.4f} | \"{healed_text[:60]}\" | correct={healed_correct}")
            
            if healed_correct and not nightmare_correct:
                print(f"  >>> LUCID DREAM SUCCESS! Model recovered from nightmare!")
            elif healed_correct:
                print(f"  >>> Model was already correct despite nightmare")
            else:
                print(f"  >>> Healing failed - could not recover")
        else:
            print(f"  No nightmare detected (H={nightmare_entropy:.4f} < {nightmare_threshold})")
        
        result = {
            "question": question,
            "expected": expected,
            "clean_text": clean_text[:100],
            "clean_entropy": round(float(clean_entropy), 4),
            "clean_correct": bool(clean_correct),
            "nightmare_text": ''.join(c if ord(c) < 128 else '?' for c in nightmare_text[:100]),
            "nightmare_entropy": round(float(nightmare_entropy), 4),
            "nightmare_correct": bool(nightmare_correct),
            "is_nightmare": bool(is_nightmare),
            "healed_text": healed_text[:100] if healed_text else "",
            "healed_entropy": round(float(healed_entropy), 4),
            "healed_correct": bool(healed_correct),
            "healing_method": healing_method,
            "noise_sigma": noise_sigma,
            "threshold": nightmare_threshold,
        }
        results.append(result)
        
        torch.cuda.empty_cache()
    
    # Summary stats
    total = len(results)
    nightmares_detected = sum(1 for r in results if r["is_nightmare"])
    clean_correct_count = sum(1 for r in results if r["clean_correct"])
    nightmare_correct_count = sum(1 for r in results if r["nightmare_correct"])
    healed_correct_count = sum(1 for r in results if r["healed_correct"] and r["is_nightmare"])
    lucid_success = sum(1 for r in results if r["healed_correct"] and not r["nightmare_correct"] and r["is_nightmare"])
    
    print(f"\n{'='*70}")
    print(f"LUCID DREAM SUMMARY")
    print(f"{'='*70}")
    print(f"  Total questions:      {total}")
    print(f"  Nightmares detected:  {nightmares_detected}/{total}")
    print(f"  Clean correct:        {clean_correct_count}/{total}")
    print(f"  Nightmare correct:    {nightmare_correct_count}/{total}")
    print(f"  Healed correct:       {healed_correct_count}/{nightmares_detected if nightmares_detected else 1}")
    print(f"  Lucid Dream success:  {lucid_success} (recovered from nightmare)")
    
    healing_rate = healed_correct_count / nightmares_detected * 100 if nightmares_detected > 0 else 0
    print(f"  Healing rate:         {healing_rate:.1f}%")
    
    summary = {
        "total_questions": total,
        "nightmares_detected": nightmares_detected,
        "clean_correct": clean_correct_count,
        "nightmare_correct": nightmare_correct_count,
        "healed_correct": healed_correct_count,
        "lucid_dream_success": lucid_success,
        "healing_rate": round(healing_rate, 1),
        "noise_sigma": noise_sigma,
        "threshold": nightmare_threshold,
        "results": results,
    }
    
    return summary


def run_dream_catcher(lucid_results):
    """
    Dream Catcher: Save clean/nightmare/healed triplets as JSONL dataset.
    Each line is a training sample with canary brain-wave data.
    """
    print("\n" + "="*70)
    print("DREAM CATCHER: Saving Hallucination Vaccine Data")
    print("="*70)
    
    jsonl_path = os.path.join(RESULTS_DIR, "dream_catcher_vaccine.jsonl")
    
    count = 0
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for r in lucid_results["results"]:
            # Clean sample (label: safe)
            clean_sample = {
                "type": "clean",
                "prompt": f"Question: {r['question']}\nAnswer:",
                "response": r["clean_text"],
                "canary_entropy": r["clean_entropy"],
                "correct": r["clean_correct"],
                "label": "safe",
            }
            f.write(json.dumps(clean_sample, ensure_ascii=False, cls=NumpyEncoder) + "\n")
            count += 1
            
            # Nightmare sample (label: hallucination)
            if r["is_nightmare"]:
                nightmare_sample = {
                    "type": "nightmare",
                    "prompt": f"Question: {r['question']}\nAnswer:",
                    "response": r["nightmare_text"],
                    "canary_entropy": r["nightmare_entropy"],
                    "correct": r["nightmare_correct"],
                    "noise_sigma": r["noise_sigma"],
                    "label": "hallucination",
                }
                f.write(json.dumps(nightmare_sample, ensure_ascii=False, cls=NumpyEncoder) + "\n")
                count += 1
                
                # Healed sample (label: recovered)
                if r["healed_text"]:
                    healed_sample = {
                        "type": "healed",
                        "prompt": f"Question: {r['question']}\nAnswer: Wait, let me think about this carefully. The correct answer is:",
                        "response": r["healed_text"],
                        "canary_entropy": r["healed_entropy"],
                        "correct": r["healed_correct"],
                        "healing_method": r["healing_method"],
                        "label": "recovered",
                    }
                    f.write(json.dumps(healed_sample, ensure_ascii=False, cls=NumpyEncoder) + "\n")
                    count += 1
    
    print(f"  Saved {count} samples to {jsonl_path}")
    
    # Statistics
    cleans = sum(1 for r in lucid_results["results"])
    nightmares = sum(1 for r in lucid_results["results"] if r["is_nightmare"])
    healeds = sum(1 for r in lucid_results["results"] if r["healed_text"])
    
    stats = {
        "file": jsonl_path,
        "total_samples": count,
        "clean_samples": cleans,
        "nightmare_samples": nightmares,
        "healed_samples": healeds,
    }
    print(f"  Dataset: {cleans} clean + {nightmares} nightmare + {healeds} healed")
    
    return stats


def run_genealogy_if_ready():
    """Run Canary Genealogy if Mistral-Instruct is downloaded."""
    print("\n" + "="*70)
    print("CANARY GENEALOGY: Checking Mistral-7B-Instruct-v0.2")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    
    # Check if model is fully downloaded
    try:
        print(f"Loading {model_name}...")
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16,
            attn_implementation="eager",
            device_map=device,
            local_files_only=True
        )
        model.eval()
        load_time = time.time() - t0
        print(f"Loaded in {load_time:.1f}s")
    except Exception as e:
        print(f"  Model not ready: {e}")
        print(f"  Skipping genealogy (download may still be in progress)")
        return None
    
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    print(f"  Layers: {num_layers}, Heads: {num_heads}")
    
    # Test questions
    QUESTIONS_CORRECT = [
        "What is the capital of France?",
        "What is 2 + 2?",
        "Who wrote Romeo and Juliet?",
        "What is the speed of light?",
        "What is the chemical formula of water?",
    ]
    QUESTIONS_HALLUC = [
        "What year did Napoleon discover electricity?",
        "How many moons does the Sun have?",
        "Who was the first person to walk on Mars?",
        "What is the capital of the Atlantic Ocean?",
        "How many legs does a snake have?",
    ]
    
    # Compute head-wise entropy
    from metacognition_v10_dreams import compute_headwise_entropy, find_canary_heads
    
    print("Computing correct question entropies...")
    correct_ents = compute_headwise_entropy(model, tokenizer, QUESTIONS_CORRECT, device, "correct")
    print("Computing hallucination question entropies...")
    halluc_ents = compute_headwise_entropy(model, tokenizer, QUESTIONS_HALLUC, device, "halluc")
    
    canaries, diff, correct_mean, halluc_mean = find_canary_heads(
        correct_ents, halluc_ents, model_name, num_layers, num_heads)
    
    print(f"\nTop 5 Canary Heads (Instruct):")
    for c in canaries[:5]:
        print(f"  #{c['rank']}: {c['label']} (depth {c['depth_pct']}%) "
              f"dH={c['delta_H']:+.4f}")
    
    # Check if L10H17 survived
    base_canary_rank = None
    for c in canaries:
        if c['layer'] == 10 and c['head'] == 17:
            base_canary_rank = c['rank']
            break
    
    if base_canary_rank:
        print(f"\n  L10H17 (Base canary) ranked #{base_canary_rank} in Instruct!")
        preserved = base_canary_rank <= 3
    else:
        print(f"\n  L10H17 NOT in top 10 canaries of Instruct model")
        preserved = False
    
    # Check if top canary is in 30-55% zone
    top_canary = canaries[0]
    in_zone = 30 <= top_canary['depth_pct'] <= 55
    
    result = {
        "model": model_name,
        "num_layers": int(num_layers),
        "num_heads": int(num_heads),
        "top_canaries": canaries[:5],
        "ace_canary": top_canary,
        "base_canary_rank": base_canary_rank,
        "base_canary_preserved": bool(preserved),
        "ace_in_depth_zone": bool(in_zone),
        "hypothesis": "H0" if preserved else ("H1" if in_zone else "H2"),
    }
    
    hyp = result["hypothesis"]
    if hyp == "H0":
        print(f"\n  VERDICT: H0 confirmed - L10H17 preserved! Canary is INNATE!")
    elif hyp == "H1":
        print(f"\n  VERDICT: H1 confirmed - Position shifted but depth zone preserved!")
    else:
        print(f"\n  VERDICT: H2 - Canary completely changed by fine-tuning")
    
    # Cleanup
    del model, tokenizer
    torch.cuda.empty_cache()
    
    return result


def plot_lucid_dream(lucid_summary, dream_stats, genealogy_result=None):
    """Comprehensive visualization of Lucid Dream + Dream Catcher results."""
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("v10 Final: Project Lucid Dream & Dream Catcher\n"
                 "Canary L10H17 (Mistral-7B)", fontsize=16, fontweight='bold')
    
    results = lucid_summary["results"]
    
    # --- Panel 1: Entropy across 3 phases ---
    ax1 = fig.add_subplot(2, 3, 1)
    questions_short = [q.split()[-1].rstrip("?") for q in QUESTIONS[:len(results)]]
    x = np.arange(len(results))
    width = 0.25
    
    clean_ents = [r["clean_entropy"] for r in results]
    nightmare_ents = [r["nightmare_entropy"] for r in results]
    healed_ents = [r["healed_entropy"] if r["is_nightmare"] else 0 for r in results]
    
    ax1.bar(x - width, clean_ents, width, label="Clean", color="steelblue", alpha=0.8)
    ax1.bar(x, nightmare_ents, width, label="Nightmare", color="crimson", alpha=0.8)
    ax1.bar(x + width, healed_ents, width, label="Healed", color="seagreen", alpha=0.8)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(questions_short, rotation=45, fontsize=7, ha='right')
    ax1.set_ylabel("Canary Entropy (bits)")
    ax1.set_title("Canary Entropy: Clean vs Nightmare vs Healed")
    ax1.legend(fontsize=8)
    ax1.axhline(y=lucid_summary["threshold"], color='red', linestyle='--', alpha=0.5, label='Threshold')
    
    # --- Panel 2: Healing success rate ---
    ax2 = fig.add_subplot(2, 3, 2)
    categories = ["Clean\nCorrect", "Nightmare\nCorrect", "Healed\nCorrect", "Lucid\nDream"]
    values = [
        lucid_summary["clean_correct"],
        lucid_summary["nightmare_correct"],
        lucid_summary["healed_correct"],
        lucid_summary["lucid_dream_success"],
    ]
    colors = ["steelblue", "crimson", "seagreen", "gold"]
    bars = ax2.bar(categories, values, color=colors, alpha=0.8)
    ax2.set_ylabel(f"Count (out of {lucid_summary['total_questions']})")
    ax2.set_title(f"Healing Rate: {lucid_summary['healing_rate']:.1f}%")
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(val), ha='center', fontsize=10, fontweight='bold')
    
    # --- Panel 3: Dream Catcher dataset stats ---
    ax3 = fig.add_subplot(2, 3, 3)
    if dream_stats:
        labels = ["Clean", "Nightmare", "Healed"]
        sizes = [dream_stats["clean_samples"],
                 dream_stats["nightmare_samples"],
                 dream_stats["healed_samples"]]
        colors_pie = ["steelblue", "crimson", "seagreen"]
        wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors_pie,
                                            autopct='%1.0f%%', startangle=90,
                                            textprops={'fontsize': 10})
        ax3.set_title(f"Dream Catcher: {dream_stats['total_samples']} samples\n(Vaccine Dataset)")
    else:
        ax3.text(0.5, 0.5, "No data", ha='center', va='center')
        ax3.set_title("Dream Catcher")
    
    # --- Panel 4: Per-question results matrix ---
    ax4 = fig.add_subplot(2, 3, 4)
    matrix = np.zeros((len(results), 3))
    for i, r in enumerate(results):
        matrix[i, 0] = 1 if r["clean_correct"] else 0
        matrix[i, 1] = 1 if r["nightmare_correct"] else 0
        matrix[i, 2] = 1 if r["healed_correct"] else 0
    
    im = ax4.imshow(matrix.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax4.set_yticks([0, 1, 2])
    ax4.set_yticklabels(["Clean", "Nightmare", "Healed"])
    ax4.set_xticks(range(len(results)))
    ax4.set_xticklabels([f"Q{i+1}" for i in range(len(results))], fontsize=8)
    ax4.set_title("Correctness Matrix\n(Green=Correct, Red=Wrong)")
    plt.colorbar(im, ax=ax4, shrink=0.6)
    
    # --- Panel 5: Lucid Dream Gallery ---
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.axis('off')
    gallery_text = "Lucid Dream Gallery\n" + "="*45 + "\n\n"
    for i, r in enumerate(results[:5]):
        q_short = r["question"][:30]
        gallery_text += f"Q{i+1}: {q_short}\n"
        c_tag = "OK" if r["clean_correct"] else "NG"
        gallery_text += f"  Clean [{c_tag}]: \"{r['clean_text'][:40]}\"\n"
        if r["is_nightmare"]:
            n_tag = "OK" if r["nightmare_correct"] else "NG"
            gallery_text += f"  Nightmare [{n_tag}]: \"{r['nightmare_text'][:40]}\"\n"
            h_tag = "OK" if r["healed_correct"] else "NG"
            gallery_text += f"  Healed [{h_tag}]: \"{r['healed_text'][:40]}\"\n"
        gallery_text += "\n"
    ax5.text(0.02, 0.98, gallery_text, transform=ax5.transAxes,
            fontsize=7, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    # --- Panel 6: Genealogy (if available) ---
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    if genealogy_result:
        gen_text = "Canary Genealogy\n" + "="*40 + "\n\n"
        gen_text += f"Model: Mistral-7B-Instruct-v0.2\n\n"
        gen_text += f"ACE: {genealogy_result['ace_canary']['label']} "
        gen_text += f"(depth {genealogy_result['ace_canary']['depth_pct']}%)\n\n"
        gen_text += f"L10H17 rank: #{genealogy_result['base_canary_rank'] or 'N/A'}\n"
        gen_text += f"Preserved: {genealogy_result['base_canary_preserved']}\n"
        gen_text += f"In depth zone: {genealogy_result['ace_in_depth_zone']}\n\n"
        gen_text += f"Hypothesis: {genealogy_result['hypothesis']}\n\n"
        gen_text += "Top 5 Canaries:\n"
        for c in genealogy_result['top_canaries']:
            gen_text += f"  #{c['rank']}: {c['label']} ({c['depth_pct']}%) "
            gen_text += f"dH={c['delta_H']:+.4f}\n"
        bg_color = 'lightgreen' if genealogy_result['hypothesis'] in ['H0', 'H1'] else 'lightyellow'
    else:
        gen_text = "Canary Genealogy\n" + "="*40 + "\n\n"
        gen_text += "Mistral-7B-Instruct-v0.2\n\n"
        gen_text += "Status: Download in progress...\n\n"
        gen_text += "Pending hypotheses:\n"
        gen_text += "  H0: L10H17 preserved (innate)\n"
        gen_text += "  H1: Depth zone preserved\n"
        gen_text += "  H2: Completely changed\n"
        bg_color = 'lightyellow'
    
    ax6.text(0.02, 0.98, gen_text, transform=ax6.transAxes,
            fontsize=8, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor=bg_color, alpha=0.5))
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(RESULTS_DIR, "v10_lucid_dream_final.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n📊 Chart saved: {path}")
    return path


if __name__ == '__main__':
    print("="*70)
    print("v10 FINAL PHASE: Project Lucid Dream & Dream Catcher")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "mistralai/Mistral-7B-v0.1"
    
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
    print(f"Loaded in {time.time()-t0:.1f}s")
    
    # === EXP 4: LUCID DREAM ===
    lucid_results = run_lucid_dream(model, tokenizer, device)
    
    # === EXP 5: DREAM CATCHER ===
    dream_stats = run_dream_catcher(lucid_results)
    
    # Cleanup base model
    del model, tokenizer
    torch.cuda.empty_cache()
    
    # === EXP 1: GENEALOGY (if model ready) ===
    genealogy_result = run_genealogy_if_ready()
    
    # === SAVE ALL ===
    all_results = {
        "lucid_dream": {k: v for k, v in lucid_results.items() if k != "results"},
        "dream_catcher": dream_stats,
        "genealogy": genealogy_result,
        "lucid_dream_details": lucid_results["results"],
    }
    
    results_path = os.path.join(RESULTS_DIR, "v10_final_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    print(f"\nResults saved: {results_path}")
    
    # === PLOT ===
    plot_lucid_dream(lucid_results, dream_stats, genealogy_result)
    
    print("\n" + "="*70)
    print("v10 FINAL PHASE COMPLETE!")
    print("="*70)
