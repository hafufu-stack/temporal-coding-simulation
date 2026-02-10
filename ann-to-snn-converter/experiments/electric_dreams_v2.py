"""
Electric Dreams v2 — Improved Noise Injection
=============================================================================
Fix: Use register_forward_pre_hook to inject noise INTO hidden states
BEFORE attention computation, so the canary head naturally responds to noise.

Approach:
  - Hook into the canary layer's input
  - Add Gaussian noise to hidden states
  - This perturbs ALL attention heads in that layer
  - Measure if canary head entropy changes more than other heads
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

QUESTION = "What is the capital of France?"
CANARY_LAYER = 10
CANARY_HEAD = 17


def run_electric_dreams_v2():
    """Inject noise into hidden states at canary layer, measure head-wise entropy."""
    print("="*70)
    print("ELECTRIC DREAMS v2 — Hidden State Noise Injection")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_name = "mistralai/Mistral-7B-v0.1"
    print(f"Loading {model_name}...")
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
    
    prompt = f"Question: {QUESTION}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=256)
    input_ids = inputs['input_ids'].to(device)
    
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    
    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    
    all_dreams = []
    
    for noise_std in noise_levels:
        # Pre-hook: inject noise into hidden states BEFORE the canary layer processes them
        hook_handle = None
        
        if noise_std > 0:
            def make_pre_hook(std):
                def pre_hook(module, args):
                    # args[0] is the hidden_states tensor
                    hidden_states = args[0]
                    noise = torch.randn_like(hidden_states) * std
                    noisy_hidden = hidden_states + noise
                    # Return modified args tuple
                    return (noisy_hidden,) + args[1:]
                return pre_hook
            
            hook_handle = model.model.layers[CANARY_LAYER].register_forward_pre_hook(
                make_pre_hook(noise_std)
            )
        
        # Forward pass with attentions
        with torch.no_grad():
            out = model(input_ids, output_attentions=True, use_cache=False)
        
        # Compute head-wise entropy for ALL layers
        layer_head_ents = []
        for l_idx, attn in enumerate(out.attentions):
            a = attn[0, :, -1, :].float()  # (heads, src_len)
            a = torch.where(torch.isnan(a), torch.zeros_like(a), a)
            a = a.clamp(min=1e-10)
            head_ent = -(a * torch.log2(a)).sum(dim=-1)  # (heads,)
            head_ent = torch.where(torch.isnan(head_ent) | torch.isinf(head_ent),
                                   torch.zeros_like(head_ent), head_ent)
            layer_head_ents.append(head_ent.cpu().numpy())
        
        all_ents = np.array(layer_head_ents)  # (layers, heads)
        
        # Key metrics
        canary_ent = float(all_ents[CANARY_LAYER, CANARY_HEAD])
        # Mean entropy of all heads in canary layer
        layer_mean = float(all_ents[CANARY_LAYER].mean())
        # Mean entropy of canary head across all layers
        head_mean = float(all_ents[:, CANARY_HEAD].mean())
        # Global mean
        global_mean = float(all_ents.mean())
        
        # Generate dream text
        if hook_handle:
            hook_handle.remove()
        
        # Re-hook for generation if noise > 0
        gen_hook = None
        if noise_std > 0:
            def make_gen_hook(std):
                def pre_hook(module, args):
                    hidden_states = args[0]
                    noise = torch.randn_like(hidden_states) * std
                    return (hidden_states + noise,) + args[1:]
                return pre_hook
            gen_hook = model.model.layers[CANARY_LAYER].register_forward_pre_hook(
                make_gen_hook(noise_std)
            )
        
        with torch.no_grad():
            gen_out = model.generate(
                input_ids, max_new_tokens=30,
                do_sample=True, temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        dream_text = tokenizer.decode(gen_out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        if gen_hook:
            gen_hook.remove()
        
        # Determine if nightmare
        baseline_ent = all_dreams[0]["canary_entropy"] if all_dreams else canary_ent
        is_nightmare = canary_ent > baseline_ent * 1.2 if all_dreams else False
        
        dream = {
            "noise_std": noise_std,
            "canary_entropy": round(canary_ent, 4),
            "canary_layer_mean": round(layer_mean, 4),
            "canary_head_mean": round(head_mean, 4),
            "global_mean": round(global_mean, 4),
            "canary_sensitivity": round(canary_ent / layer_mean, 4) if layer_mean > 0 else 0,
            "dream_text": dream_text[:120],
            "is_nightmare": bool(is_nightmare),
            "all_layer_entropies": all_ents.tolist()
        }
        all_dreams.append(dream)
        
        tag = "🌙 NIGHTMARE!" if is_nightmare else "😴 Normal"
        print(f"  σ={noise_std:.2f}: "
              f"H_canary={canary_ent:.4f} (layer_mean={layer_mean:.4f}) "
              f"sensitivity={canary_ent/layer_mean:.2f}x "
              f"[{tag}]")
        print(f"    dream: \"{dream_text[:70]}\"")
        
        del out, gen_out
        torch.cuda.empty_cache()
    
    # Analyze: how does canary respond vs other heads?
    print("\n--- CANARY SENSITIVITY ANALYSIS ---")
    baseline = all_dreams[0]
    for d in all_dreams[1:]:
        delta_canary = d["canary_entropy"] - baseline["canary_entropy"]
        delta_global = d["global_mean"] - baseline["global_mean"]
        ratio = delta_canary / delta_global if abs(delta_global) > 1e-6 else 999.0
        print(f"  σ={d['noise_std']:.2f}: ΔH_canary={delta_canary:+.4f}, "
              f"ΔH_global={delta_global:+.4f}, ratio={ratio:.2f}x")
    
    # Cleanup
    del model, tokenizer
    torch.cuda.empty_cache()
    
    return all_dreams


def plot_electric_dreams_v2(dreams):
    """Visualize Electric Dreams v2 results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Electric Dreams v2: Hidden State Noise Injection\n"
                 f"Canary: L{CANARY_LAYER}H{CANARY_HEAD} (Mistral-7B)",
                 fontsize=14, fontweight='bold')
    
    noise_levels = [d["noise_std"] for d in dreams]
    canary_ents = [d["canary_entropy"] for d in dreams]
    layer_means = [d["canary_layer_mean"] for d in dreams]
    global_means = [d["global_mean"] for d in dreams]
    sensitivities = [d["canary_sensitivity"] for d in dreams]
    nightmares = [d["is_nightmare"] for d in dreams]
    
    # Panel 1: Canary entropy vs noise
    ax1 = axes[0, 0]
    colors = ['red' if n else 'steelblue' for n in nightmares]
    ax1.bar(range(len(noise_levels)), canary_ents, color=colors, alpha=0.8)
    ax1.plot(range(len(noise_levels)), layer_means, 'g--o', label='Layer Mean', markersize=5)
    ax1.plot(range(len(noise_levels)), global_means, 'k--s', label='Global Mean', markersize=4)
    ax1.set_xticks(range(len(noise_levels)))
    ax1.set_xticklabels([f"{n:.2f}" for n in noise_levels])
    ax1.set_xlabel("Noise σ")
    ax1.set_ylabel("Entropy (bits)")
    ax1.set_title("Canary Head Entropy vs Noise\n(Red = Nightmare)")
    ax1.legend(fontsize=8)
    if canary_ents:
        ax1.axhline(y=canary_ents[0], color='gray', linestyle=':', alpha=0.4, label='Baseline')
    
    # Panel 2: Sensitivity ratio
    ax2 = axes[0, 1]
    ax2.plot(noise_levels, sensitivities, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel("Noise σ")
    ax2.set_ylabel("Canary / Layer Mean")
    ax2.set_title("Canary Sensitivity Ratio\n(>1.0 = canary more sensitive)")
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Entropy change from baseline
    ax3 = axes[1, 0]
    if len(dreams) > 1:
        baseline_canary = dreams[0]["canary_entropy"]
        baseline_global = dreams[0]["global_mean"]
        delta_canary = [d["canary_entropy"] - baseline_canary for d in dreams]
        delta_global = [d["global_mean"] - baseline_global for d in dreams]
        ax3.plot(noise_levels, delta_canary, 'r-o', label=f'L{CANARY_LAYER}H{CANARY_HEAD}', linewidth=2)
        ax3.plot(noise_levels, delta_global, 'b-s', label='Global Mean', linewidth=2)
        ax3.set_xlabel("Noise σ")
        ax3.set_ylabel("ΔEntropy from Baseline (bits)")
        ax3.set_title("Entropy Change: Canary vs Global")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Panel 4: Dream Gallery
    ax4 = axes[1, 1]
    ax4.axis('off')
    text = "🛌 Electric Dreams Gallery\n" + "="*45 + "\n\n"
    for d in dreams:
        tag = "🌙" if d["is_nightmare"] else "😴"
        text += f"{tag} σ={d['noise_std']:.2f}: H={d['canary_entropy']:.4f}\n"
        text += f"   \"{d['dream_text'][:65]}...\"\n\n"
    ax4.text(0.02, 0.98, text, transform=ax4.transAxes,
            fontsize=7.5, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(RESULTS_DIR, "electric_dreams_v2.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n📊 Saved: {path}")
    return path


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


if __name__ == '__main__':
    dreams = run_electric_dreams_v2()
    
    # Save (without full layer data for readability)
    dreams_summary = []
    for d in dreams:
        summary = {k: v for k, v in d.items() if k != "all_layer_entropies"}
        dreams_summary.append(summary)
    
    try:
        with open(os.path.join(RESULTS_DIR, "electric_dreams_v2.json"), "w") as f:
            json.dump(dreams_summary, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        print("JSON saved!")
    except Exception as e:
        print(f"JSON save error: {e}")
    
    try:
        plot_electric_dreams_v2(dreams)
    except Exception as e:
        print(f"Plot error: {e}")
        import traceback; traceback.print_exc()
    
    print("\nDone!")

