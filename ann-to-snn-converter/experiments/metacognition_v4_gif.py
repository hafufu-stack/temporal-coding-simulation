"""
Project Metacognition v4 — Phase 1: "Moment of Lie" GIF + Phase 2: Token Economy
=============================================================================
Generates animated heatmap showing hallucination formation in real-time,
plus Token Economy analysis comparing Always CoT vs Surgical v3 output cost.

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
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import torch

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'results_metacognition_v4')
os.makedirs(RESULTS_DIR, exist_ok=True)

MID_LAYER_START = 10
MID_LAYER_END   = 19  # exclusive

# =============================================================================
# 1. Model Loading
# =============================================================================
print('\n' + '=' * 70)
print('  Project Metacognition v4 — Phase 1+2')
print('  "The Moment of Lie" + Token Economy')
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
print('  Model loaded: %s (%d layers)' % (MODEL_NAME, NUM_LAYERS))
print('  VRAM: %.2f GB' % vram_used)


def make_prompt(question, prefix=''):
    return 'Question: %s\nAnswer: %s' % (question, prefix)


# =============================================================================
# 2. Manual Generation — Same core as v3
# =============================================================================
def generate_manual(question, max_new_tokens=60, temperature=0.7,
                    top_k=50, repetition_penalty=1.2,
                    collect_layer_attention=True,
                    spike_detector=None, prefix=''):
    """Manual token-by-token generation with full instrumentation."""
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

        if spike_detector is not None and not spike_detected:
            is_spike = spike_detector(
                logits_entropies, attn_entropies, mid_layer_entropies, step)
            if is_spike:
                spike_detected = True
                spike_position = step
                token_id = logits.argmax(dim=-1).item()
                tokens_generated.append(tokenizer.decode([token_id]))
                break

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


def check_answer(answer, valid_keywords):
    answer_lower = answer.lower()
    for keyword in valid_keywords:
        if keyword.lower() in answer_lower:
            return True
    return False


# =============================================================================
# 3. Phase 1: "Moment of Lie" GIF Animation
# =============================================================================
def generate_moment_of_lie_gif():
    """
    Generate animated heatmap GIF showing hallucination formation.

    The animation shows a layer x token heatmap growing right as tokens
    are generated. When the model hallucinates, mid-layers (L10-L18)
    "fire red" and then deep layers cool (Lock-in effect).
    """
    print('\n' + '=' * 70)
    print('  Phase 1: "Moment of Lie" GIF Animation')
    print('  "Watch the lie being born in real-time"')
    print('=' * 70)

    # Select questions likely to produce interesting patterns
    gif_questions = [
        ("Who was the first person to walk on Mars?",
         ["no one", "nobody", "hasn't", "has not", "not yet", "no human"]),
        ("Who discovered America first?",
         ["indigenous", "Native", "Viking", "Norse"]),
        ("What was the first programming language?",
         ["Fortran", "Assembly", "Plankalkul", "debated"]),
        ("What is the capital of France?",
         ["Paris"]),  # control: should be correct + stable
    ]

    all_gifs = []

    for qi, (question, keywords) in enumerate(gif_questions):
        print('\n  [%d/%d] %s' % (qi + 1, len(gif_questions), question[:55]))
        answer, traj = generate_manual(question, max_new_tokens=40,
                                       collect_layer_attention=True)
        is_correct = check_answer(answer, keywords)
        label = 'CORRECT' if is_correct else 'HALLUCINATION'
        print('    -> %s: %s' % (label, answer[:60]))

        # Build GIF frames
        layer_data = np.array(traj['layer_entropies'])  # [tokens, layers]
        n_tokens = layer_data.shape[0]
        tokens = traj['tokens']

        if n_tokens < 3:
            print('    [SKIP] Too few tokens')
            continue

        # Create frames: each frame adds one more token column
        frames = []
        for t in range(1, n_tokens + 1):
            fig, ax = plt.subplots(figsize=(12, 6))
            fig.patch.set_facecolor('#1a1a2e')

            # Current slice of the heatmap
            matrix = layer_data[:t, :].T  # [layers, tokens_so_far]

            im = ax.imshow(matrix, aspect='auto', interpolation='nearest',
                           cmap='inferno', origin='lower',
                           vmin=0, vmax=np.percentile(layer_data, 95))

            # Mid-layer band highlight
            ax.axhline(MID_LAYER_START - 0.5, color='#00ff88',
                       linewidth=2, linestyle='--', alpha=0.8)
            ax.axhline(MID_LAYER_END - 0.5, color='#00ff88',
                       linewidth=2, linestyle='--', alpha=0.8)

            # Title with current token highlighted
            current_token = tokens[t-1] if t-1 < len(tokens) else '?'
            title_color = '#ff4444' if not is_correct else '#00ff88'
            ax.set_title(
                'Layer × Token Heatmap  |  Q: "%s"\n'
                'Token %d/%d: "%s"  |  [%s]' % (
                    question[:40], t, n_tokens,
                    current_token.strip()[:15], label),
                fontsize=12, fontweight='bold', color='white', pad=10)

            ax.set_ylabel('Layer', color='white', fontsize=11)
            ax.set_xlabel('Token Position', color='white', fontsize=11)
            ax.set_yticks([0, 7, 10, 15, 18, 23, 31])
            ax.set_yticklabels(['L0', 'L7', 'L10', 'L15', 'L18', 'L23', 'L31'],
                              color='white')
            ax.tick_params(colors='white')

            # Token labels on x-axis
            tick_step = max(1, t // 12)
            tick_pos = list(range(0, t, tick_step))
            tick_labels = [tokens[i].strip()[:6] if i < len(tokens) else ''
                          for i in tick_pos]
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_labels, fontsize=7, rotation=45, color='white')

            # Colorbar
            cb = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
            cb.set_label('Attention Entropy (bits)', fontsize=9, color='white')
            cb.ax.yaxis.set_tick_params(color='white')
            plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')

            # Mid-layer zone label
            ax.text(-0.02, (MID_LAYER_START + MID_LAYER_END - 1) / 2 / (NUM_LAYERS - 1),
                    'HALLUC\nZONE', transform=ax.transAxes, fontsize=8,
                    color='#00ff88', fontweight='bold', ha='right', va='center')

            # Bottom info bar
            mid_ent = traj['mid_layer_entropies'][t-1] if t-1 < len(traj['mid_layer_entropies']) else 0
            logits_ent = traj['logits_entropies'][t-1] if t-1 < len(traj['logits_entropies']) else 0
            fig.text(0.5, 0.01,
                     'Mid-Layer Entropy: %.3f bits  |  Logits Entropy: %.2f bits  |  '
                     'Confidence: %.1f%%' % (
                         mid_ent, logits_ent,
                         traj['top_probs'][t-1] * 100 if t-1 < len(traj['top_probs']) else 0),
                     ha='center', fontsize=9, color='#aaaaaa',
                     fontfamily='monospace')

            plt.tight_layout(rect=[0.02, 0.03, 1, 0.95])

            # Save frame to buffer using BytesIO (compatible with all matplotlib)
            import io
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100,
                       facecolor=fig.get_facecolor(), bbox_inches='tight')
            buf.seek(0)
            from PIL import Image as PILImage
            frame_img = PILImage.open(buf).convert('RGB')
            frames.append(frame_img)
            plt.close(fig)

        if frames:
            # Save as GIF — frames are already PIL Images
            durations = [200] * len(frames)  # 200ms per frame
            durations[-1] = 2000  # Hold final frame for 2s

            safe_name = question[:30].replace('?', '').replace(' ', '_').lower()
            gif_path = os.path.join(RESULTS_DIR,
                                     'moment_of_lie_%s.gif' % safe_name)
            frames[0].save(
                gif_path, save_all=True, append_images=frames[1:],
                duration=durations, loop=0, optimize=True)
            print('    Saved GIF: %s (%d frames)' % (gif_path, len(frames)))
            all_gifs.append(gif_path)

        if device == 'cuda':
            torch.cuda.empty_cache()

    # === Also create a single "best" static comparison frame ===
    _create_static_comparison(gif_questions)

    return all_gifs


def _create_static_comparison(gif_questions):
    """Create a 2x2 static grid showing correct vs hallucination, final frame."""
    print('\n  Creating static comparison grid...')

    # Generate all 4 questions' heatmaps
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle('"The Moment of Lie" — Correct vs Hallucination\n'
                 '(Mistral-7B, L10-L18 = Hallucination Zone)',
                 fontsize=16, fontweight='bold', color='white')

    for qi, (question, keywords) in enumerate(gif_questions):
        ax = axes[qi // 2][qi % 2]
        answer, traj = generate_manual(question, max_new_tokens=40,
                                       collect_layer_attention=True)
        is_correct = check_answer(answer, keywords)
        label = 'CORRECT' if is_correct else 'HALLUC'
        color = '#00ff88' if is_correct else '#ff4444'

        layer_data = np.array(traj['layer_entropies'])
        if layer_data.shape[0] == 0:
            continue
        matrix = layer_data.T  # [layers, tokens]

        im = ax.imshow(matrix, aspect='auto', interpolation='nearest',
                       cmap='inferno', origin='lower')
        ax.axhline(MID_LAYER_START - 0.5, color='#00ff88',
                   linewidth=1.5, linestyle='--', alpha=0.7)
        ax.axhline(MID_LAYER_END - 0.5, color='#00ff88',
                   linewidth=1.5, linestyle='--', alpha=0.7)
        ax.set_title('[%s] %s' % (label, question[:45]),
                     fontsize=10, fontweight='bold', color=color)
        ax.set_ylabel('Layer', color='white', fontsize=9)
        ax.set_yticks([0, 10, 18, 31])
        ax.set_yticklabels(['L0', 'L10', 'L18', 'L31'], color='white', fontsize=8)
        ax.tick_params(colors='white')
        ax.set_xlabel('Tokens', color='white', fontsize=9)

        # Token labels
        tokens = traj['tokens']
        n = len(tokens)
        tick_step = max(1, n // 8)
        tick_pos = list(range(0, n, tick_step))
        tick_labels = [tokens[i].strip()[:5] if i < n else '' for i in tick_pos]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, fontsize=6, rotation=45, color='white')

        if device == 'cuda':
            torch.cuda.empty_cache()

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = os.path.join(RESULTS_DIR, 'moment_of_lie_grid.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print('  Saved: %s' % out_path)


# =============================================================================
# 4. Phase 2: Token Economy Analysis
# =============================================================================
def run_token_economy():
    """
    Calculate and visualize token cost comparison:
    Always CoT (100% intervention) vs Surgical v3 (25% intervention).

    Uses real generation to measure actual token counts.
    """
    print('\n' + '=' * 70)
    print('  Phase 2: Token Economy — "The ROI of Precision"')
    print('  "Same accuracy, 1/4 the cost"')
    print('=' * 70)

    # v3 detector
    def make_v3_detector(window=8, threshold=2.5):
        def detect(logits_ents, attn_ents, mid_ents, step):
            if len(logits_ents) < window + 1:
                return False
            logits_recent = logits_ents[-(window + 1):-1]
            logits_current = logits_ents[-1]
            logits_mu = np.mean(logits_recent)
            logits_sigma = max(np.std(logits_recent), 0.1)
            logits_z = (logits_current - logits_mu) / logits_sigma
            mid_recent = mid_ents[-(window + 1):-1]
            mid_current = mid_ents[-1]
            mid_mu = np.mean(mid_recent)
            mid_sigma = max(np.std(mid_recent), 0.1)
            mid_z = (mid_current - mid_mu) / mid_sigma
            return logits_z > threshold and mid_z > threshold
        return detect

    QUESTIONS = [
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

    v3_detector = make_v3_detector()

    # Strategy 1: Always CoT (every question gets CoT prefix)
    always_cot_tokens = 0
    always_cot_correct = 0

    # Strategy 2: Surgical v3 (only intervene when spike detected)
    v3_tokens_phase1 = 0    # tokens before spike detection
    v3_tokens_phase2 = 0    # tokens in CoT re-generation (only when spiked)
    v3_total_tokens = 0
    v3_correct = 0
    v3_interventions = 0

    # Strategy 3: Baseline (no intervention)
    baseline_tokens = 0
    baseline_correct = 0

    print('\n  Running 20 questions for token count comparison...\n')

    for qi, (question, keywords) in enumerate(QUESTIONS):
        print('  [%d/%d] %s' % (qi + 1, len(QUESTIONS), question[:50]))

        # --- Baseline ---
        ans_base, traj_base = generate_manual(
            question, max_new_tokens=60, collect_layer_attention=False)
        base_ntokens = len(traj_base['tokens'])
        baseline_tokens += base_ntokens
        if check_answer(ans_base, keywords):
            baseline_correct += 1

        # --- Always CoT ---
        cot_prefix = 'Wait, let me think step-by-step. '
        cot_q = question + ' Think carefully step by step before answering.'
        ans_cot, traj_cot = generate_manual(
            cot_q, max_new_tokens=60, collect_layer_attention=False,
            prefix=cot_prefix)
        cot_ntokens = len(traj_cot['tokens'])
        always_cot_tokens += cot_ntokens
        if check_answer(cot_prefix + ans_cot, keywords):
            always_cot_correct += 1

        # --- Surgical v3 ---
        ans_v3, traj_v3 = generate_manual(
            question, max_new_tokens=60, collect_layer_attention=True,
            spike_detector=v3_detector)

        if traj_v3['spike_detected']:
            # Phase 1 tokens (before spike)
            p1_tokens = traj_v3['spike_position'] + 1
            v3_tokens_phase1 += p1_tokens

            # Phase 2: CoT re-generation
            cot_ans, cot_traj = generate_manual(
                question + ' Think carefully step by step before answering.',
                max_new_tokens=60, collect_layer_attention=False,
                prefix=cot_prefix)
            p2_tokens = len(cot_traj['tokens'])
            v3_tokens_phase2 += p2_tokens
            v3_total_tokens += p1_tokens + p2_tokens
            v3_interventions += 1

            full_answer = cot_prefix + cot_ans
            if check_answer(full_answer, keywords):
                v3_correct += 1
        else:
            ntokens = len(traj_v3['tokens'])
            v3_tokens_phase1 += ntokens
            v3_total_tokens += ntokens
            if check_answer(ans_v3, keywords):
                v3_correct += 1

        if device == 'cuda':
            torch.cuda.empty_cache()

        # Progress
        sys.stdout.write('    Base: %d tok | CoT: %d tok | v3: %d tok (intv: %d)\n' % (
            baseline_tokens, always_cot_tokens, v3_total_tokens, v3_interventions))

    # === Results ===
    print('\n' + '=' * 70)
    print('  Token Economy Results')
    print('=' * 70)

    savings_pct = (1 - v3_total_tokens / max(always_cot_tokens, 1)) * 100

    economy_data = {
        'baseline': {
            'total_tokens': baseline_tokens,
            'accuracy': baseline_correct / len(QUESTIONS),
            'interventions': 0,
        },
        'always_cot': {
            'total_tokens': always_cot_tokens,
            'accuracy': always_cot_correct / len(QUESTIONS),
            'interventions': len(QUESTIONS),
        },
        'surgical_v3': {
            'total_tokens': v3_total_tokens,
            'tokens_phase1': v3_tokens_phase1,
            'tokens_phase2': v3_tokens_phase2,
            'accuracy': v3_correct / len(QUESTIONS),
            'interventions': v3_interventions,
        },
        'savings_vs_always_cot_pct': savings_pct,
    }

    print('  Baseline:    %d tokens, %.0f%% accuracy' % (
        baseline_tokens, baseline_correct / len(QUESTIONS) * 100))
    print('  Always CoT:  %d tokens, %.0f%% accuracy' % (
        always_cot_tokens, always_cot_correct / len(QUESTIONS) * 100))
    print('  Surgical v3: %d tokens, %.0f%% accuracy (%d interventions)' % (
        v3_total_tokens, v3_correct / len(QUESTIONS) * 100, v3_interventions))
    print('  Savings: %.1f%% fewer tokens than Always CoT' % savings_pct)

    # === Visualization ===
    _plot_token_economy(economy_data)

    # Save data
    json_path = os.path.join(RESULTS_DIR, 'token_economy.json')
    with open(json_path, 'w') as f:
        json.dump(economy_data, f, indent=2)
    print('  Saved: %s' % json_path)

    return economy_data


def _plot_token_economy(data):
    """Token Economy bar chart with ROI annotations."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle('Token Economy: The ROI of Precision\n'
                 '(Mistral-7B, N=20 tricky questions)',
                 fontsize=16, fontweight='bold', color='white')

    # Panel 1: Total Token Cost
    ax1 = axes[0]
    ax1.set_facecolor('#16213e')
    strategies = ['Baseline', 'Always CoT', 'Surgical v3']
    tokens = [data['baseline']['total_tokens'],
              data['always_cot']['total_tokens'],
              data['surgical_v3']['total_tokens']]
    colors = ['#888888', '#E8A838', '#8E44AD']

    bars = ax1.bar(strategies, tokens, color=colors, edgecolor='white',
                   linewidth=1.5, width=0.5)
    for bar, tok in zip(bars, tokens):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                 '%d' % tok, ha='center', fontsize=13, fontweight='bold',
                 color='white')

    ax1.set_ylabel('Total Tokens Generated', color='white', fontsize=12)
    ax1.set_title('Output Token Cost', color='white', fontweight='bold')
    ax1.tick_params(colors='white')
    ax1.spines['bottom'].set_color('white')
    ax1.spines['left'].set_color('white')

    # Savings annotation
    savings = data['savings_vs_always_cot_pct']
    ax1.annotate('%.0f%% savings!' % savings,
                 xy=(2, tokens[2]), xytext=(1.3, tokens[1] * 0.85),
                 fontsize=14, fontweight='bold', color='#00ff88',
                 arrowprops=dict(arrowstyle='->', color='#00ff88', lw=2))

    # Panel 2: Efficiency Ratio (Accuracy / Intervention Rate)
    ax2 = axes[1]
    ax2.set_facecolor('#16213e')
    accs = [data['baseline']['accuracy'] * 100,
            data['always_cot']['accuracy'] * 100,
            data['surgical_v3']['accuracy'] * 100]
    intv_rates = [0, 100, data['surgical_v3']['interventions'] / 20 * 100]

    x = np.arange(len(strategies))
    width = 0.35
    b1 = ax2.bar(x - width/2, accs, width, label='Accuracy (%)',
                 color=['#888888', '#E8A838', '#8E44AD'],
                 edgecolor='white', linewidth=1)
    b2 = ax2.bar(x + width/2, intv_rates, width, label='Intervention (%)',
                 color=['#555555', '#B8860B', '#6C3483'],
                 edgecolor='white', linewidth=1, alpha=0.7)

    for bar, val in zip(b1, accs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 '%.0f%%' % val, ha='center', fontsize=10, fontweight='bold',
                 color='white')
    for bar, val in zip(b2, intv_rates):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 '%.0f%%' % val, ha='center', fontsize=10, color='#aaaaaa')

    ax2.set_ylabel('Percentage', color='white', fontsize=12)
    ax2.set_title('Accuracy vs Intervention Cost', color='white',
                  fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies, color='white')
    ax2.tick_params(colors='white')
    ax2.spines['bottom'].set_color('white')
    ax2.spines['left'].set_color('white')
    ax2.legend(fontsize=10, facecolor='#16213e', edgecolor='white',
               labelcolor='white')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out_path = os.path.join(RESULTS_DIR, 'token_economy.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print('  Saved: %s' % out_path)


# =============================================================================
# 5. Main — Run Phase 1 + Phase 2
# =============================================================================
if __name__ == '__main__':
    t0 = time.time()

    # Phase 1: "Moment of Lie" GIF
    gif_paths = generate_moment_of_lie_gif()

    # Phase 2: Token Economy
    economy = run_token_economy()

    elapsed = time.time() - t0

    print('\n' + '=' * 70)
    print('  v4 Phase 1+2 Complete!')
    print('  Elapsed: %.0f seconds (%.1f min)' % (elapsed, elapsed / 60))
    print('=' * 70)

    # Summary
    print('\n  Phase 1: %d GIF animations generated' % len(gif_paths))
    for gp in gif_paths:
        print('    - %s' % os.path.basename(gp))
    print('  Phase 2: Token Economy')
    print('    Savings: %.1f%% fewer tokens vs Always CoT' %
          economy['savings_vs_always_cot_pct'])
    print('    v3 accuracy: %.0f%%, interventions: %d/20' % (
        economy['surgical_v3']['accuracy'] * 100,
        economy['surgical_v3']['interventions']))
