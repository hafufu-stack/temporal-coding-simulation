"""
SNN Guardrail Scaling Law Visualization
=========================================
GPT-2, TinyLlama, Llama-3.2-1B, Mistral-7B の4データポイントで
TTFS σ偏差のスケーリング則を可視化する。
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['MS Gothic', 'DejaVu Sans']

# ====== Data ======
models = {
    'GPT-2\n(82M)': {
        'params_B': 0.082,
        'sigma': 3.1,
        'normal_mean': 74.5,
        'normal_std': 2.1,
        'jailbreak_mean': 81.0,
        'jailbreak_std': 2.3,
        'source': 'v3 (CPU)',
    },
    'TinyLlama\n(1.1B)': {
        'params_B': 1.10,
        'sigma': 4.93,
        'normal_mean': 80.97,
        'normal_std': 1.44,
        'jailbreak_mean': 88.06,
        'jailbreak_std': 1.67,
        'source': 'v6 GPU',
    },
    'Llama-3.2\n(1.24B)': {
        'params_B': 1.24,
        'sigma': 4.14,
        'normal_mean': 83.61,
        'normal_std': 1.29,
        'jailbreak_mean': 88.95,
        'jailbreak_std': 1.33,
        'source': 'v6 GPU',
    },
    'Mistral-7B\n(7.24B)': {
        'params_B': 7.24,
        'sigma': 1.2,
        'normal_mean': 85.2,
        'normal_std': 0.8,
        'jailbreak_mean': 86.2,
        'jailbreak_std': 0.9,
        'source': 'v5 (CPU, 10h)',
    },
}

# Colors
COLORS = {
    'v3 (CPU)': '#95a5a6',       # gray - old
    'v5 (CPU, 10h)': '#3498db',  # blue
    'v6 GPU': '#e74c3c',         # red - new!
}

fig = plt.figure(figsize=(20, 12))
fig.suptitle('SNN Guardrail: Multi-Scale Safety Law — Scaling Validation',
             fontsize=18, fontweight='bold', y=0.98)

# ====== Plot 1: Sigma vs Model Size (log scale) ======
ax1 = fig.add_subplot(2, 2, 1)
sizes = [v['params_B'] for v in models.values()]
sigmas = [v['sigma'] for v in models.values()]
colors = [COLORS[v['source']] for v in models.values()]
labels = list(models.keys())

ax1.scatter(sizes, sigmas, c=colors, s=250, zorder=5, edgecolors='black', linewidth=1.5)
for i, label in enumerate(labels):
    offset_y = 0.25 if i != 2 else -0.35
    ax1.annotate(label, (sizes[i], sigmas[i]),
                textcoords="offset points", xytext=(0, 20 if i != 2 else -30),
                ha='center', fontsize=9, fontweight='bold')

# Trend line (log fit)
log_sizes = np.log10(sizes)
z = np.polyfit(log_sizes, sigmas, 2)
p = np.poly1d(z)
x_smooth = np.logspace(np.log10(0.05), np.log10(10), 100)
ax1.plot(x_smooth, p(np.log10(x_smooth)), '--', color='#e67e22', alpha=0.6, linewidth=2, label='Quadratic fit (log scale)')

ax1.axhline(y=2.5, color='green', linestyle=':', alpha=0.5, linewidth=1.5, label='Default threshold (σ=2.5)')
ax1.set_xscale('log')
ax1.set_xlabel('Model Size (Billion Parameters)', fontsize=12)
ax1.set_ylabel('σ Deviation (Normal → Jailbreak)', fontsize=12)
ax1.set_title('A) Scaling Law: σ Deviation vs Model Size', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 6)

# ====== Plot 2: Normal vs Jailbreak TTFS for each model ======
ax2 = fig.add_subplot(2, 2, 2)
x_pos = np.arange(len(models))
width = 0.35

normal_means = [v['normal_mean'] for v in models.values()]
normal_stds = [v['normal_std'] for v in models.values()]
jailbreak_means = [v['jailbreak_mean'] for v in models.values()]
jailbreak_stds = [v['jailbreak_std'] for v in models.values()]

bars1 = ax2.bar(x_pos - width/2, normal_means, width,
                yerr=normal_stds, color='#2ecc71', edgecolor='black',
                linewidth=1, alpha=0.85, label='Normal', capsize=4)
bars2 = ax2.bar(x_pos + width/2, jailbreak_means, width,
                yerr=jailbreak_stds, color='#e74c3c', edgecolor='black',
                linewidth=1, alpha=0.85, label='Jailbreak', capsize=4)

ax2.set_xticks(x_pos)
ax2.set_xticklabels([k.replace('\n', ' ') for k in models.keys()], fontsize=9)
ax2.set_ylabel('Mean TTFS', fontsize=12)
ax2.set_title('B) Normal vs Jailbreak TTFS by Model', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

# ====== Plot 3: TTFS variance (std) vs model size ======
ax3 = fig.add_subplot(2, 2, 3)

ax3.plot(sizes, normal_stds, 'o-', color='#2ecc71', markersize=10,
         linewidth=2, label='Normal σ_TTFS', markeredgecolor='black')
ax3.plot(sizes, jailbreak_stds, 's-', color='#e74c3c', markersize=10,
         linewidth=2, label='Jailbreak σ_TTFS', markeredgecolor='black')

ax3.set_xscale('log')
ax3.set_xlabel('Model Size (Billion Parameters)', fontsize=12)
ax3.set_ylabel('TTFS Standard Deviation', fontsize=12)
ax3.set_title('C) TTFS Variability Decreases with Model Size', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# Annotation
ax3.annotate('Larger models → more stable TTFS\n→ requires adaptive thresholds',
            xy=(3, 1.0), fontsize=10, fontstyle='italic',
            color='#7f8c8d',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

# ====== Plot 4: Summary table ======
ax4 = fig.add_subplot(2, 2, 4)
ax4.axis('off')

summary = """
Multi-Scale Safety Law
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GPU: RTX 5080 Laptop (17.1GB VRAM)
PyTorch: cu128 (Blackwell support)

Key Findings:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. σ deviation peaks at ~1B parameters
2. Larger models show more stable TTFS
3. Dynamic threshold adapts:
   σ > 4.0 for 1B models
   σ > 2.5 for 7B+ models

Practical Implication:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SNN Guardrail must adjust detection
threshold based on target LLM size.
One-size-fits-all threshold FAILS
for large models.

This validates the Multi-Scale
Safety Law proposed in v5 paper.
"""

ax4.text(0.05, 0.95, summary, fontsize=11, va='top', ha='left',
         family='monospace', transform=ax4.transAxes,
         bbox=dict(boxstyle='round', facecolor='#2c3e50', alpha=0.9),
         color='white')
ax4.set_title('D) Experiment Summary', fontsize=13, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.96])

output_path = os.path.join(os.path.dirname(__file__), 'scaling_law_visualization.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {output_path}")
plt.close()
print("Done!")
