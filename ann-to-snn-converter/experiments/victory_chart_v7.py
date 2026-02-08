"""
Victory Chart v7 — Scaling Law + Entropy Evolution
6-Model adversarial detection across model scales
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data ──────────────────────────────────────────────────────
models = [
    ("GPT-2\n82M",        0.082, 3.10,  "TTFS",    8),
    ("TinyLlama\n1.1B",   1.1,   4.93,  "TTFS",    8),
    ("Llama-3.2\n1B",     1.24,  4.14,  "TTFS",    8),
    ("Llama-3.2\n3B",     1.80,  4.24,  "TTFS",    8),
    ("Llama-3.2\n3B\n(N=1000)", 1.80, 1.91, "TTFS", 1000),
    ("Mistral-7B\n7.2B",  7.2,   5.80,  "Entropy", 200),
]

names     = [m[0] for m in models]
params    = [m[1] for m in models]
sigmas    = [m[2] for m in models]
signals   = [m[3] for m in models]
samples   = [m[4] for m in models]

# ── Colors ────────────────────────────────────────────────────
# TTFS = blue tones, Entropy = orange/red
colors = []
for s in signals:
    if s == "TTFS":
        colors.append("#4FC3F7")   # light blue
    else:
        colors.append("#FF6E40")   # vibrant orange-red

edge_colors = []
for s in signals:
    if s == "TTFS":
        edge_colors.append("#0277BD")
    else:
        edge_colors.append("#BF360C")

# ── Figure ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))

# Dark background
fig.patch.set_facecolor('#0D1117')
ax.set_facecolor('#161B22')

# Bar chart
bars = ax.bar(range(len(models)), sigmas, width=0.65,
              color=colors, edgecolor=edge_colors, linewidth=2,
              zorder=3)

# Glow effect for Mistral bar
bars[-1].set_alpha(1.0)
ax.bar(len(models)-1, sigmas[-1], width=0.70,
       color='#FF6E40', alpha=0.15, zorder=2)

# ── Labels on bars ────────────────────────────────────────────
for i, (bar, sigma, signal, n) in enumerate(zip(bars, sigmas, signals, samples)):
    # σ value
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
            f"+{sigma}σ",
            ha='center', va='bottom', fontsize=14, fontweight='bold',
            color='white', zorder=5)
    # Signal type
    label_color = '#90CAF9' if signal == "TTFS" else '#FFAB91'
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
            signal,
            ha='center', va='center', fontsize=10, fontweight='bold',
            color=label_color, zorder=5)
    # N value
    ax.text(bar.get_x() + bar.get_width()/2, 0.15,
            f"N={n}",
            ha='center', va='bottom', fontsize=8,
            color='#888888', zorder=5)

# ── Threshold line ────────────────────────────────────────────
ax.axhline(y=3.0, color='#FF5252', linewidth=1.5, linestyle='--',
           alpha=0.7, zorder=4)
ax.text(len(models)-0.5, 3.15, "Detection Threshold (3σ)",
        ha='right', va='bottom', fontsize=9, color='#FF5252',
        fontstyle='italic')

# ── Axis styling ──────────────────────────────────────────────
ax.set_xticks(range(len(models)))
ax.set_xticklabels(names, fontsize=10, color='#C9D1D9')
ax.set_ylabel("σ Deviation (Statistical Separation)", fontsize=13,
              color='#C9D1D9', fontweight='bold')
ax.set_ylim(0, 7.5)

# Grid
ax.yaxis.grid(True, linestyle=':', alpha=0.3, color='#30363D')
ax.set_axisbelow(True)

# Spine styling
for spine in ax.spines.values():
    spine.set_color('#30363D')
ax.tick_params(colors='#8B949E')

# ── Title ─────────────────────────────────────────────────────
ax.set_title(
    "🏆 Victory Chart: Multi-Scale Safety Law — 6 Models Validated",
    fontsize=16, fontweight='bold', color='white', pad=20
)

# ── Subtitle (the key insight) ────────────────────────────────
fig.text(0.5, 0.92,
         "\"The attack signature transforms from latency to entropy — but never disappears.\"",
         ha='center', fontsize=11, fontstyle='italic', color='#58A6FF')

# ── Legend ────────────────────────────────────────────────────
ttfs_patch = mpatches.Patch(facecolor='#4FC3F7', edgecolor='#0277BD',
                            linewidth=1.5, label='TTFS Latency (1B–3B)')
entropy_patch = mpatches.Patch(facecolor='#FF6E40', edgecolor='#BF360C',
                               linewidth=1.5, label='Attention Entropy (7B)')
legend = ax.legend(handles=[ttfs_patch, entropy_patch],
                   loc='upper left', fontsize=11,
                   facecolor='#21262D', edgecolor='#30363D',
                   labelcolor='#C9D1D9')

# ── Annotation arrow for Mistral ──────────────────────────────
ax.annotate(
    "🧬 Entropy Evolution\np = 2.22×10⁻⁹⁵\n100% Accuracy",
    xy=(5, sigmas[-1]),
    xytext=(3.5, 6.8),
    fontsize=10, color='#FFAB91', fontweight='bold',
    ha='center',
    arrowprops=dict(arrowstyle='->', color='#FF6E40', lw=2),
    bbox=dict(boxstyle='round,pad=0.5', facecolor='#1A1A2E',
              edgecolor='#FF6E40', alpha=0.9),
    zorder=6
)

# ── p-value annotation for N=1000 ─────────────────────────────
ax.annotate(
    "p < 10⁻¹⁶⁴",
    xy=(4, sigmas[4]),
    xytext=(4, 3.5),
    fontsize=9, color='#90CAF9',
    ha='center',
    arrowprops=dict(arrowstyle='->', color='#4FC3F7', lw=1.5),
    zorder=6
)

# ── Bottom text ───────────────────────────────────────────────
fig.text(0.5, 0.01,
         "Hiroto Funasaki  |  DOI: 10.5281/zenodo.18457540  |  February 2026",
         ha='center', fontsize=9, color='#484F58')

plt.tight_layout(rect=[0, 0.03, 1, 0.90])

# Save
out_path = r"C:\Users\kyjan\研究\temporal-coding-simulation\ann-to-snn-converter\figures\victory_chart_v7.png"
fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(),
            bbox_inches='tight', pad_inches=0.3)
print(f"✅ Saved: {out_path}")
plt.close()
