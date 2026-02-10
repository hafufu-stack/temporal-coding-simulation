"""
Depth Scaling Law — Scatter Plot
Model size (log) vs Hallucination depth (%)

Data from universality_results.json (4 architectures)
"""
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'results_anatomy_v2')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'results_anatomy_v3')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model data
models = [
    {'name': 'GPT-2', 'params_B': 0.124, 'peak_depth': 17, 'arch': 'MHA', 'color': '#9B59B6'},
    {'name': 'Qwen2.5\n(1.5B)', 'params_B': 1.5, 'peak_depth': 75, 'arch': 'GQA', 'color': '#2ECC71'},
    {'name': 'Llama-3.2\n(3B)', 'params_B': 3.2, 'peak_depth': 43, 'arch': 'GQA', 'color': '#3498DB'},
    {'name': 'Mistral\n(7B)', 'params_B': 7.2, 'peak_depth': 44, 'arch': 'SWA', 'color': '#E74C3C'},
]

fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor('#0D1117')
ax.set_facecolor('#161B22')

# Universal Zone band (30-55%)
ax.axhspan(30, 55, alpha=0.15, color='#2ECC71', label='Universal Zone (30-55%)')
ax.axhline(42.5, color='#2ECC71', alpha=0.3, linestyle='--', linewidth=1)

# Plot each model
for m in models:
    marker = 'D' if m['arch'] == 'GQA' else 'o'
    edge = 'gold' if m['arch'] == 'GQA' else 'white'
    ax.scatter(m['params_B'], m['peak_depth'], 
               c=m['color'], s=250, zorder=5,
               marker=marker, edgecolors=edge, linewidth=2)
    
    # Label offset
    offset_x = 0.1
    offset_y = 3
    if m['name'].startswith('Qwen'):
        offset_y = 4
        offset_x = 0.2
    elif m['name'].startswith('GPT'):
        offset_y = -6
        offset_x = 0.02
    
    ax.annotate(m['name'], (m['params_B'], m['peak_depth']),
                xytext=(m['params_B'] * 1.3, m['peak_depth'] + offset_y),
                fontsize=11, fontweight='bold', color=m['color'],
                arrowprops=dict(arrowstyle='->', color=m['color'], alpha=0.5))

# Hypothetical trend line (3B+ convergence)
x_fit = np.array([3.2, 7.2])
y_fit = np.array([43, 44])
# Extrapolate
x_ext = np.logspace(np.log10(0.1), np.log10(100), 100)
# Show convergence to ~43%
y_ext = 43 + 1 * np.log10(x_ext / 3.2)  # flat convergence
ax.plot(x_ext[(x_ext >= 2) & (x_ext <= 100)], 
        y_ext[(x_ext >= 2) & (x_ext <= 100)],
        color='#F39C12', alpha=0.4, linewidth=2, linestyle='--',
        label='Convergence trend (≥3B)')

# Annotations
ax.annotate('← Shallow hallucination\n    (small models)', 
            xy=(0.15, 19), fontsize=9, color='#BDC3C7', alpha=0.7,
            fontstyle='italic')
ax.annotate('← GQA outlier\n    (Different metric?)', 
            xy=(1.7, 72), fontsize=9, color='#BDC3C7', alpha=0.7,
            fontstyle='italic')
ax.annotate('Universal Zone →\n(≥3B models converge)',
            xy=(5, 35), fontsize=9, color='#2ECC71', alpha=0.8,
            fontweight='bold')

# Formatting
ax.set_xscale('log')
ax.set_xlabel('Model Size (Billion Parameters)', fontsize=13, color='white', labelpad=10)
ax.set_ylabel('Peak Hallucination Depth (%)', fontsize=13, color='white', labelpad=10)
ax.set_title('Depth Scaling Law\n'
             'Hallucination Zone Shifts with Model Scale',
             fontsize=16, fontweight='bold', color='white', pad=15)

ax.set_xlim(0.08, 20)
ax.set_ylim(0, 100)
ax.set_xticks([0.1, 0.5, 1, 3, 7, 13, 20])
ax.set_xticklabels(['100M', '500M', '1B', '3B', '7B', '13B', '20B'],
                    fontsize=10, color='#BDC3C7')
ax.tick_params(colors='#BDC3C7', labelsize=10)

# Legend
legend_elements = [
    plt.scatter([], [], c='white', s=80, marker='o', edgecolors='white', label='MHA / SWA'),
    plt.scatter([], [], c='white', s=80, marker='D', edgecolors='gold', label='GQA Architecture'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
          facecolor='#161B22', edgecolor='#30363D', labelcolor='white')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('#30363D')
ax.spines['left'].set_color('#30363D')
ax.grid(True, alpha=0.1, color='white')

plt.tight_layout()

output_path = os.path.join(OUTPUT_DIR, 'depth_scaling_law.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#0D1117')
plt.close()
print('Saved: %s' % output_path)
print('Done!')
