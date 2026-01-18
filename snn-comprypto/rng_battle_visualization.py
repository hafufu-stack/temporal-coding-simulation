"""
RNG Battle Royale Visualization
================================

Creates a beautiful chart showing SNN vs DNN vs LSTM prediction rates.

Author: roll
Date: 2026-01-18
"""

import matplotlib.pyplot as plt
import numpy as np

# Enable non-ASCII font support
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']

# Data from the rigorous 100,000-round experiment
generators = ['SNN\n(0.390%)', 'Py-Random\n(0.396%)', 'DNN\n(7.21%)', 'LSTM\n(21.95%)']
prediction_rates = [0.390, 0.396, 7.210, 21.954]
colors = ['#2ecc71', '#27ae60', '#e74c3c', '#c0392b']  # Green for good, red for bad

# Create figure with dark background for modern look
fig, ax = plt.subplots(figsize=(12, 7), facecolor='#1a1a2e')
ax.set_facecolor('#1a1a2e')

# Create bars
bars = ax.bar(generators, prediction_rates, color=colors, edgecolor='white', linewidth=2, width=0.6)

# Add value labels on bars
for bar, rate in zip(bars, prediction_rates):
    height = bar.get_height()
    label = f'{rate:.2f}%' if rate < 1 else f'{rate:.1f}%'
    ax.annotate(label,
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 8),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=14, fontweight='bold', color='white')

# Add theoretical random line
ax.axhline(y=0.391, color='#3498db', linestyle='--', linewidth=2, alpha=0.8, label='Theoretical Random (0.39%)')

# Title and labels
ax.set_title('RNG Battle Royale Results\n(Lower = Better Randomness)', 
             fontsize=20, fontweight='bold', color='white', pad=20)
ax.set_ylabel('Prediction Rate by AI Attackers (%)', fontsize=14, color='white', labelpad=10)
ax.set_xlabel('Random Number Generator', fontsize=14, color='white', labelpad=10)

# Style the axes
ax.tick_params(axis='both', colors='white', labelsize=12)
ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add legend
ax.legend(loc='upper left', fontsize=12, facecolor='#1a1a2e', edgecolor='white', labelcolor='white')

# Add annotations for key findings
ax.annotate('SNN is\nUNPREDICTABLE!', 
            xy=(0, 0.39), xytext=(0.5, 5),
            fontsize=11, color='#2ecc71', fontweight='bold',
            ha='center',
            arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=2))

ax.annotate('LSTM is 56x\nmore predictable!', 
            xy=(3, 21.95), xytext=(2.3, 18),
            fontsize=11, color='#e74c3c', fontweight='bold',
            ha='center',
            arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2))

# Add watermark
ax.text(0.98, 0.02, '@hafufu-stack', transform=ax.transAxes,
        fontsize=10, color='gray', alpha=0.5, ha='right', va='bottom')

plt.tight_layout()

# Save the figure
output_path = 'results/rng_battle_royale_result.png'
plt.savefig(output_path, dpi=150, facecolor='#1a1a2e', edgecolor='none', bbox_inches='tight')
print(f"Saved: {output_path}")

# Also save a light version for different use cases
fig2, ax2 = plt.subplots(figsize=(12, 7))

bars2 = ax2.bar(generators, prediction_rates, color=colors, edgecolor='black', linewidth=1, width=0.6)

for bar, rate in zip(bars2, prediction_rates):
    height = bar.get_height()
    label = f'{rate:.2f}%' if rate < 1 else f'{rate:.1f}%'
    ax2.annotate(label,
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=12, fontweight='bold')

ax2.axhline(y=0.391, color='blue', linestyle='--', linewidth=2, alpha=0.5, label='Theoretical Random (0.39%)')
ax2.set_title('RNG Battle Royale: SNN vs DNN vs LSTM\n(100,000 rounds - Lower = Better)', fontsize=16, fontweight='bold')
ax2.set_ylabel('Prediction Rate (%)', fontsize=12)
ax2.set_xlabel('Generator Type', fontsize=12)
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
output_path2 = 'results/rng_battle_royale_result_light.png'
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"Saved: {output_path2}")

plt.show()
print("\nDone! Two versions created:")
print("  1. Dark theme (for presentations/Twitter)")
print("  2. Light theme (for papers/Zenn)")
