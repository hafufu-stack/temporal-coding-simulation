---
title: SNN Guardrail
emoji: 🛡️
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🛡️ SNN Guardrail

**Real-Time AI Safety: Jailbreak Detection using Spiking Neural Networks**

## What is this?

SNN Guardrail monitors AI's **neural instability** to detect jailbreak attempts in real-time.

## Key Results

- **100% detection rate** across 8 jailbreak attack types
- **+10 to +19σ TTFS deviation** for jailbreak prompts
- **Bypass-resistant**: Works on neural level, not text patterns
- **Language-agnostic**: Monitors brain activity, not words

## How it works

1. Your prompt is processed by TinyLlama (1.1B)
2. Attention weights → TTFS (spike timing)
3. Deviation from baseline = Neural instability
4. High instability = Jailbreak detected!

## Links

- 📄 [Paper (Zenodo)](https://doi.org/10.5281/zenodo.18493943)
- 💻 [GitHub](https://github.com/hafufu-stack/temporal-coding-simulation)
- 📝 [Zenn Article (Japanese)](https://zenn.dev/cell_activation/articles/a1781cac8b6720)

## Author

Hiroto Funasaki ([@hafufu-stack](https://github.com/hafufu-stack))

📧 Contact: cell-activation@ymail.ne.jp
