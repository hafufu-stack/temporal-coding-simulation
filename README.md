# Temporal Coding Simulation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> 🧠 Research repository for Spiking Neural Network (SNN) experiments — temporal coding, AI safety, compression, and cryptography

## 📁 Repository Structure

```
temporal-coding-simulation/
├── ann-to-snn-converter/      # 🛡️ ANN→SNN Conversion + AI Safety + SNN Guardrail
│   ├── experiments/           # Experiment scripts (TTFS, hallucination, jailbreak, brain imaging)
│   ├── figures/               # Result visualizations (20+ PNGs)
│   ├── api/                   # Real-time detection API
│   ├── demos/                 # HuggingFace Spaces demo
│   └── README.md              # Detailed documentation
├── snn-comprypto/             # 🔐 SNN-based cryptography (chaotic reservoir)
├── snn-compression/           # 📦 SNN compression (correlation coding)
├── snn-genai/                 # 🎨 SNN image generation (Spiking VAE)
├── snn-operation/             # ➕ SNN arithmetic operations
├── 10-neuron-memory/          # 💾 10-neuron memory experiment
└── assets/                    # 📊 Shared figures and images
```

## 🔥 Featured Projects

### 1. [ANN-to-SNN Converter](./ann-to-snn-converter/) 🛡️
**AI Interpretability & SNN Guardrail (v11 — The Migration Map Edition)**

- **SNN Guardrail**: 100% jailbreak detection rate (8/8 types)
- **AI Immune System**: Sense→Alert→Heal→Learn (97.9% healing)
- **Safety Vaccination**: +18% immunity via QLoRA SFT (Project Morpheus)
- **Cross-Species Vaccination**: Mistral→Llama transfer (Project Chimera)
- **Migration Map**: Non-monotonic scaling — Novice→Thinker→Expert
- **14B Scaling**: Canary migrates to 12.5% depth (Project Titan)

| Experiment | Result |
|------------|--------|
| **Jailbreak Detection** | **100% (8/8 attack types)** |
| **AI Immune System** | **Sense→Alert→Heal→Learn** |
| **Safety Vaccination** | **+18% immunity, -6% tax** |
| **Cross-Species** | **+4% (22% transfer efficiency)** |
| **Migration Map** | **5-model non-monotonic scaling** |
| **N=1,000 Proof** | **p < 10⁻¹⁰⁰** |

### 2. [SNN-Comprypto](./snn-comprypto/)
**SNN Cryptography (Simultaneous Encryption)**

- Chaotic reservoir dynamics
- High-security spike-based encryption
- Spike-based authentication

### 3. [SNN-Compression](./snn-compression/)
**SNN Compression (Correlation Coding)**

- Differential coding + zlib
- 91% improvement for binary data
- Correlation encoding (12.4 bits/neuron)

### 4. [SNN-GenAI](./snn-genai/)
**Image Generation with SNNs**

- Spiking VAE with Posterior Collapse fix
- 70/30 Hybrid Readout (spike + membrane)
- Energy-efficient image generation

### 5. [SNN-Operation](./snn-operation/)
**Neural Arithmetic**

- 30-neuron adder
- Spike-based arithmetic operations

## 📊 Key Results

| Project | Key Result |
|---------|------------|
| ANN-SNN Conversion | α = 2.0 preserves 100% accuracy |
| Hallucination Detection | AUC 0.75 (Ensemble + Auto-Threshold) |
| **Jailbreak Detection** | **100% (TTFS +10~19σ deviation)** |
| **N=1,000 Proof** | **p < 10⁻¹⁰⁰, Cohen's d = 2.13** |
| SNN-Comprypto | Chaotic encryption system |
| SNN-Compression | 91% compression improvement (binary) |
| Spiking VAE | 96% spike rate, 30% membrane contribution |

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/hafufu-stack/temporal-coding-simulation.git
cd temporal-coding-simulation

# ANN-SNN Conversion (Hallucination Detection)
cd ann-to-snn-converter
python experiments/hallucination_detector_v3.py

# Jailbreak Detection (SNN Guardrail)
python experiments/jailbreak_detection.py

# SNN Cryptography
cd ../snn-comprypto
python snn_comprypto.py

# SNN Compression
cd ../snn-compression
python correlation_compressor.py
```

## 🤝 Author

**Hiroto Funasaki**
- ORCID: 0009-0004-2517-0177
- Email: cell-activation@ymail.ne.jp
Hiroto Funasaki
- GitHub: [@hafufu-stack](https://github.com/hafufu-stack)
- Zenodo: [DOI: 10.5281/zenodo.18457540](https://doi.org/10.5281/zenodo.18457540)

## 📜 License

MIT License

## 🙏 Acknowledgments

- PyTorch & TorchVision
- HuggingFace Transformers
- snnTorch for SNN experiments
- Neuromorphic Computing Community
