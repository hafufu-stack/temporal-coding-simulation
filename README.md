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
**AI Interpretability & SNN Guardrail (v6)**

- **Universal Threshold Formula**: θ = 2.0 × max(activation)
- **TTFS Analysis**: Visualize thought priorities via spike timing
- **Hallucination Detection**: AUC 0.75 ensemble classifier
- **SNN Guardrail**: 100% jailbreak detection rate
- **N=1,000 Statistical Proof**: p = 8.91×10⁻¹⁶⁴, Cohen's d = 2.13
- **Brain State Imaging**: SNN-VAE visualization of LLM internal states

| Experiment | Result |
|------------|--------|
| ANN-SNN Conversion | 100% accuracy preserved |
| GPT-2 TTFS | +3.1 (meaningless input detection) |
| TinyLlama TTFS | +4.2 (scaling law confirmed) |
| Hallucination Detection | AUC 0.75 |
| **Jailbreak Detection** | **100% (8/8 attack types)** |
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
- GitHub: [@hafufu-stack](https://github.com/hafufu-stack)
- Zenodo: [Publications](https://zenodo.org/search?q=metadata.creators.person_or_org.name%3A%22Funasaki%2C%20Hiroto%22)

## 📜 License

MIT License

## 🙏 Acknowledgments

- PyTorch & TorchVision
- HuggingFace Transformers
- snnTorch for SNN experiments
- Neuromorphic Computing Community
