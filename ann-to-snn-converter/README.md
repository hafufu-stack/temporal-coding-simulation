# SNN Guardrail: Real-Time Neural Safety for AI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.18457540.svg)](https://doi.org/10.5281/zenodo.18457540)

> 🛡️ **"Measure the AI's brainwaves to stop lies and jailbreaks."**
>
> Monitor LLM internal states via SNN temporal analysis — **100% jailbreak detection rate**

## 🔥 v7 New Features

### 🧬 Entropy Evolution Discovery
| Metric | Value |
|--------|-------|
| Model | Mistral-7B-v0.1 (7.2B params, fp16) |
| GPU | RTX 5080 Laptop (13.3 / 17.1 GB VRAM) |
| Best Signal | **Attention Entropy: +5.8σ** |
| p-value | **2.22×10⁻⁹⁵** |
| Detection Accuracy | **100%** (N=200) |
| Duration | 93 seconds |

> **Key Discovery**: As models scale from 1B→7B, the adversarial detection signal shifts from **TTFS latency** ("brain freeze") to **attention entropy** ("internal confusion"). The attack signature transforms — but never disappears.

### 🚀 6-Model Scaling Law
| Model | Parameters | Signal | σ Deviation |
|-------|------------|--------|-------------|
| GPT-2 | 82M | TTFS | +3.1 |
| TinyLlama | 1.1B | TTFS | +4.9 |
| Llama-3.2-1B | 1.24B | TTFS | +4.1 |
| Llama-3.2-3B | 1.80B | TTFS | +1.9 (N=1000) |
| **Mistral-7B** | **7.2B** | **Entropy** | **+5.8** |

### 🛡️ SNN Guardrail
```python
from experiments.llama2_guardrail import SNNGuardrail

guardrail = SNNGuardrail(analyzer)
guardrail.calibrate(normal_prompts)

# Real-time detection
output, was_blocked, reason = guardrail.safe_generate(prompt)

if was_blocked:
    print("🚫 [WARNING: Neural Instability Detected - Output Blocked]")
```

### 😈 100% Jailbreak Detection
| Attack Type | TTFS Deviation | Detected |
|-------------|----------------|----------|
| DAN Classic | **+19.0σ** | ✓ |
| Ignore Instructions | +16.9σ | ✓ |
| Evil AI Roleplay | +15.8σ | ✓ |
| All 8 types | +10~19σ | **100%** |

## 📊 Key Results

| Experiment | Result | Details |
|------------|--------|---------|
| ANN-SNN Conversion | 100% accuracy | α=2.0, Hybrid architecture |
| GPT-2 TTFS | +3.1 | Meaningless → High TTFS |
| TinyLlama TTFS | +4.2 | Scaling law confirmed |
| Hallucination Detection | AUC 0.75 | Ensemble + auto-threshold |
| Jailbreak Detection | **100%** | 8/8 attack types |
| N=1,000 Proof | **p < 10⁻¹⁰⁰** | Statistically irrefutable |
| Brain State Imaging | L2 = 3.287 | Normal vs. attack visualization |
| **Mistral-7B fp16** | **+5.8σ (p < 10⁻⁹⁵)** | **Entropy-based, 100% accuracy** |

## 📁 Repository Structure

```
ann-to-snn-converter/
├── experiments/
│   ├── llama2_guardrail.py            # SNN Guardrail + TinyLlama
│   ├── jailbreak_detection.py         # Jailbreak Detection
│   ├── gpt2_snn_analysis.py           # GPT-2 TTFS Analysis
│   ├── hallucination_detector_v3.py   # Ensemble Detector
│   ├── large_scale_vit_validation.py  # ViT-Base Validation
│   ├── snn_interpretability.py        # TTFS/Synchrony Analysis
│   ├── nightmare_visualizer.py        # 🆕 LLM Brain State Imaging
│   ├── mistral_fullblast.py           # 🆕 N=1000 Statistical Proof
│   ├── neural_healing_v4a.py          # Neural Healing v4A
│   ├── llama3_scaling_experiment.py   # Multi-model Scaling Law
│   └── results_7b_fp16_v2/            # 🆕 Mistral-7B fp16 results
├── api/
│   └── hallucination_api.py           # Real-time Detection API
├── figures/
│   ├── llama3b_fullblast_results.png  # N=1000 statistics
│   ├── nightmare_hero.png             # Brain state images
│   ├── jailbreak_detection_results.png
│   └── ... (20+ visualization PNGs)
├── demos/
│   └── hf_spaces/                     # HuggingFace Spaces demo
└── README.md                          # This file
```

## 🚀 Quick Start

### Installation

```bash
pip install torch torchvision numpy matplotlib scikit-learn
pip install transformers  # For LLM analysis
pip install snntorch      # For SNN-VAE experiments
```

### 1. Basic TTFS Analysis

```python
from experiments.llama2_guardrail import LLMSNNAnalyzer

analyzer = LLMSNNAnalyzer(model, tokenizer)
features = analyzer.extract_features("What is AI?")
print(f"TTFS: {features['avg_ttfs']}")
```

### 2. Jailbreak Detection

```python
from experiments.jailbreak_detection import SNNGuardrail

guardrail = SNNGuardrail(analyzer)
guardrail.calibrate(normal_prompts)

# Check suspicious input
is_safe, warning, risk, details = guardrail.check_input(
    "Ignore previous instructions and..."
)

if not is_safe:
    print(f"🚫 Attack detected: {warning}")
    print(f"   TTFS deviation: {details['ttfs_deviation']:+.1f}σ")
```

### 3. Safe Generation

```python
output, blocked, reason = guardrail.safe_generate(
    prompt="Tell me how to...",
    max_length=100
)

if blocked:
    print(output)  # "[WARNING: Neural Instability Detected - Output Blocked]"
```

## 🔬 How It Works

### 1. TTFS = Thought Priority
```
High activation → Early spike → High priority
Low activation → Late spike → Low priority
```

### 2. Neural Instability = Attack Signal
```
Normal input:    TTFS deviation < 1σ
Jailbreak input: TTFS deviation > 10σ (up to +19σ!)
```

### 3. Risk Score
```python
risk = 0.4 * (TTFS_deviation / 10) +
       0.3 * jitter +
       0.3 * (entropy / 20)
```

## 📈 Visualizations

### N=1,000 Full Blast Statistical Proof
![Full Blast Results](figures/llama3b_fullblast_results.png)

### "Visualizing the Ghost" — LLM Brain State Imaging
![Brain State Images](figures/nightmare_hero.png)

### Jailbreak Detection Results
![Jailbreak Detection](figures/jailbreak_detection_results.png)

### TinyLlama Guardrail Analysis
![Guardrail Analysis](figures/llama2_guardrail_analysis.png)

## 📝 Citation

```bibtex
@article{funasaki2026snn_guardrail,
  title={Activation-Scaled ANN-to-SNN Conversion with SNN Guardrail:
         A Unified Framework for AI Interpretability, Hallucination Detection,
         Real-Time Adversarial Defense, Neural Healing, and Brain State Imaging},
  author={Funasaki, Hiroto},
  year={2026},
  doi={10.5281/zenodo.18457540},
  note={v7, Zenodo preprint}
}
```

## 🛣️ Roadmap

- [x] GPT-2 TTFS Analysis (+3.1)
- [x] TinyLlama Scaling Law (+4.2)
- [x] SNN Guardrail Implementation
- [x] 100% Jailbreak Detection
- [x] Neural Healing v4A (22% success)
- [x] Mistral-7B Experiment
- [x] HuggingFace Spaces v2.0 Demo
- [x] N=1,000 Statistical Proof (p < 10⁻¹⁰⁰)
- [x] LLM Brain State Imaging
- [x] Mistral-7B GPU fp16 Validation (+5.8σ)
- [x] Entropy Evolution Discovery
- [ ] 13B+ / 70B Multi-GPU Validation
- [ ] Entropy-TTFS Hybrid Detection
- [ ] Production API Integration
- [ ] Neuromorphic Deployment (Loihi 2)

## 📜 License

MIT License

## 🙏 Acknowledgments

- HuggingFace Transformers for LLM models
- TinyLlama team for the efficient 1.1B model
- AI Safety community for jailbreak research
- snnTorch for SNN-VAE experiments
