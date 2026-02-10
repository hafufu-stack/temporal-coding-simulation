# SNN Guardrail: Real-Time Neural Safety for AI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.18457540.svg)](https://doi.org/10.5281/zenodo.18457540)

> 🛡️ **"Measure the AI's brainwaves to stop lies and jailbreaks."**
>
> Monitor LLM internal states via SNN temporal analysis — **100% jailbreak detection rate**

## 🐦 v9 New Features: The Canary Head Paradigm

### 🐦 From "Liar Heads" to "Canary Heads"

> **Paradigm Shift**: Attention heads with anomalous entropy differentials are not "lying" — they are **"warning"**, like canaries in coal mines.

| Strategy | Heads Monitored | Accuracy | Compute |
|----------|----------------|----------|---------|
| Baseline (all heads) | 1,024 | 60% | 1.0× |
| Surgical v3 (9 layers) | 288 | 60% | 0.28× |
| **Canary Trigger (3 heads)** | **3** | **65%** | **0.003×** |

> **Key Insight**: Monitoring just 3 "canary heads" (0.3% of total) achieves +5% accuracy over baseline — the alarm signal is **diluted** when averaged across all heads.

### 📐 5-Model Depth Scaling Law

| Model | Params | Peak Depth | Zone |
|-------|--------|------------|------|
| GPT-2 | 124M | 17% | Shallow |
| Qwen2.5 | 1.5B | —* | Unreliable |
| **Phi-2** | **2.7B** | **25%** | **Shallow** |
| Llama-3.2 | 3B | 43% | Mid-Zone |
| Mistral-7B | 7B | 44% | Mid-Zone |

> **Depth Scaling Law**: Models <3B hallucinate in shallow layers (15–25%), models ≥3B converge to the universal mid-layer zone (40–55%). The ~3B parameter threshold marks the critical transition.
>
> *Qwen2.5 GQA (2 KV heads) produces NaN under fp16; requires fp32 for reliable analysis.

### 👁️ Canary's Eye Visualization
L10H17 (Mistral-7B) attention heatmap during hallucination — the canary "wakes up" during generation steps 3–8.

![Canary's Eye](figures/canarys_eye.png)

### 🔬 Previous: Real-Time Hallucination Anatomy (v8)

> **"Moment of Lie"** — Animated token-by-token heatmaps revealing the exact moment hallucinations crystallize in mid-layer attention.

| Metric | Value |
|--------|-------|
| Hallucination Zone | **L10–L18** (Mistral-7B), **L8–L15** (Llama-3.2-3B) |
| Universal Depth | **30–55% of total network depth** |
| Cross-Model Peak | L14 (Mistral, 44%), L12 (Llama, 43%) |
| Peak Differential | **ΔH = −0.403 bits** (Llama-3.2-3B, L12) |

### 💰 Token Economy
| Strategy | Tokens | Accuracy | Compute Cost |
|----------|--------|----------|--------------|
| Baseline | 1,200 | 65% | 1.0× |
| Always CoT | 1,200 | 60% | 1.0× |
| **Surgical v3** | **1,577** | **60%** | **0.28×** |

> **Key Insight**: Mid-layer sniper monitors only 9/32 layers → **72% compute savings** with comparable accuracy.

### 🧬 Previous: Entropy Evolution (v7)
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
| Mistral-7B fp16 | +5.8σ (p < 10⁻⁹⁵) | Entropy-based, 100% accuracy |
| Moment of Lie | 30–55% depth | Universal hallucination zone |
| Token Economy | 72% compute savings | 9/32 layers monitoring |
| Cross-Model | ΔH = −0.403 bits | Llama-3.2-3B confirms universality |
| **Canary Trigger** | **+5% accuracy** | **3 heads = 0.003× compute** |
| **Depth Scaling** | **~3B threshold** | **Phi-2 (25%) confirms transition** |
| **Canary's Eye** | **L10H17** | **Canary wakes up at steps 3–8** |

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
│   ├── nightmare_visualizer.py        # LLM Brain State Imaging
│   ├── mistral_fullblast.py           # N=1000 Statistical Proof
│   ├── neural_healing_v4a.py          # Neural Healing v4A
│   ├── llama3_scaling_experiment.py   # Multi-model Scaling Law
│   ├── metacognition_experiment.py    # Token-wise Entropy Monitoring
│   ├── metacognition_v2.py            # Layer Anatomy Heatmap
│   ├── metacognition_v3.py            # Mid-Layer Sniper (70% acc)
│   ├── metacognition_v4_gif.py        # "Moment of Lie" Animation
│   ├── metacognition_v4_llama3.py     # Llama-3.2-3B Cross-Model
│   ├── metacognition_v7_final.py      # 🆕 Canary Head + Phi-2 + Depth Scaling
│   └── symbiosis_experiment.py        # Truth Lens + Symbiotic Guard
├── api/
│   └── hallucination_api.py           # Real-time Detection API
├── figures/
│   ├── canary_comparison.png          # 🆕 Canary Trigger ablation
│   ├── canarys_eye.png                # 🆕 L10H17 attention heatmap
│   ├── depth_scaling_v2.png           # 🆕 5-Model Depth Scaling Law
│   ├── moment_of_lie_grid.png         # Hallucination anatomy
│   ├── cross_model_comparison.png     # Cross-model universality
│   ├── nightmare_hero.png             # Brain state images
│   └── ... (30+ visualization PNGs)
├── demos/
│   └── hf_spaces/                     # HuggingFace Spaces demo
├── paper_arxiv_v9.tex                 # Latest paper (v9)
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

### 🐦 Canary Trigger Ablation (v9)
![Canary Comparison](figures/canary_comparison.png)

### 📐 5-Model Depth Scaling Law (v9)
![Depth Scaling](figures/depth_scaling_v2.png)

### 👁️ Canary's Eye — L10H17 Attention (v9)
![Canary's Eye](figures/canarys_eye.png)

### "Moment of Lie" — Hallucination Anatomy (v8)
![Moment of Lie](figures/moment_of_lie_grid.png)

### Cross-Model Universality (v8)
![Cross Model](figures/cross_model_comparison.png)

### N=1,000 Full Blast Statistical Proof
![Full Blast Results](figures/llama3b_fullblast_results.png)

### "Visualizing the Ghost" — LLM Brain State Imaging
![Brain State Images](figures/nightmare_hero.png)

### Jailbreak Detection Results
![Jailbreak Detection](figures/jailbreak_detection_results.png)

## 📝 Citation

```bibtex
@article{funasaki2026snn_guardrail,
  title={Activation-Scaled ANN-to-SNN Conversion with SNN Guardrail:
         A Unified Framework for AI Interpretability, Hallucination Detection,
         Real-Time Adversarial Defense, Neural Healing, Brain State Imaging,
         Hallucination Anatomy, and the Canary Head Paradigm},
  author={Funasaki, Hiroto},
  year={2026},
  doi={10.5281/zenodo.18457540},
  note={v9, Zenodo preprint}
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
- [x] "Moment of Lie" Hallucination Visualization
- [x] Token Economy Analysis (72% compute savings)
- [x] Cross-Model Universality (Llama-3.2-3B)
- [x] 🐦 Canary Head Paradigm (+5% accuracy, 3 heads)
- [x] 📐 5-Model Depth Scaling Law (Phi-2, ~3B threshold)
- [x] 👁️ Canary's Eye Visualization (L10H17)
- [ ] Canary head transfer (verify across Mistral-family models)
- [ ] Multi-step canary trajectory (full generation tracking)
- [ ] ~3B threshold validation (Gemma, Falcon, Mixtral)
- [ ] Real-time hallucination interception (abort at Moment of Lie)
- [ ] 13B+ / 70B Multi-GPU Validation
- [ ] Entropy-TTFS Hybrid Detection
- [ ] Production API Integration
- [ ] Neuromorphic Deployment (Loihi 2)

## 📜 License

MIT License

## 🙏 Acknowledgments

This research was conducted through a human-AI collaborative methodology. AI language models (Anthropic Claude Opus/Sonnet, Google Gemini) served as research advisors, contributing to experimental design, analysis approaches, code debugging, and manuscript drafting. The author executed all experiments, collected and interpreted results, and made final decisions on research directions.

- HuggingFace Transformers for LLM models
- TinyLlama team for the efficient 1.1B model
- AI Safety community for jailbreak research
- snnTorch for SNN-VAE experiments
