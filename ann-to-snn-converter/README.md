# SNN Guardrail: Real-Time Neural Safety for AI
# SNNガードレール - AIの暴走を止める安全装置

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2026.XXXXX-b31b1b.svg)](https://arxiv.org/)

> 🛡️ **「AIの脳波を測って、嘘や暴走を止める」**
> 
> SNNを使ってLLMの内部状態を監視し、脱獄攻撃を**100%検知**

## 🔥 v4 New Features

### 🚀 Scaling Law Discovery
| Model | Parameters | TTFS Difference |
|-------|------------|-----------------|
| GPT-2 | 82M | +3.1 |
| **TinyLlama** | **1.1B** | **+4.2** |

→ モデルが大きいほど検知感度UP！

### 🛡️ SNN Guardrail
```python
from experiments.llama2_guardrail import SNNGuardrail

guardrail = SNNGuardrail(analyzer)
guardrail.calibrate(normal_prompts)

# リアルタイム検知
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
| TinyLlama TTFS | **+4.2** | Scaling law confirmed |
| Hallucination Detection | AUC 0.75 | Ensemble + auto-threshold |
| **Jailbreak Detection** | **100%** | 8/8 attack types |

## 📁 Repository Structure

```
ann-to-snn-converter/
├── experiments/
│   ├── llama2_guardrail.py          # 🆕 SNN Guardrail + TinyLlama
│   ├── jailbreak_detection.py       # 🆕 Jailbreak Detection
│   ├── gpt2_snn_analysis.py         # GPT-2 TTFS Analysis
│   ├── hallucination_detector_v3.py # Ensemble Detector
│   ├── large_scale_vit_validation.py # ViT-Base Validation
│   └── snn_interpretability.py      # TTFS/Synchrony Analysis
├── api/
│   └── hallucination_api.py         # Real-time Detection API
├── figures/
│   ├── jailbreak_detection_results.png  # 🆕
│   └── llama2_guardrail_analysis.png    # 🆕
├── paper_arxiv_v4.tex               # 🆕 Latest Paper
└── README.md                        # This file
```

## 🚀 Quick Start

### Installation

```bash
pip install torch torchvision numpy matplotlib scikit-learn
pip install transformers  # For LLM analysis
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

### Jailbreak Detection Results
![Jailbreak Detection](figures/jailbreak_detection_results.png)

### TinyLlama Guardrail Analysis
![Guardrail Analysis](figures/llama2_guardrail_analysis.png)

## 📝 Citation

```bibtex
@article{funasaki2026snn_guardrail,
  title={SNN Guardrail: Real-Time Neural Safety for Large Language Models},
  author={Funasaki, Hiroto},
  journal={arXiv preprint},
  year={2026},
  note={v4}
}
```

## 🛣️ Roadmap

- [x] GPT-2 TTFS Analysis (+3.1)
- [x] TinyLlama Scaling Law (+4.2)
- [x] SNN Guardrail Implementation
- [x] 100% Jailbreak Detection
- [ ] Llama-2-7B Validation
- [ ] Gradio/Streamlit Demo
- [ ] Production API Integration
- [ ] Neuromorphic Deployment (Loihi 2)

## 📜 License

MIT License - ろーる (cell_activation)

## 🙏 Acknowledgments

- HuggingFace Transformers for LLM models
- TinyLlama team for the efficient 1.1B model
- AI Safety community for jailbreak research
