# ANN-to-SNN Converter

## 📐 Activation-Scaled Conversion Formula

This module implements a universal ANN-to-SNN conversion formula:

```
θ = α × max(activation)
```

where `α ≥ 2.0` is a universal scaling factor.

## Key Features

- **No training required** - Pure weight copy from ANN
- **100% accuracy preservation** - Tested on MLP, CNN, ResNet
- **Hybrid readout** - Combines spike rate (70%) + membrane potential (30%)
- **IF neurons with soft reset** - Preserves residual information

## Files

| File | Description |
|------|-------------|
| `ann_to_snn_converter.py` | Full pipeline: PyTorch CNN training → SNN conversion |
| `autonomous_snn_optimizer.py` | Evolutionary threshold optimization using autonomous agents |
| `test_cnn_alpha.py` | Validation on CNN architecture |
| `test_resnet_alpha.py` | Validation on ResNet architecture |

## Quick Start

```python
# 1. Train ANN
python ann_to_snn_converter.py

# 2. Test α formula on different architectures
python test_cnn_alpha.py
python test_resnet_alpha.py

# 3. Run autonomous optimization
python autonomous_snn_optimizer.py
```

## Results

| Model | Required α | SNN Accuracy |
|-------|-----------|--------------|
| MLP | ≥ 0.3 | 100% |
| CNN | ≥ 1.5 | 100% |
| ResNet | ≥ 2.0 | 100% |

**Universal recommendation: `α = 2.0`**

## Citation

```bibtex
@misc{funasaki2026annsnn,
  title={Activation-Scaled ANN-to-SNN Conversion},
  author={Funasaki, Hiroto},
  year={2026},
  howpublished={Zenodo}
}
```

## License

MIT License
