# Data Setup

The evaluation uses **LoCoMo-10** from the MMA benchmark (arXiv:2602.16493).

## Download

```bash
git clone https://github.com/AIGeeksGroup/MMA.git
```

The evaluation script expects the data at:
```
MMA/MMA/public_evaluations/data/locomo10.json
```

Place the cloned `MMA/` directory at the **repo root**, or set the environment variable:

```bash
export MMA_LOCOMO_PATH=/path/to/locomo10.json
```

## Citation

If you use LoCoMo data, please cite the MMA paper:
```
@article{mma2025,
  title={MMA: Multimodal Memory Agent with Retrieve-Time Reliability Scoring},
  year={2025},
  url={https://arxiv.org/abs/2602.16493}
}
```
