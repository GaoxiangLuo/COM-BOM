# COM-BOM [EMNLP'25]

This repository contains the reference BoTorch implementation of COM-BOM for the following paper: <br/>

> **[COM-BOM: Bayesian Exemplar Search for Efficiently Exploring the Accuracy-Calibration Pareto Frontier]()**  
> [Gaoxiang Luo](https://gaoxiangluo.github.io/), [Aryan Deshwal](https://aryandeshwal.github.io/)    
> *In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, 2025* 
>

[[`Paper`]()] [[`BibTeX`](#CitingCOM-BOM)]

---
## Quickstart

### Installation

We manage dependencies with `uv`, which recreates the Python ">=3.12" environment described in `pyproject.toml`.

```bash
uv sync
```

### vLLM Serving

Training and evaluation needs a locally-hosted LLM. The default config (`configs/mmlupro.yaml`) expects `Qwen/Qwen3-8B` to be served on `http://localhost:8000`.

```bash
# 2xA100-40GB
uv run vllm serve Qwen/Qwen3-8B --port 8000 --chat-template misc/qwen3_nonthinking.jinja --enable-prefix-caching --gpu-memory-utilization 0.75 -tp 2

# 1xA100-80GB / 1xH100-80GB
uv run vllm serve Qwen/Qwen3-8B --port 8000 --chat-template misc/qwen3_nonthinking.jinja --enable-prefix-caching --gpu-memory-utilization 0.75
```

### Training

Kick off the Bayesian optimization pipeline with your chosen configuration.

```bash
uv run python train.py --config configs/mmlupro.yaml
```

### Testing

Evaluate Pareto candidates and baselines using the same configuration.

```bash
uv run python test.py --config configs/mmlupro.yaml
```

## Repository Structure

- **`train.py`**: Task-agnostic Bayesian optimization driver that loads YAML configurations
- **`test.py`**: Evaluation script for trained models and baselines
- **`configs/`**: YAML configuration templates for experiments
- **`combom/`**: Core optimization components:
  - Trust region management
  - Gaussian process modeling with categorical kernels
  - Multi-objective acquisition function optimization
- **`tasks/`**: Task definitions with evaluation logic and metric specifications
- **`utils/`**: Shared utilities (random seeding, etc.)

## <a name="CitingCOM-BOM"></a>Citing COM-BOM
If you use COM-BOM in your research, find the code useful, or would like to acknowledge our work, please consider citing our paper:

```BibTeX
@inproceedings{COMBOM,
  author    = {Luo, Gaoxiang and Deshwal, Aryan},
  title     = {COM-BOM: Bayesian Exemplar Search for Efficiently Exploring the Accuracy-Calibration Pareto Frontier},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  year      = {2025},
  publisher = "Association for Computational Linguistics",
}
