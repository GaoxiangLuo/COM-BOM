# COM-BOM [EMNLP'25]
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
<a href='https://arxiv.org/abs/2510.01178'><img src='https://img.shields.io/badge/arXiv-2510.01178-brown.svg?logo=arxiv&logoColor=white'></a>
![BoTorch](https://img.shields.io/badge/Built%20with-BoTorch-EE4C2C?logo=pytorch&logoColor=white)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository contains the reference BoTorch implementation of COM-BOM for the following paper: <br/>

> **[COM-BOM: Bayesian Exemplar Search for Efficiently Exploring the Accuracy-Calibration Pareto Frontier](https://aclanthology.org/2025.emnlp-main.1027/)**  
> [Gaoxiang Luo](https://gaoxiangluo.github.io/), [Aryan Deshwal](https://aryandeshwal.github.io/)    
> *In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, 2025* 
>

[[`Paper`](https://aclanthology.org/2025.emnlp-main.1027/)] [[`BibTeX`](#CitingCOM-BOM)]

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
@inproceedings{luo-deshwal-2025-com,
    title = "COM-BOM: Bayesian Exemplar Search for Efficiently Exploring the Accuracy-Calibration Pareto Frontier",
    author = "Luo, Gaoxiang and Deshwal, Aryan",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.1027/",
    pages = "20350--20363",
    ISBN = "979-8-89176-332-6",
}
```
