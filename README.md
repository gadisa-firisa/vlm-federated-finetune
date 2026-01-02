## Project Overview
This repository contains federated fine-tuning of a vision‑language model using [Flower](https://flower.ai/).
The base VLM used here is [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), trained with [PEFT](https://github.com/huggingface/peft) via [QLoRA](https://arxiv.org/abs/2305.14314).

- **`federated-vlm`**: general VLM fine‑tuning on paired image–text data.
- **`multimodal-imbalance`**: federated multimodal learning under modality imbalance (TEXT_ONLY, IMAGE_ONLY, PAIRED clients) with FedAvg, Modality‑Aware, and Alternating strategies.

## Directory Structure
```bash 
vlm-federated-finetune/
├── federated-vlm/
│   ├── federated_vlm/
│   │   ├── __init__.py
│   │   ├── client_app.py
│   │   ├── dataset.py
│   │   ├── models.py
│   │   ├── plot.py
│   │   └── server_app.py
│   ├── .rayignore
│   ├── pyproject.toml
│   └── test.py
│
├── multimodal-imbalance/
│   ├── vlm_federated/
│   │   ├── __init__.py
│   │   ├── client_app.py
│   │   ├── dataset.py
│   │   ├── models.py
│   │   ├── server_app.py
│   │   └── strategy.py
│   └── pyproject.toml
│
└── README.md
```
## Prerequisites
- Python: `3.12` recommended for `federated-vlm` (matches its `pyproject.toml`).
- GPU: Recommended. Quantized loading (4‑bit) is enabled.
- CUDA/ROCm: Required for GPU acceleration and bitsandbytes.
- Optional: Hugging Face token if the model/datasets require authentication (`huggingface-cli login`).

## A. Federated VLM
```bash 
cd federated-vlm      # navigate to the directory 
```
### Run
#### Using uv (recommended):
  - Install uv: https://docs.astral.sh/uv/

  ```bash 
  uv pip install "flwr[simulation]"         # install Flower
  uv run flwr run .                         # start the training 
  ```
#### Using pip:
```bash
python3 -m venv vlm-env && source vlm-env/bin/activate  # create venv and activate
pip install "flwr[simulation]"                          # install Flower
pip install -e .                                        # install dependencies 
flwr run .                                              # start training 
```


**CPU only:** set `options.backend.client-resources.num-gpus = 0` under `[tool.flwr.federations.local-simulation]` in `pyproject.toml`.

## B. Multimodal Imbalance
```bash
cd multimodal-imbalance
```

### Run

#### Using uv:
  ```bash 
  uv pip install "flwr[simulation]"         
  uv run flwr run .                         
  ```
#### Using pip:
```bash
python3 -m venv multimodal-env && source multimodal-env/bin/activate  
pip install "flwr[simulation]"                          
pip install -e .                                        
flwr run .                                              
```

## Configs
- **Rounds and strategy:** edit `pyproject.toml` under `tool.flwr.app.config`.
  - `num-server-rounds`: number of FL rounds.
  - `strategy.name`: one of `fedavg`, `modality_aware`, `alternating`.
  - For `modality_aware`/`alternating`, you can change weights under `tool.flwr.app.config.strategy`.
- **Model and quantization:**
  - `federated-vlm`: `tool.flwr.app.config.model` (e.g., `Qwen/Qwen2.5-VL-3B-Instruct`) and 4‑bit quantization.
  - `multimodal-imbalance`: see `tool.flwr.app.config.model` and `[tool.flwr.app.config.quantization]`.
- **Modality datasets (multimodal):** `tool.flwr.app.config.static.datasets` (e.g., `TEXT_ONLY=tatsu-lab/alpaca`, `PAIRED=coco_captions`, `IMAGE_ONLY=cifar10`).
- **Client mix (multimodal):** `tool.flwr.app.config.static.client_distribution` and `client_order`.

## Handling Errors
- **bitsandbytes/CUDA errors**: make sure compatible CUDA/ROCm toolkit and GPU drivers are installed. You can set GPU count to 0 for CPU only run.
- **OOM or slow runs**: lower `per-device-train-batch-size`, increase `gradient-accumulation-steps`, or reduce sequence length/rounds.
- **Ray/GPU allocation issues:** edit `options.backend.client-resources.num-gpus` under `[tool.flwr.federations.local-simulation]`.

## Status
- The **modality imbalance** task is an ongoing project. Plots and result tables will be added after runs complete. A detailed write‑up will be added later.

