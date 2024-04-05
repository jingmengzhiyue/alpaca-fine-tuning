# Alpaca-fine-tuning: 高效且高质量的语言模型微调

**Alpaca-fine-tuning** 是一个提供高效微调大型语言模型的解决方案。它利用量化低秩自适应 (Quantized Low-Rank Adaptation, LoRA) 技术和 Hugging Face 的 PEFT 库以及 Tim Dettmers 的 bitsandbytes 库,在单个 RTX 4090 GPU (或者其它有24GB显存的GPU) 上只需数小时即可完成微调。

## 主要特性

- **预训练模型**: 提供与 `text-davinci-003` 相媲美质量的指令遵循模型,在资源受限设备(如树莓派)上亦可运行(用于研究目的)。
- **可扩展性**: 代码支持对 7B、13B、30B 和 65B 等不同大小的模型进行微调。
- **量化低秩自适应 (QLoRA)**: 采用 QLoRA 技术进行高效内存友好的微调。 
- **公开权重**: 预训练的 QLoRA 权重可在 Hugging Face Hub 上获取,便于即时使用。

## 快速开始

1. 安装所需依赖:

```bash
pip install -r requirements.txt
```

2. 微调模型:

```bash
python finetune.py \
    --base_model 'baffo32/decapoda-research-llama-7B-hf' \
    --data_path 'yahma/alpaca-cleaned' \
    --output_dir './Qlora-alpaca'
```

3. 运行推理:

```bash
python generate.py \
    --load_8bit \
    --base_model 'baffo32/decapoda-research-llama-7B-hf' \
    --lora_weights 'tloen/alpaca-lora-7b'
```
或者使用上一步本地微调之后的权重
```bash
python generate.py \
    --load_8bit \
    --base_model 'baffo32/decapoda-research-llama-7B-hf' \
    --lora_weights './Qlora-alpaca'
```
## Docker 部署

Alpaca-fine-tuning 提供了 Docker 支持,方便部署和推理。只需构建并运行容器:

```bash
docker build -t alpaca-lora .
docker run --gpus=all --shm-size 64g -p 7860:7860 -v ${HOME}/.cache:/root/.cache --rm alpaca-lora generate.py \
    --load_8bit \
    --base_model 'baffo32/decapoda-research-llama-7B-hf' \
    --lora_weights 'tloen/alpaca-lora-7b'
```

此外,也支持使用 Docker Compose 进行更简化的设置和管理。

