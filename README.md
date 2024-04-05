
# Alpaca Fine-tuning: Efficient Low-Rank Adaptation of Large Language Models

This project serves as an educational resource to help beginners quickly get started with QLoRA (Quantized Low-Rank Adaptation) for fine-tuning large language models. QLoRA is a technique that enables efficient adaptation of pre-trained models, making it suitable for resource-constrained environments.

Learning Objectives:

Understand the concept of QLoRA and its advantages over traditional fine-tuning methods.
Learn how to perform QLoRA fine-tuning on popular large language models.
Explore techniques for quantization and compression of model weights.
Gain hands-on experience with state-of-the-art natural language processing tools and libraries.
By following this project, beginners will be able to fine-tune large language models using QLoRA, allowing them to leverage the power of these models on devices with limited computational resources. The project is designed to provide a step-by-step guide, making it accessible to those new to the field of natural language processing and model fine-tuning.

This repository provides an efficient method for fine-tuning the Stanford Alpaca language model using low-rank adaptation (LoRA) techniques. The resulting model achieves performance comparable to `text-davinci-003` while enabling deployment on resource-constrained devices like the Raspberry Pi.

## Highlights

- **High-quality Instruction Model**: The fine-tuned Alpaca-LoRA model demonstrates strong performance in various natural language tasks, including question answering, code generation, and translation.
- **Efficient Training**: The training process leverages PEFT (Hugging Face's Parameter-Efficient Fine-Tuning library) and bitsandbytes, enabling rapid fine-tuning on a single consumer GPU (e.g., RTX 4090) within hours.
- **Flexible Deployment**: The LoRA weights can be merged into the base model or used as adapters, facilitating deployment on diverse hardware platforms, from high-end servers to resource-constrained edge devices.
- **Open-Source**: The project is open-source, and we welcome contributions from the community to enhance the codebase and explore new applications.

## Getting Started

### Installation

1. Clone the repository:

```bash
git clone https://github.com/jingmengzhiyue/alpaca-fine-tuning.git
cd alpaca-fine-tuning
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. If bitsandbytes fails to install, follow the [instructions](https://github.com/TimDettmers/bitsandbytes/blob/main/compile_from_source.md) to compile it from source.

### Fine-tuning

The `finetune.py` script provides a straightforward implementation of PEFT for fine-tuning the LLaMA model. Here's an example command:

```bash
python finetune.py \
    --base_model 'baffo32/decapoda-research-llama-7B-hf' \
    --data_path 'yahma/alpaca-cleaned' \
    --output_dir './Qlora-alpaca'
```

You can adjust various hyperparameters, such as batch size, learning rate, and LoRA configuration, by modifying the script arguments.

### Inference

The `generate.py` script loads the fine-tuned model and provides a Gradio interface for interactive inference. Example usage:

```bash
python generate.py \
    --load_8bit \
    --base_model 'baffo32/decapoda-research-llama-7B-hf' \
    --lora_weights 'tloen/alpaca-lora-7b'
```

### Docker Support

For a seamless setup and inference process, we provide Docker and Docker Compose configurations. Follow the instructions in the README to build and run the container image.


## License

This project is licensed under the [MIT License](LICENSE).