import os
import sys

import fire

# import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

# Determine the best device to run on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    if torch.backends.mps.is_available():
        device = "mps"  # Use Metal Performance Shaders (MPS) backend if available on macOS.
except AttributeError:
    pass  # MPS not available, fall back to CUDA or CPU


def load_model(base_model, lora_weights, load_8bit, device):
    """Loads the specified model with LoRA weights onto the given device.

    Args:
        base_model (str): Identifier for the base transformer model.
        lora_weights (str): Path or identifier for the LoRA weights.
        load_8bit (bool): Whether to load the model in 8-bit mode for efficiency.
        device (torch.device): The device to load the model onto.

    Returns:
        torch.nn.Module: The loaded model.
    """
    # Common keyword arguments for model loading, adjusted for the target device.
    common_kwargs = {
        "device_map": {
            "" if device.type in ["cuda", "mps"] else "cpu": device
        },
        "torch_dtype": (
            torch.float16 if device.type in ["cuda", "mps"] else None
        ),
        "low_cpu_mem_usage": device.type == "cpu",
    }
    # Load the base transformer model.
    model = LlamaForCausalLM.from_pretrained(
        base_model, load_in_8bit=load_8bit, **common_kwargs
    )
    model = PeftModel.from_pretrained(model, lora_weights, **common_kwargs)
    return model


def main(
    load_8bit=False,
    base_model="",
    lora_weights="tloen/alpaca-lora-7b",
    prompt_template="",
    server_name="0.0.0.0",
    share_gradio=False,
):
    """Main function to run the text generation model with command-line interface support.

    Args:
        load_8bit (bool): Load model in 8-bit mode for efficiency.
        base_model (str): Base model identifier for pretraining.
        lora_weights (str): Path or identifier for LoRA weights.
        prompt_template (str): Template name for generating prompts.
        server_name (str): Server name for hosting the model (if using Gradio).
        share_gradio (bool): Flag to share the Gradio app publicly.
    """
    # Fallback to BASE_MODEL environment variable if base_model argument is not provided.
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    # Load the model and move it to the appropriate device.
    model = load_model(base_model, lora_weights, load_8bit, device).to(device)

    # Adjust model configuration for token IDs.
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1 # Beginning of sequence
    model.config.eos_token_id = 2 # End of sequence

    if not load_8bit:
        model.half()  # Convert to half precision to address potential bugs.

    model.eval() # Set model to evaluation mode.
    # Compile model for performance improvement if supported.
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # Function to evaluate a given instruction and generate text.
    def evaluate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        stream_output=False,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }

        if stream_output:
            # Stream the reply 1 token at a time.
            # This is based on the trick of using 'stopping_criteria' to create an iterator,
            # from https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/text_generation.py#L216-L243.

            def generate_with_callback(callback=None, **kwargs):
                kwargs.setdefault(
                    "stopping_criteria", transformers.StoppingCriteriaList()
                )
                kwargs["stopping_criteria"].append(
                    Stream(callback_func=callback)
                )
                with torch.no_grad():
                    model.generate(**kwargs)

            def generate_with_streaming(**kwargs):
                return Iteratorize(
                    generate_with_callback, kwargs, callback=None
                )

            with generate_with_streaming(**generate_params) as generator:
                for output in generator:
                    # new_tokens = len(output) - len(input_ids[0])
                    decoded_output = tokenizer.decode(output)

                    if output[-1] in [tokenizer.eos_token_id]:
                        break

                    yield prompter.get_response(decoded_output)
            return  # early return for stream_output

        # Generate text without streaming.
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        yield prompter.get_response(output)

    # gr.Interface(
    #     fn=evaluate,
    #     inputs=[
    #         gr.components.Textbox(
    #             lines=2,
    #             label="Instruction",
    #             placeholder="Tell me about alpacas.",
    #         ),
    #         gr.components.Textbox(lines=2, label="Input", placeholder="none"),
    #         gr.components.Slider(
    #             minimum=0, maximum=1, value=0.1, label="Temperature"
    #         ),
    #         gr.components.Slider(
    #             minimum=0, maximum=1, value=0.75, label="Top p"
    #         ),
    #         gr.components.Slider(
    #             minimum=0, maximum=100, step=1, value=40, label="Top k"
    #         ),
    #         gr.components.Slider(
    #             minimum=1, maximum=4, step=1, value=4, label="Beams"
    #         ),
    #         gr.components.Slider(
    #             minimum=1, maximum=2000, step=1, value=128, label="Max tokens"
    #         ),
    #         gr.components.Checkbox(label="Stream output"),
    #     ],
    #     outputs=[
    #         gr.inputs.Textbox(
    #             lines=5,
    #             label="Output",
    #         )
    #     ],
    #     title="🦙🌲 Alpaca-LoRA",
    #     description="Alpaca-LoRA is a 7B-parameter LLaMA model finetuned to follow instructions. It is trained on the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset and makes use of the Huggingface LLaMA implementation. For more information, please visit [the project's website](https://github.com/tloen/alpaca-lora).",  # noqa: E501
    # ).queue().launch(server_name="0.0.0.0", share=share_gradio)
    # Old testing code follows.
    
    # Example usage: print responses for a list of instructions.
    for instruction in [
        "Tell me about alpacas.",
        "Tell me about the president of Mexico in 2019.",
        "Tell me about the king of France in 2019.",
        "List all Canadian provinces in alphabetical order.",
        "Write a Python program that prints the first 10 Fibonacci numbers.",
        "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",  # noqa: E501
        "Tell me five words that rhyme with 'shock'.",
        "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
        "Count up from 1 to 500.",
    ]:
        print("Instruction:", instruction)
        print("Response:", list(evaluate(instruction)))
        print()


if __name__ == "__main__":
    fire.Fire(main)
