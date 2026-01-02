from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel
from datasets import load_dataset
import torch
from argparse import ArgumentParser

def main():
    parser = ArgumentParser(description="Peft path")
    parser.add_argument("--peft-path", type=str, required=True)
    args = parser.parse_args()
    peft_path = args.peft_path
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )

    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)

    dataset = load_dataset("lmms-lab/DocVQA", "DocVQA", split='test')
    sample = dataset.select([10])

    print(sample[0]['question'])
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful assistant.",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": sample[0]['image'],
                },
                {"type": "text", "text": sample[0]['question']},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    model = PeftModel.from_pretrained(base_model, peft_path)
    model.eval()

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)

if __name__ == '__main__':
    main()