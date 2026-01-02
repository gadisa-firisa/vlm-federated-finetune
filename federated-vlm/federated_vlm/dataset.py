from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from transformers import AutoProcessor 
from datasets import Dataset

FDS = None 


def get_formatting_prompts_func(tokenizer):

    def formatting_func(example):
        return tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)

    return formatting_func

def get_tokenizer_and_propt_formatting(model_name: str):
    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "right"

    tokenizer = processor.tokenizer
    formatting_prompts_func = get_formatting_prompts_func(tokenizer)

    return tokenizer, formatting_prompts_func

def format_dataset(dataset: str):
   
    processed = []

    for i in range(len(dataset)):
        image = dataset[i]['image']
        question = dataset[i]['question']
        answer = dataset[i]['answers'][0]

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful assistant."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": question},
                ],
            },

            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": answer,
                    }
                ],
            }
        ]

        processed.append({"messages": messages})

    return Dataset.from_list(processed)


def load_data(partition_id: int, num_partitions: int, dataset_name: str, dataset_subset: str):

    global FDS
    if FDS is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        FDS = FederatedDataset(
            dataset=dataset_name,
            subset=dataset_subset,
            partitioners={"validation": partitioner},
        )
    client_trainset = FDS.load_partition(partition_id, "validation")
    client_trainset = format_dataset(client_trainset)

    return client_trainset


def replace_keys(input_dict, match="-", target="_"):
    """Recursively replace match string with target string in dictionary keys."""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict
