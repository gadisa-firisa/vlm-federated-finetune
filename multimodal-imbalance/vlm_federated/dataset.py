import random
from enum import Enum
from typing import Callable
from flwr_datasets.partitioner import IidPartitioner
from flwr_datasets import FederatedDataset

FDS = None

class ClientType(str, Enum):
    TEXT_ONLY = "TEXT_ONLY"
    PAIRED = "PAIRED"
    IMAGE_ONLY = "IMAGE_ONLY"


def get_conversation_formatter(client_type: ClientType = ClientType.TEXT_ONLY) -> Callable:

    def get_image(example, i):
        imgs = example.get("image")
        if imgs is not None and i < len(imgs):
            return imgs[i]
        return None

    def text_chat(system_text: str, user_text: str, assistant_text: str):
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_text}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": assistant_text},
                ],
            },
        ]

    def image_chat(
        system_text: str,
        prompt_text: str,
        fallback_text: str,
        assistant_text: str,
        image,
    ):
        user_content = (
            [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ]
            if image is not None
            else [
                {"type": "text", "text": fallback_text},
            ]
        )
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_text}],
            },
            {
                "role": "user",
                "content": user_content,
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": assistant_text},
                ],
            },
        ]

    def format_text_only(example):
        outputs = []
        system = "You are a helpful assistant."
        for i in range(len(example["instruction"])):
            instr = str(example["instruction"][i])
            if "input" in example and i < len(example["input"]) and str(example["input"][i]).strip():
                user_text = instr + "\n" + str(example["input"][i])
            else:
                user_text = instr
            assistant_text = str(example.get("response", [""])[i])
            messages = text_chat(system, user_text, assistant_text)
            outputs.append(messages)
        return {"messages": outputs}

    def format_paired(example):
        outputs = []
        system = "You are a helpful assistant that describes images."
        for i in range(len(example["caption"])):
            img = get_image(example, i)
            cap = str(example["caption"][i])
            messages = image_chat(
                system,
                "Describe the image.",
                "Describe the image: <image>.",
                cap,
                img,
            )
            outputs.append(messages)
        return {"messages": outputs}

    def format_image_only(example):
        outputs = []
        system = "You are a helpful assistant that classifies images."
        for i in range(len(example["label"])):
            img = get_image(example, i)
            label_text = str(example["label"][i])
            messages = image_chat(
                system,
                "Provide the class label.",
                "Classify the image: <image>.",
                label_text,
                img,
            )
            outputs.append(messages)
        return {"messages": outputs}

    def format_weakly_paired(example):
        random.seed(42)
        outputs = []
        system = "You are a helpful assistant that describes images."
        for i in range(len(example["caption"])):
            cap = str(example["caption"][i])
            if random.random() < 0.3:
                cap = cap.split(" ")[: max(1, len(cap.split(" ")) // 3)]
                cap = " ".join(cap)
            img = get_image(example, i)
            messages = image_chat(
                system,
                "Describe the image.",
                "Describe the image: <image>.",
                cap,
                img,
            )
            outputs.append(messages)
        return {"messages": outputs}

    if client_type == ClientType.TEXT_ONLY:
        return format_text_only
    elif client_type == ClientType.PAIRED:
        return format_paired
    elif client_type == ClientType.IMAGE_ONLY:
        return format_image_only
    else:
        return format_weakly_paired


def get_datasets(
    partition_id: int,
    num_partitions: int,
    dataset_name: str,
    client_type: ClientType,
):
    global FDS
    if FDS is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        FDS = FederatedDataset(
            dataset=dataset_name,
            partitioners={"train": partitioner},
        )
    client_trainset = FDS.load_partition(partition_id, "train")

    fmt = get_conversation_formatter(client_type=client_type)
    client_trainset = client_trainset.map(fmt, batched=True)

    return client_trainset

    


def get_all_datasets():
    
    return {
        "TEXT_ONLY": {
            "dataset": "tatsu-lab/alpaca",
            "notes": "Uses columns instruction, input, output. We normalize output->response and include input if present.",
        },
        "PAIRED": {
            "dataset": "coco_captions",
            "notes": "Image captioning; columns typically image and captions (renamed to caption).",
        },
        "IMAGE_ONLY": {
            "dataset": "cifar10",
            "notes": "Image classification; columns image and label already match expected names.",
        },
    }

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
