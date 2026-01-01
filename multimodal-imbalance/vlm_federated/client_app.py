import os
import warnings

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common.config import unflatten_dict
from omegaconf import DictConfig
from peft import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
)
from transformers import TrainingArguments
from trl import SFTTrainer

from vlm_federated.dataset import (
    get_datasets,
    replace_keys,
)
from vlm_federated.models import cosine_annealing, get_model_and_processor, Config
from vlm_federated.dataset import ClientType


app = ClientApp()

@app.train()
def train(msg: Message, context: Context):

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    num_rounds = context.run_config["num-server-rounds"]
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))
    training_arguments = TrainingArguments(**cfg.train.training_arguments)

    client_type_str = None
    if isinstance(context.node_config, dict):
        client_type_str = context.node_config.get("client-type")
    if not client_type_str and hasattr(cfg, "static"):
        client_type_str = getattr(cfg.static, "client_type", None)

    if (not client_type_str) and hasattr(cfg, "static") and hasattr(cfg.static, "client_types"):

        types_list = list(getattr(cfg.static, "client_types"))
        if 0 <= partition_id < len(types_list):
            client_type_str = types_list[partition_id]
        
    if not client_type_str and hasattr(cfg, "static") and hasattr(cfg.static, "client_distribution"):
        dist = list(getattr(cfg.static, "client_distribution"))
        order = list(getattr(cfg.static, "client_order", ["TEXT_ONLY", "IMAGE_ONLY", "PAIRED"]))

        s = sum(float(x) for x in dist) if dist else 0.0
        if s <= 0:
            dist = [1.0]
            order = ["TEXT_ONLY"]
        else:
            dist = [float(x) / s for x in dist]
        cum = []
        c = 0.0
        for p in dist:
            c += p
            cum.append(c)
        pos = (partition_id + 1) / float(num_partitions)
        idx = 0
        for i, th in enumerate(cum):
            if pos <= th:
                idx = i
                break

        if idx >= len(order):
            idx = len(order) - 1
        client_type_str = order[idx]

    if not client_type_str:
        client_type_str = "TEXT_ONLY"

    client_type = ClientType(client_type_str)

    dataset_name = None
    if hasattr(cfg, "static") and hasattr(cfg.static, "datasets"):
        dataset_name = getattr(cfg.static.datasets, client_type.name)

    if not dataset_name:
        dataset_name = cfg.static.dataset.name

    trainset = get_datasets(
        partition_id,
        num_partitions,
        dataset_name,
        client_type=client_type,
    )

    config = Config()
    model_id = getattr(cfg.model, "name", None)
    
    if not model_id:
        model_id = "Qwen/Qwen-2.5-VL-3B-Instruct"

    quantization = dict(cfg.quantization) if hasattr(cfg, "quantization") else None
   

    gc_enabled = bool(getattr(cfg.train.training_arguments, "gradient_checkpointing", False))

    base_model, processor = get_model_and_processor(
        config,
        model_id=model_id,
        quantization=quantization,
        gradient_checkpointing=gc_enabled,
    )
    tokenizer = processor.tokenizer

    def formatting_prompts_func(sample):
        return tokenizer.apply_chat_template(
            sample["messages"], tokenize=False, add_generation_prompt=False
        )

    lora_dict = {}
    if hasattr(cfg, "lora"):
        lora_dict = dict(cfg.lora)
    peft_config = LoraConfig(**lora_dict)

    new_lr = cosine_annealing(
        msg.content["config"]["server-round"],
        num_rounds,
        cfg.train.learning_rate_max,
        cfg.train.learning_rate_min,
    )

    training_arguments.learning_rate = new_lr
    training_arguments.output_dir = msg.content["config"]["save_path"]

    trainer = SFTTrainer(
        model=base_model,
        args=training_arguments,
        max_seq_length=cfg.train.seq_length,
        train_dataset=trainset,
        formatting_func=formatting_prompts_func,
        data_collator=None,
        peft_config=peft_config,
    )

    set_peft_model_state_dict(trainer.model, msg.content["arrays"].to_torch_state_dict())

    results = trainer.train()
    model_record = ArrayRecord(get_peft_model_state_dict(trainer.model))
    metrics = {
        "train_loss": results.training_loss,
        "num-examples": len(trainset),
        "is_text_only": 1.0 if client_type == ClientType.TEXT_ONLY else 0.0,
        "is_image_only": 1.0 if client_type == ClientType.IMAGE_ONLY else 0.0,
        "is_paired": 1.0 if client_type == ClientType.PAIRED else 0.0,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)
