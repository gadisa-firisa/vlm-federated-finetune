

import os
import warnings

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common.config import unflatten_dict
from omegaconf import DictConfig
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from transformers import TrainingArguments
from trl import SFTTrainer

from federated_vlm.dataset import (
    get_tokenizer_and_propt_formatting,
    load_data,
    replace_keys,
)
from federated_vlm.models import cosine_annealing, get_model

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    num_rounds = context.run_config["num-server-rounds"]
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))
    training_arguments = TrainingArguments(**cfg.train.training_arguments)

    trainset = load_data(partition_id, num_partitions, cfg.dataset.name, cfg.dataset.subset)
    (
        tokenizer,
        formatting_prompts_func,
    ) = get_tokenizer_and_propt_formatting(cfg.model.name)

    model = get_model(cfg.model)
    set_peft_model_state_dict(model, msg.content["arrays"].to_torch_state_dict())

    new_lr = cosine_annealing(
        msg.content["config"]["server-round"],
        num_rounds,
        cfg.train.learning_rate_max,
        cfg.train.learning_rate_min,
    )

    training_arguments.learning_rate = new_lr
    training_arguments.output_dir = msg.content["config"]["save_path"]

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_arguments,
        max_seq_length=cfg.train.seq_length,
        train_dataset=trainset,
        formatting_func=formatting_prompts_func,
    )

    results = trainer.train()

    model_record = ArrayRecord(get_peft_model_state_dict(model))
    metrics = {
        "train_loss": results.training_loss,
        "num-examples": len(trainset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)
