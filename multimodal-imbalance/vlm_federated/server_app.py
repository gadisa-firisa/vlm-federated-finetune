
import os
from datetime import datetime

from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.common.config import unflatten_dict
from flwr.serverapp import Grid, ServerApp
from omegaconf import DictConfig
from peft import get_peft_model_state_dict, set_peft_model_state_dict, LoraConfig, get_peft_model

from vlm_federated.dataset import replace_keys
from vlm_federated.models import get_model_and_processor, Config
from vlm_federated.strategy import (
    VLMFederated,
    ModalityAwareFedAvg,
    AlternatingObjectiveFedAvg,
)


app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    current_time = datetime.now()
    folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(os.getcwd(), f"results/{folder_name}")
    os.makedirs(save_path, exist_ok=True)

    num_rounds = context.run_config["num-server-rounds"]
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))

    config = Config()
    model, _  = get_model_and_processor(config)
    peft_config = LoraConfig(**config.lora)
    init_peft_model = get_peft_model(model, peft_config)
    arrays = ArrayRecord(get_peft_model_state_dict(init_peft_model))

    
    strategy_name = getattr(cfg.strategy, "name", "fedavg") if hasattr(cfg, "strategy") else "fedavg"
    fraction_train = getattr(cfg.strategy, "fraction_train", 1.0) if hasattr(cfg, "strategy") else 1.0
    fraction_evaluate = getattr(cfg.strategy, "fraction_evaluate", 0.0) if hasattr(cfg, "strategy") else 0.0

    if strategy_name == "modality_aware":
        default_weights = {"TEXT_ONLY": 0.5, "IMAGE_ONLY": 0.75, "PAIRED": 1.0}
        modality_weights = getattr(cfg.strategy, "modality_weights", default_weights)
        strategy = ModalityAwareFedAvg(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
            modality_weights=dict(modality_weights),
        )

    elif strategy_name == "alternating":
        odd_weights = getattr(
            cfg.strategy,
            "odd_round_weights",
            {"TEXT_ONLY": 1.0, "IMAGE_ONLY": 0.75, "PAIRED": 0.5},
        )
        even_weights = getattr(
            cfg.strategy,
            "even_round_weights",
            {"TEXT_ONLY": 0.5, "IMAGE_ONLY": 0.9, "PAIRED": 1.1},
        )
        strategy = AlternatingObjectiveFedAvg(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
            odd_round_weights=dict(odd_weights),
            even_round_weights=dict(even_weights),
        )
    else:
        strategy = VLMFederated(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
        )

    strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"save_path": save_path}),
        num_rounds=num_rounds,
        evaluate_fn=get_evaluate_fn(
            LoraConfig(**config.lora), cfg.train.save_every_round, num_rounds, save_path, config,
        ),
    )

def get_evaluate_fn(peft_cfg, save_every_round, total_round, save_path, config):
   
    def evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        if server_round != 0 and (
            server_round == total_round or server_round % save_every_round == 0
        ):
            model, _  = get_model_and_processor(config)
            init_peft_model = get_peft_model(model, peft_cfg)
            set_peft_model_state_dict(init_peft_model, arrays.to_torch_state_dict())
            init_peft_model.save_pretrained(f"{save_path}/peft_{server_round}")

        return MetricRecord()

    return evaluate
