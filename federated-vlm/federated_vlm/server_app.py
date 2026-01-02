
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, cast

import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord, RecordDict
from flwr.common import Scalar
from flwr.common.config import unflatten_dict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from omegaconf import DictConfig
from peft import get_peft_model_state_dict, set_peft_model_state_dict

from federated_vlm.dataset import replace_keys
from federated_vlm.models import get_model

app = ServerApp()


class LossLogger:

    def __init__(self):
        self.losses: List[float] = []

    def __call__(
        self, records: list[RecordDict], weighting_metric_name: str
    ) -> Dict[str, Scalar]:
        total_loss = 0.0
        total_weight = 0.0
        for r in records:
            metric_record = next(iter(r.metric_records.values()))
            weight = cast(float, metric_record[weighting_metric_name])
            total_loss += cast(float, metric_record["train_loss"]) * weight
            total_weight += weight

        if total_weight == 0.0:
            return {}

        aggregated_loss = total_loss / total_weight
        self.losses.append(aggregated_loss)
        print(f"Aggregated loss: {aggregated_loss}")

        return {"aggregated_loss": aggregated_loss}

@app.main()
def main(grid: Grid, context: Context) -> None:
    current_time = datetime.now()
    folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(os.getcwd(), f"results/{folder_name}")
    os.makedirs(save_path, exist_ok=True)

    num_rounds = context.run_config["num-server-rounds"]
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))

    init_model = get_model(cfg.model)
    arrays = ArrayRecord(get_peft_model_state_dict(init_model))

    loss_logger = LossLogger()
    strategy = FedAvg(
        fraction_train=cfg.strategy.fraction_train,
        fraction_evaluate=cfg.strategy.fraction_evaluate,
        train_metrics_aggr_fn=loss_logger,
    )

    strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"save_path": save_path}),
        num_rounds=num_rounds,
        evaluate_fn=get_evaluate_fn(
            cfg.model, cfg.train.save_every_round, num_rounds, save_path
        ),
    )

    loss_file = os.path.join(save_path, "aggregated_losses.npy")
    np.save(loss_file, np.array(loss_logger.losses))

    print(f"Aggregated losses saved to {loss_file}")

def get_evaluate_fn(model_cfg, save_every_round, total_round, save_path):

    def evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        if server_round != 0 and (
            server_round == total_round or server_round % save_every_round == 0
        ):
            model = get_model(model_cfg)
            set_peft_model_state_dict(model, arrays.to_torch_state_dict())

            model.save_pretrained(f"{save_path}/peft_{server_round}")

        return MetricRecord()

    return evaluate
