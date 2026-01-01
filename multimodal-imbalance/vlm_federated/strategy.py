from collections.abc import Iterable
from logging import INFO, WARN
from typing import Optional

from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.common import log
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg


class VLMFederated(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def configure_train(
            self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:

        messages = super().configure_train(server_round, arrays, config, grid)

        return messages

    def aggregate_train(
            self,
            server_round: int,
            replies: Iterable[Message],
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:

        arrays, metrics = super().aggregate_train(server_round, replies)

        return arrays, metrics
    

class CustomFedAvg(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def extract_metrics_dict(metric_record: MetricRecord) -> dict:
        for attr in ("metrics", "data", "dict", "values"):
            if hasattr(metric_record, attr):
                val = getattr(metric_record, attr)
                if isinstance(val, dict):
                    return val

        for meth in ("to_dict", "as_dict"):
            if hasattr(metric_record, meth):
                d = getattr(metric_record, meth)()
                if isinstance(d, dict):
                    return d
                pass
        return {}

    def aggregate_with_weights(
        self,
        server_round: int,
        replies: Iterable[Message],
        modality_weights: dict,
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        replies = list(replies)
        if not replies:
            return None, None

        sum_weights = 0.0
        agg_state = {}

        total_examples = 0.0
        wloss_sum = 0.0

        for msg in replies:
            if not msg.has_content():
                continue

            arrays: ArrayRecord = msg.content["arrays"]
            metrics_rec: MetricRecord = msg.content.get("metrics") if "metrics" in msg.content else None

            metrics = self.extract_metrics_dict(metrics_rec) if metrics_rec else {}
            num_examples = float(metrics.get("num-examples", 1.0))

        
            is_text = float(metrics.get("is_text_only", 0.0))
            is_img = float(metrics.get("is_image_only", 0.0))
            is_pair = float(metrics.get("is_paired", 0.0))
            if is_text >= is_img and is_text >= is_pair and is_text > 0.0:
                modality = "TEXT_ONLY"
            elif is_img >= is_text and is_img >= is_pair and is_img > 0.0:
                modality = "IMAGE_ONLY"
            elif is_pair > 0.0:
                modality = "PAIRED"
            else:
                modality = None

            m_weight = modality_weights.get(modality, 1.0) if modality is not None else 1.0
            weight = num_examples * m_weight

            state = arrays.to_torch_state_dict()

            if not agg_state:
                for k, v in state.items():
                    agg_state[k] = v.detach().to("cpu") * weight
            else:
                for k, v in state.items():
                    agg_state[k] += v.detach().to("cpu") * weight

            sum_weights += weight
            total_examples += num_examples
            if "train_loss" in metrics:
                wloss_sum += float(metrics["train_loss"]) * num_examples

        if sum_weights == 0.0:
            return None, None

        for k in agg_state:
            agg_state[k] /= sum_weights

        arrays_aggregated = ArrayRecord(agg_state)

        out_metrics = {}
        if total_examples > 0.0 and wloss_sum > 0.0:
            out_metrics["train_loss"] = wloss_sum / total_examples
        out_metrics["num-examples"] = total_examples

        return arrays_aggregated, MetricRecord(out_metrics)
    

class ModalityAwareFedAvg(CustomFedAvg):
    def __init__(self, modality_weights: dict, **kwargs):
        super().__init__(**kwargs)
        # e.g., {"TEXT_ONLY": 0.5, "PAIRED": 1.0, "IMAGE_ONLY": 0.75}
        self.modality_weights = modality_weights

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        return self.aggregate_with_weights(server_round, replies, self.modality_weights)


class AlternatingObjectiveFedAvg(CustomFedAvg):
    def __init__(self, odd_round_weights: dict, even_round_weights: dict, **kwargs):
        super().__init__(**kwargs)
        self.odd_weights = odd_round_weights
        self.even_weights = even_round_weights

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        weights = self.odd_weights if (server_round % 2 == 1) else self.even_weights
        return self.aggregate_with_weights(server_round, replies, weights)

