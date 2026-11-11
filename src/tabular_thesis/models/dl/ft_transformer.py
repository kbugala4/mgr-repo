"""FT-Transformer delegated to the pinned authors' reference implementation."""

from rtdl_revisiting_models import FTTransformer
from torch import nn

from tabular_thesis.models.dl.torch_classifier import TorchClassifier

# Bounds one float32 attention-score tensor, not total training VRAM; floor is eight rows.
ATTENTION_BUDGET_BYTES = 4 * 1024**3


class _FTTransformerNetwork(nn.Module):
    def __init__(
        self,
        n_numerical,
        n_categorical,
        cat_cardinalities,
        n_classes,
        d_token=192,
        n_blocks=3,
        attention_n_heads=8,
        attention_dropout=0.2,
        ffn_d_hidden=256,
        ffn_dropout=0.1,
        residual_dropout=0.0,
    ):
        super().__init__()
        self.reference = FTTransformer(
            n_cont_features=n_numerical,
            cat_cardinalities=cat_cardinalities,
            d_out=n_classes,
            n_blocks=n_blocks,
            d_block=d_token,
            attention_n_heads=attention_n_heads,
            attention_dropout=attention_dropout,
            ffn_d_hidden=ffn_d_hidden,
            ffn_d_hidden_multiplier=None,
            ffn_dropout=ffn_dropout,
            residual_dropout=residual_dropout,
        )

    def forward(self, x_num, x_cat):
        return self.reference(
            x_num if x_num is not None and x_num.shape[1] else None,
            x_cat.long() if x_cat is not None and x_cat.shape[1] else None,
        )


class FTTransformerModel(TorchClassifier):
    separate_inputs = True

    def _batch_size(self, X):
        score_bytes_per_row = self.params["attention_n_heads"] * (X.shape[1] + 1) ** 2 * 4
        maximum = ATTENTION_BUDGET_BYTES // score_bytes_per_row
        cap = 1 << max(3, maximum.bit_length() - 1)
        return min(super()._batch_size(X), cap)

    def _network(self, n_classes, arrays):
        params = {
            k: self.params[k]
            for k in [
                "d_token",
                "n_blocks",
                "attention_n_heads",
                "attention_dropout",
                "ffn_d_hidden",
                "ffn_dropout",
                "residual_dropout",
            ]
        }
        return _FTTransformerNetwork(
            len(self.numerical_columns),
            len(self.categorical_columns),
            self.preprocessor_.cardinalities,
            n_classes,
            **params,
        )
