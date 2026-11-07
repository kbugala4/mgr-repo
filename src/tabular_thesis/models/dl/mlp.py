"""Configurable dense baseline with fold-local, bounded one-hot encoding."""

from tabular_thesis.models.dl.torch_classifier import TorchClassifier


class MLPModel(TorchClassifier):
    def _network(self, n_classes, arrays):
        widths = self.params["hidden_dims"]
        if not widths or any(width < 1 for width in widths):
            raise ValueError("hidden_dims must contain positive widths")
        from torch import nn

        layers = []
        previous = arrays[0].shape[1]
        for width in widths:
            layers.extend(
                [nn.Linear(previous, width), nn.ReLU(), nn.Dropout(self.params["dropout"])]
            )
            previous = width
        layers.append(nn.Linear(previous, n_classes))
        return nn.Sequential(*layers)
