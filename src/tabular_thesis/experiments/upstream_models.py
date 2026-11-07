"""Small integration guards around pinned TabArena reference models."""

import numpy as np
from tabarena.models.knn.model import KNNNewModel
from tabarena.models.modernnca.model import ModernNCAModel
from tabarena.models.realmlp.model import RealMLPModel
from tabarena.models.tabm.model import TabMModel


class FoldLocalCategories:
    def _fit(self, X, y, **kwargs):
        X = X[self.features].copy()
        self.fold_categories_ = {}
        for column in X.select_dtypes(include="category"):
            # Pandas slices retain categories absent from this training fold.
            X[column] = X[column].cat.remove_unused_categories()
            self.fold_categories_[column] = X[column].dtype
        return super()._fit(X, y, **kwargs)

    def _preprocess(self, X, **kwargs):
        X = X.copy()
        for column, dtype in self.fold_categories_.items():
            # AutoGluon's feature alignment can remove a constant categorical column.
            if column in X:
                X[column] = X[column].astype(dtype)
        return super()._preprocess(X, **kwargs)


class StudyRealMLP(FoldLocalCategories, RealMLPModel):
    pass


class StudyModernNCA(FoldLocalCategories, ModernNCAModel):
    seed_name = "random_state"

    def _fit(self, X, y, **kwargs):
        from functools import partial

        from tabarena.models.modernnca import model as upstream

        # shortcut: scoped factory shim assumes sequential fits; replace before threaded fitting.
        # TabArena 0.1.0 omits the seed when constructing its noisy quantile transformer.
        original = upstream.RTDLQuantileTransformer
        upstream.RTDLQuantileTransformer = partial(original, random_state=int(self.random_seed))
        try:
            return super()._fit(X, y, **kwargs)
        finally:
            upstream.RTDLQuantileTransformer = original


class StudyTabM(FoldLocalCategories, TabMModel):
    seed_name = "random_state"


class StudyKNN(FoldLocalCategories, KNNNewModel):
    def _fit(self, X, y, **kwargs):
        # The upstream quantile scaler uses NumPy's global RNG when subsampling.
        np.random.seed(int(self.random_seed))
        result = super()._fit(X, y, **kwargs)
        self.params_trained["available_training_rows"] = len(X)
        self.params_trained["effective_training_rows"] = self.model.n_samples_fit_
        self.params_trained["n_neighbors"] = self.model.n_neighbors
        return result


UPSTREAM_MODELS = {
    "realmlp": StudyRealMLP,
    "modernnca": StudyModernNCA,
    "tabm": StudyTabM,
    "knn": StudyKNN,
}
