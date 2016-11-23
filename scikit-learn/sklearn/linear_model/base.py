from ..base import ClassifierMixin


class SparseCoefMixin(object):
    """Mixin for converting coef_ to and from CSR format.
    L1-regularizing estimators should inherit this.
    """


class LinearClassifierMixin(ClassifierMixin):
    """Mixin for linear classifiers.
    Handles prediction for sparse and dense X.
    """