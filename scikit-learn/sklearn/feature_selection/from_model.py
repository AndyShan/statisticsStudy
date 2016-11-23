from ..base import TransformerMixin


class _LearntSelectorMixin(TransformerMixin):
    # Note because of the extra threshold parameter in transform, this does
    # not naturally extend from SelectorMixin
    """Transformer mixin selecting features based on importance weights.
    This implementation can be mixin on any estimator that exposes a
    ``feature_importances_`` or ``coef_`` attribute to evaluate the relative
    importance of individual features for feature selection.
    """