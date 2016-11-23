class TransformerMixin(object):
    """Mixin class for all transformers in scikit-learn."""


class BaseEstimator(object):
    """Base class for all estimators in scikit-learn
    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).
    """


class ClassifierMixin(object):
    """
    Mixin class for all classifiers in scikit-learn.
    """