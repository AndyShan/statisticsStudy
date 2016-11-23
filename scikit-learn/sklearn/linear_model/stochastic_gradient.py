from abc import ABCMeta, abstractmethod
from ..base import BaseEstimator
from .base import SparseCoefMixin, LinearClassifierMixin
from ..externals import six


class BaseSGD(six.with_metaclass(ABCMeta, BaseEstimator, SparseCoefMixin)):
    """Base class for SGD classification and regression."""


class BaseSGDClassifier(six.with_metaclass(ABCMeta, BaseSGD,
                                           LinearClassifierMixin)):
    """

    """