"""
numcompute_stream -- a streaming, decision-tree learning framework.

Extends the group package ``numcompute`` (Assignment 2.1) so that every
component learns from data that arrives in chunks:

========================  =====================================================
module                    what it gives you
========================  =====================================================
``io``                    CSV -> arrays (via numcompute.io), chunk iterators
``stats``                 chunk-based mean / variance / quantiles / histograms
``preprocessing``         StandardScaler, MinMaxScaler, Imputer, OneHotEncoder
                          with ``partial_fit``
``tree``                  DecisionTreeClassifier with buffered retraining
``metrics``               accuracy, precision, recall, F1, confusion matrix,
                          AUC with ``update`` / ``reset`` / ``result``
``pipeline``              Pipeline and ColumnTransformer with ``partial_fit``
``stream``                StreamTrainer: prequential training + logging
``visualise``             matplotlib plots of metrics over time
``datasets``              synthetic streams with concept drift
========================  =====================================================

Only NumPy and matplotlib are used.
"""

from .metrics import F1, Accuracy, ConfusionMatrix, ErrorRate, Precision, Recall, StreamingAUC
from .pipeline import ColumnTransformer, Pipeline
from .preprocessing import Imputer, MinMaxScaler, OneHotEncoder, SimpleImputer, StandardScaler
from .stats import StreamingHistogram, StreamingQuantile, StreamingStats
from .stream import StreamTrainer
from .tree import DecisionTreeClassifier

__version__ = "0.2.0"

__all__ = [
    "Accuracy", "ConfusionMatrix", "ErrorRate", "F1", "Precision", "Recall", "StreamingAUC",
    "ColumnTransformer", "Pipeline",
    "Imputer", "MinMaxScaler", "OneHotEncoder", "SimpleImputer", "StandardScaler",
    "StreamingHistogram", "StreamingQuantile", "StreamingStats",
    "StreamTrainer",
    "DecisionTreeClassifier",
]
