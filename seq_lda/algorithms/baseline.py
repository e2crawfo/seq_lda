from __future__ import print_function
import abc
import six
from sklearn.base import clone

from spectral_dagger import Estimator
from seq_lda import MultitaskPredictor


@six.add_metaclass(abc.ABCMeta)
class _MultitaskSeqEstimator(MultitaskPredictor, Estimator):
    def fit_base_generator(self, sequences, X, previous=None):
        bg = clone(self.bg)
        bg.fit(sequences)
        return bg


class OneByOne(_MultitaskSeqEstimator):
    def __init__(self, bg=None, name=None, directory=None):
        self._set_attrs(locals())

    def fit(self, X, y=None):
        self.record_indices(X.indices)

        tasks = X.core_data.as_sequences() + X.transfer_data.as_sequences()
        self.base_generators_ = [
            self.fit_base_generator(task, X) for task in tasks]

        return self

    def _predictor_for_task(self, idx):
        return self.base_generators_[idx]


class Aggregate(_MultitaskSeqEstimator):
    def __init__(self, bg=None, name=None, directory=None, add_transfer_data=False):
        self._set_attrs(locals())

    def fit(self, X, y=None):
        self.record_indices(X.indices)

        if self.add_transfer_data:
            tasks = X.data.as_sequences()
        else:
            tasks = X.core_data.as_sequences()
        all_sequences = [
            seq for sequences in tasks for seq in sequences]

        self.base_generator_ = self.fit_base_generator(all_sequences, X)

        return self

    def _predictor_for_task(self, idx):
        return self.base_generator_
