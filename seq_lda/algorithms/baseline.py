from __future__ import print_function
import abc
import six

from spectral_dagger.sequence import (
    AdjustedMarkovChain, SpectralSA, ExpMaxSA, GmmHmm)
from spectral_dagger.utils import Estimator

from seq_lda import MultitaskPredictor


@six.add_metaclass(abc.ABCMeta)
class OneByOne(MultitaskPredictor, Estimator):
    def __init__(self, bg_kwargs=None, name=None, directory=None):
        self._init(locals())

    def fit(self, X, y=None):
        self.record_indices(X.indices)

        tasks = X.core_data.as_sequences() + X.transfer_data.as_sequences()
        self.base_generators_ = [
            self.fit_base_generator(task, X) for task in tasks]

        return self

    def point_distribution(self, context):
        return super(OneByOne, self).point_distribution(context)

    def _predictor_for_task(self, idx):
        return self.base_generators_[idx]

    @abc.abstractmethod
    def fit_base_generator(self, sequences, X, previous=None):
        raise NotImplementedError()


@six.add_metaclass(abc.ABCMeta)
class Aggregate(MultitaskPredictor, Estimator):
    def __init__(self, bg_kwargs=None, name=None, directory=None, add_transfer_data=False):
        self._init(locals())

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

    def point_distribution(self, context):
        return super(Aggregate, self).point_distribution(context)

    def _predictor_for_task(self, idx):
        return self.base_generator_

    @abc.abstractmethod
    def fit_base_generator(self, sequences, X, previous=None):
        raise NotImplementedError()


class MarkovBase(object):
    def fit_base_generator(self, sequences, X, previous=None):
        return AdjustedMarkovChain.from_sequences(
            sequences, X.learn_halt, n_symbols=X.n_symbols)


class Markov1x1(MarkovBase, OneByOne):
    pass


class MarkovAgg(MarkovBase, Aggregate):
    pass


@six.add_metaclass(abc.ABCMeta)
class StatesMixin(object):
    def __init__(self, n_states=1, bg_kwargs=None, directory=None, name=None):
        self._init(locals())

    @property
    def record_attrs(self):
        return super(StatesMixin, self).record_attrs or set(['n_states'])

    def point_distribution(self, context):
        pd = super(StatesMixin, self).point_distribution(context)
        pd.update(n_states=range(2, context['max_states']))
        return pd


class SpectralBase(StatesMixin):
    def fit_base_generator(self, sequences, X, previous=None):
        bg_kwargs = self.bg_kwargs or dict()
        sa = SpectralSA(self.n_states, X.n_symbols, **bg_kwargs)
        sa.fit(sequences)
        return sa


class Spectral1x1(SpectralBase, OneByOne):
    pass


class SpectralAgg(SpectralBase, Aggregate):
    def __init__(
            self, n_states=1, add_transfer_data=False,
            bg_kwargs=None, directory=None, name=None):
        self._init(locals())


class ExpMaxBase(StatesMixin):
    def fit_base_generator(self, sequences, X, previous=None):
        if not X.learn_halt:
            raise Exception("Cannot run EM if ``learn_halt`` False.")
        bg_kwargs = self.bg_kwargs or dict()
        sa = ExpMaxSA(self.n_states, X.n_symbols, directory=self.directory, **bg_kwargs)
        sa.fit(sequences)
        return sa


class ExpMax1x1(ExpMaxBase, OneByOne):
    pass


class ExpMaxAgg(ExpMaxBase, Aggregate):
    def __init__(
            self, n_states=1, bg_kwargs=None, directory=None, add_transfer_data=False, name=None):
        self._init(locals())


class NeuralBase(object):
    def __init__(
            self, nn_class, n_hidden=2, bg_kwargs=None, directory=None, name=None):
        self._init(locals())

    def record_attrs(self):
        return super(NeuralBase, self).record_attrs or set(['n_hidden'])

    def point_distribution(self, context):
        pd = super(NeuralBase, self).point_distribution(context)
        pd.update(n_hidden=range(2, context['max_states']))
        return pd

    def fit_base_generator(self, sequences, X, previous=None):
        if not X.learn_halt:
            raise Exception("Cannot run Neural if ``learn_halt`` False.")
        bg_kwargs = self.bg_kwargs or dict()
        nn = previous or self.nn_class(n_hidden=self.n_hidden, **bg_kwargs)

        nn.fit(sequences)
        return nn


class Neural1x1(NeuralBase, OneByOne):
    pass


class NeuralAgg(NeuralBase, Aggregate):
    def __init__(
            self, nn_class, n_hidden=2, bg_kwargs=None,
            add_transfer_data=False, directory=None, name=None):
        self._init(locals())


class GmmHmmBase(StatesMixin):
    def fit_base_generator(self, sequences, X, previous=None):
        bg_kwargs = self.bg_kwargs or dict()
        gmm_hmm = previous or GmmHmm(
            n_states=self.n_states, directory=self.directory, **bg_kwargs)
        gmm_hmm.fit(sequences)
        return gmm_hmm


class GmmHmm1x1(GmmHmmBase, OneByOne):
    pass


class GmmHmmAgg(GmmHmmBase, Aggregate):
    def __init__(
            self, n_states=2, bg_kwargs=None,
            add_transfer_data=False, directory=None,
            name=None):
        self._init(locals())
