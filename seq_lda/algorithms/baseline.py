from __future__ import print_function
import numpy as np

from spectral_dagger.sequence import (
    AdjustedMarkovChain, SpectralSA, ExpMaxSA,
    ProbabilisticLSTM)
from spectral_dagger.utils import Estimator

from seq_lda import MultitaskPredictor


class Markov1x1(MultitaskPredictor, Estimator):
    record_attrs = []

    def __init__(self, name="Markov1x1"):
        self._init(locals())

    def point_distribution(self, context):
        return {}

    def fit(self, X, y=None):
        self.record_indices(X.indices)

        tasks = X.core_data.as_sequences() + X.transfer_data.as_sequences()
        self.markov_chains_ = [
            AdjustedMarkovChain.from_sequences(
                task, X.learn_halt,
                n_symbols=X.n_symbols) for task in tasks]

        return self

    def _predictor_for_task(self, idx):
        return self.markov_chains_[idx]


class MarkovAgg(MultitaskPredictor, Estimator):
    record_attrs = []

    def __init__(self, add_transfer_data=False, name="MarkovAgg"):
        self._init(locals())

    def point_distribution(self, context):
        return {}

    def fit(self, X, y=None):
        self.record_indices(X.indices)

        if self.add_transfer_data:
            tasks = X.data.as_sequences()
        else:
            tasks = X.core_data.as_sequences()
        all_sequences = [
            seq for sequences in tasks for seq in sequences]

        self.markov_chain_ = (
            AdjustedMarkovChain.from_sequences(
                all_sequences, X.learn_halt,
                n_symbols=X.n_symbols))

        return self

    def _predictor_for_task(self, idx):
        return self.markov_chain_


class StatesMixin(object):
    def point_distribution(self, context):
        return dict(n_states=np.arange(2, context['max_states']))


class Spectral1x1(MultitaskPredictor, StatesMixin, Estimator):
    record_attrs = ['n_states']

    def __init__(self, n_states=1, estimator='prefix', name="Spectral1x1"):
        self._init(locals())

    def fit(self, X, y=None):
        self.record_indices(X.indices)

        self.stoch_autos_ = []

        tasks = X.data.as_sequences()
        for sequences in tasks:
            sa = SpectralSA(
                self.n_states, X.n_symbols,
                estimator=self.estimator)
            sa.fit(sequences)
            self.stoch_autos_.append(sa)

        return self

    def _predictor_for_task(self, idx):
        return self.stoch_autos_[idx]


class SpectralAgg(MultitaskPredictor, StatesMixin, Estimator):
    record_attrs = ['n_states']

    def __init__(
            self, n_states=1, estimator='prefix',
            add_transfer_data=False, name="SpectralAgg"):
        self._init(locals())

    def fit(self, X, y=None):
        self.record_indices(X.indices)

        if self.add_transfer_data:
            tasks = X.data.as_sequences()
        else:
            tasks = X.core_data.as_sequences()
        all_sequences = [
            seq for sequences in tasks for seq in sequences]

        self.stoch_auto_ = SpectralSA(
            self.n_states, X.n_symbols, estimator=self.estimator)
        self.stoch_auto_.fit(all_sequences)

        return self

    def _predictor_for_task(self, idx):
        return self.stoch_auto_


class ExpMax1x1(MultitaskPredictor, StatesMixin, Estimator):
    def __init__(
            self, n_states=1, em_kwargs=None,
            directory="results/expmax1x1", name="ExpMax1x1"):
        self._init(locals())

    def fit(self, X, y=None):
        if not X.learn_halt:
            raise Exception("Cannot run EM if ``learn_halt`` False.")
        self.record_indices(X.indices)

        self.stoch_autos_ = []

        em_kwargs = (
            dict(verbose=False)
            if self.em_kwargs is None else self.em_kwargs)

        tasks = X.data.as_sequences()
        for sequences in tasks:
            sa = ExpMaxSA(
                self.n_states, X.n_symbols,
                directory=self.directory, **em_kwargs)
            sa.fit(sequences)
            self.stoch_autos_.append(sa)

        return self

    def _predictor_for_task(self, idx):
        return self.stoch_autos_[idx]


class ExpMaxAgg(MultitaskPredictor, StatesMixin, Estimator):
    def __init__(
            self, n_states=1, em_kwargs={},
            add_transfer_data=False, directory="results/expmaxagg",
            name="ExpMaxAgg"):
        self._init(locals())

    def fit(self, X, y=None):
        if not X.learn_halt:
            raise Exception("Cannot run EM if ``learn_halt`` False.")
        self.record_indices(X.indices)
        em_kwargs = (
            dict(verbose=False)
            if self.em_kwargs is None else self.em_kwargs)

        if self.add_transfer_data:
            tasks = X.data.as_sequences()
        else:
            tasks = X.core_data.as_sequences()
        all_sequences = [
            seq for sequences in tasks for seq in sequences]

        self.stoch_auto_ = ExpMaxSA(
            self.n_states, X.n_symbols,
            directory=self.directory, **em_kwargs)
        self.stoch_auto_.fit(all_sequences)

        return self

    def _predictor_for_task(self, idx):
        return self.stoch_auto_


class LSTM1x1(MultitaskPredictor, Estimator):
    def __init__(
            self, n_hidden=2, lstm_kwargs=None,
            directory="results/lstm1x1", name="Lstm1x1"):
        self._init(locals())

    def point_distribution(self, context):
        return dict(n_hidden=np.arange(2, context['max_states']))

    def fit(self, X, y=None):
        if not X.learn_halt:
            raise Exception("Cannot run LSTM if ``learn_halt`` False.")
        self.record_indices(X.indices)

        self.lstms_ = []

        lstm_kwargs = dict() if self.lstm_kwargs is None else self.lstm_kwargs

        tasks = X.data.as_sequences()
        for sequences in tasks:
            sa = ProbabilisticLSTM(directory=self.directory, **lstm_kwargs)
            sa.fit(sequences)
            self.lstms_.append(sa)

        return self

    def _predictor_for_task(self, idx):
        return self.lstms_[idx]


class Lstm(MultitaskPredictor, Estimator):
    def __init__(
            self, n_hidden=2, lstm_kwargs={},
            add_transfer_data=False, directory="results/lstmagg",
            name="LstmAgg"):
        self._init(locals())

    def point_distribution(self, context):
        return dict(n_hidden=np.arange(2, context['max_states']))

    def fit(self, X, y=None):
        if not X.learn_halt:
            raise Exception("Cannot run LSTM if ``learn_halt`` is False.")
        self.record_indices(X.indices)
        lstm_kwargs = dict() if self.lstm_kwargs is None else self.lstm_kwargs

        if self.add_transfer_data:
            tasks = X.data.as_sequences()
        else:
            tasks = X.core_data.as_sequences()
        all_sequences = [
            seq for sequences in tasks for seq in sequences]

        self.lstm_ = ProbabilisticLSTM(
            directory=self.directory, **lstm_kwargs)
        self.lstm_.fit(all_sequences)

        return self

    def _predictor_for_task(self, idx):
        return self.lstm_
