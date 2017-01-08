from __future__ import print_function
import abc
import os
import numpy as np
from sklearn.utils import check_random_state

from spectral_dagger.utils import normalize, Estimator
from spectral_dagger.sequence import sample_words

from seq_lda import (
    SequentialLDA, fit_callback_lda, lda_inference, write_settings)
from seq_lda.algorithms.baseline import (
    MarkovBase, SpectralBase, ExpMaxBase, NeuralBase, GmmHmmBase)

LOG_ZERO = -10000


class MSSG(SequentialLDA, Estimator):
    def __init__(
            self, n_topics=1, n_samples=100, n_samples_scale=None,
            bg_kwargs=None, lda_settings=None, directory=None, name=None,
            random_state=None, verbose=False):
        self._init(locals())

    @property
    def record_attrs(self):
        return super(MSSG, self).record_attrs or set(['n_topics'])

    def point_distribution(self, context):
        pd = super(MSSG, self).point_distribution(context)
        pd.update(n_topics=range(2, context['max_topics']))
        return pd

    @abc.abstractmethod
    def fit_base_generator(self, sequences, X, previous=None):
        raise NotImplementedError()

    def fit(self, X, y=None):
        """ X is instance of ``MultitaskSequenceDataset``.  """
        random_state = check_random_state(self.random_state)

        self.record_indices(X.indices)

        corpus = X.core_data
        training_words = corpus.get_all_words()
        n_word_types = len(training_words)

        self.base_generators_ = None

        def callback(class_word):
            class_word = normalize(class_word, ord=1, axis=1)
            log_prob_w = np.zeros_like(class_word)

            old_bgs = self.base_generators_
            self.base_generators_ = []

            for k, dist in enumerate(class_word):
                if self.n_samples_scale is not None:
                    n_samples = int(self.n_samples_scale * len(dist))
                else:
                    n_samples = self.n_samples

                data, _ = sample_words(n_samples, dist, training_words, random_state)

                if old_bgs is None:
                    bg = self.fit_base_generator(data, X)
                else:
                    bg = self.fit_base_generator(data, X, previous=old_bgs[k])

                for w, word in enumerate(training_words):
                    if X.learn_halt:
                        log_prob_w[k][w] = bg.string_prob(word, log=True)
                    else:
                        log_prob_w[k][w] = bg.prefix_prob(word, log=True)

                self.base_generators_.append(bg)

            log_prob_w[np.isnan(log_prob_w)] = LOG_ZERO
            log_prob_w[np.isinf(log_prob_w)] = LOG_ZERO

            return log_prob_w

        settings = None
        if self.lda_settings is not None:
            settings = os.path.join(self.directory, 'settings.txt')
            write_settings(settings, **self.lda_settings)

        self.alpha_, _, self.gamma_ = fit_callback_lda(
            corpus, self.directory, self.n_topics,
            n_word_types, callback,
            settings=settings,
            log_name="" if self.verbose else ("%s.log" % self.name))

        if X.n_transfer > 0:
            transfer_corpus = X.transfer_data
            transfer_words = transfer_corpus.get_all_words()

            log_topics = self.log_topics(
                transfer_words, prefix=not X.learn_halt)
            n_transfer_word_types = len(transfer_words)

            lhood, transfer_gamma, phi = lda_inference(
                transfer_corpus, self.directory, self.n_topics,
                n_transfer_word_types, self.alpha_, log_topics)

            self.gamma_ = np.concatenate((self.gamma_, transfer_gamma))

        return self


class MarkovMSSG(MarkovBase, MSSG):
    pass


class SpectralMSSG(SpectralBase, MSSG):
    def __init__(
            self, n_states=2, n_topics=1, n_samples=100, n_samples_scale=None,
            bg_kwargs=None, lda_settings=None, directory=None, name=None,
            random_state=None, verbose=False):
        self._init(locals())


class ExpMaxMSSG(ExpMaxBase, MSSG):
    def __init__(
            self, n_states=2, n_topics=1, n_samples=100, n_samples_scale=None,
            bg_kwargs=None, lda_settings=None, directory=None, name=None,
            random_state=None, verbose=False):
        self._init(locals())


class NeuralMSSG(NeuralBase, MSSG):
    def __init__(
            self, nn_class, n_hidden=2, n_topics=1, n_samples=100, n_samples_scale=None,
            bg_kwargs=None, lda_settings=None, directory=None, name=None,
            random_state=None, verbose=False):
        self._init(locals())


class GmmHmmMSSG(GmmHmmBase, MSSG):
    def __init__(
            self, n_states=2, n_topics=1, n_samples=100, n_samples_scale=None,
            bg_kwargs=None, lda_settings=None, directory=None, name=None,
            random_state=None, verbose=False):
        self._init(locals())
