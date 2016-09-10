from __future__ import print_function
import numpy as np
from sklearn.utils import check_random_state

from spectral_dagger.utils import normalize, Estimator
from spectral_dagger.sequence import (
    ExpMaxSA, sample_words, sample_words_balanced)

from seq_lda import (
    SequentialLDA, fit_callback_lda, lda_inference)


class EmPfaLDA(SequentialLDA, Estimator):
    record_attrs = ['n_topics', 'n_states']

    def __init__(
            self, n_topics=1, n_states=1, n_samples=100,
            em_kwargs=None, balanced_samples=False,
            directory="results/lda_spectral/",
            random_state=None, verbose=False, name="EmPfaLDA"):
        self._init(locals())

    def point_distribution(self, context):
        return dict(
            n_states=range(2, context['max_states_per_topic']),
            n_topics=range(2, context['max_topics']))

    def fit(self, X, y=None):
        """ X is instance of ``MultitaskSequenceDataset``
            with a ``dictionary`` attr.  """
        random_state = check_random_state(self.random_state)

        if not X.learn_halt:
            raise Exception("Cannot run EM if ``learn_halt`` False.")
        self.record_indices(X.indices)
        em_kwargs = (
            dict(verbose=False)
            if self.em_kwargs is None else self.em_kwargs)

        corpus = X.core_data
        training_words = corpus.get_all_words()
        n_word_types = len(training_words)

        def em_callback(class_word):
            class_word = normalize(class_word, ord=1, axis=1)
            log_prob_w = np.zeros_like(class_word)

            self.generators_ = []

            for k, dist in enumerate(class_word):
                if self.balanced_samples:
                    data, _ = sample_words_balanced(
                        self.n_samples, dist, training_words, random_state)
                else:
                    data, _ = sample_words(
                        self.n_samples, dist, training_words, random_state)

                sa = ExpMaxSA(
                    n_observations=X.n_symbols, n_states=self.n_states,
                    directory=self.directory, **em_kwargs)
                sa.fit(data)

                for w, word in enumerate(training_words):
                    log_prob_w[k][w] = sa.get_string_prob(word, log=True)

                self.generators_.append(sa)

            return log_prob_w

        self.alpha_, _, self.gamma_ = fit_callback_lda(
            corpus, self.directory, self.n_topics,
            n_word_types, em_callback,
            log_name="" if self.verbose else "em.log")

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
