from __future__ import print_function
import numpy as np
from sklearn.utils import check_random_state

from spectral_dagger.utils import normalize, Estimator
from spectral_dagger.sequence import (
    ProbabilisticLSTM, sample_words, sample_words_balanced)

from seq_lda import (
    SequentialLDA, fit_callback_lda, lda_inference)


class LstmLDA(SequentialLDA, Estimator):
    record_attrs = ['n_topics', 'n_states']

    def __init__(
            self, n_topics=1, n_hidden=1, n_samples=100,
            lstm_kwargs=None, reuse=True,
            directory="results/lda_spectral/",
            random_state=None, verbose=False, name="LstmLDA"):
        self._init(locals())

    def point_distribution(self, context):
        return dict(
            n_hidden=range(2, context['max_states_per_topic']),
            n_topics=range(2, context['max_topics']))

    def fit(self, X, y=None):
        """ X is instance of ``MultitaskSequenceDataset``
            with a ``dictionary`` attr.  """
        random_state = check_random_state(self.random_state)

        if not X.learn_halt:
            raise Exception("Cannot run LSTM if ``learn_halt`` False.")
        self.record_indices(X.indices)
        lstm_kwargs = (
            dict(verbose=False)
            if self.em_kwargs is None else self.em_kwargs)

        corpus = X.dictionary.encode_as_bow(X.core_data)
        n_word_types = len(X.dictionary)
        n_input = X.dictionary.n_input

        self.generators_ = [
            ProbabilisticLSTM(
                n_input=n_input, n_hidden=self.n_hidden,
                directory=self.directory, **lstm_kwargs)
            for k in self.n_topics]

        def lstm_callback(class_word):
            class_word = normalize(class_word, ord=1, axis=1)
            log_prob_w = np.zeros_like(class_word)

            for k, dist in enumerate(class_word):
                data, _ = sample_words(
                    self.n_samples, dist, X.dictionary.words,
                    random_state)

                sa = self.generators_[k]
                sa.fit(data, reuse=self.reuse)

                for w, word in enumerate(X.dictionary.words):
                    log_prob_w[k][w] = sa.get_string_prob(word, log=True)

            return log_prob_w

        self.alpha_, _, self.gamma_ = fit_callback_lda(
            corpus, self.directory, self.n_topics,
            n_word_types, lstm_callback,
            log_name="" if self.verbose else "lstm.log")

        if X.n_transfer > 0:
            transfer_corpus = (
                X.transfer_dictionary.encode_as_bow(X.transfer_data))
            log_topics = self.log_topics(
                X.transfer_dictionary.words, prefix=not X.learn_halt)
            n_transfer_word_types = len(X.transfer_dictionary)

            lhood, transfer_gamma, phi = lda_inference(
                transfer_corpus, self.directory, self.n_topics,
                n_transfer_word_types, self.alpha_, log_topics)

            self.gamma_ = np.concatenate((self.gamma_, transfer_gamma))

        return self
