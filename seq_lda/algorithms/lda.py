from __future__ import print_function

from spectral_dagger.utils import Estimator

from seq_lda import fit_lda, lda_inference, LDA


class LDAEstimator(LDA, Estimator):
    record_indices = []

    def __init__(
            self, n_topics=1, directory='results/lda_markov/', name="LDA"):
        self._init(locals())

        self.alpha_ = None
        self.log_topics_ = None
        self.gamma_ = {}

    def point_distribution(self, context):
        return {'n_topics': range(context['max_topics'])}

    def fit(self, X, y=None):
        self.record_indices(X.indices)

        corpus = X.dictionary.encode_as_bow(X.core_data)
        n_word_types = len(X.dictionary)
        self.alpha_, self.log_topics_, gamma = fit_lda(
            corpus, self.directory, self.n_topics, n_word_types)

        for idx, g in zip(X.core_indices, gamma):
            self.gamma_[idx] = g

        return self

    def log_topics(self, words=None):
        # TODO: Add check thats ``words`` matches
        # the words that this instance was trained with
        return self.log_topics_

    def _predictor_for_task(self, idx):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError("Cannot do symbol prediction with LDA.")

    def score(self, X, y=None):
        """ For now, we treat all datasets in X the same. In particular, even if
        X contains data from a task that we have seen before, we still evaluate
        it in the same way. This should be fixed in the future; the data we've
        previously seen may allow us to infer better values for gamma than the
        data we have presently. However, in general this function will not be
        used for data from tasks that we have already seen.

        """
        n_word_types = len(X.dictionary)
        corpus = X.dictionary.encode_as_bow(X.core_data + X.transfer_data)

        log_topics = self.log_topics(X.dictionary.words)

        lhood, gamma, phi = lda_inference(
            corpus, self.directory, self.n_topics, n_word_types,
            self.alpha_, log_topics)
        return lhood.sum()
