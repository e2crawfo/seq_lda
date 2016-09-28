import numpy as np
from collections import defaultdict, Counter
import six
import abc
import os

from sklearn.utils import check_random_state
from sklearn.cross_validation import train_test_split

from spectral_dagger.sequence import AdjustedMarkovChain
from spectral_dagger.sequence import MixtureSeqGen
from spectral_dagger.utils import sample_multinomial, normalize


import seq_lda._seq_lda


EST_SETTINGS = "/home/eric/seq_lda/seq_lda/_seq_lda/settings.txt"
INF_SETTINGS = "/home/eric/seq_lda/seq_lda/_seq_lda/inf-settings.txt"


class BowCorpus(object):
    """ Collection of documents represented in BOW fashion. """
    def __init__(self, word_indices, word_counts, dictionary):
        assert len(word_indices) == len(word_counts)
        self.word_indices = word_indices
        self.word_counts = word_counts

        self.n_tokens = [sum(wc) for wc in word_counts]
        self.dictionary = dictionary
        self.all_indices = sorted(set([i for wi in word_indices for i in wi]))

    def normalized_word_indices(self):
        nwi_map = {i: n for n, i in enumerate(self.all_indices)}
        nwi = [[nwi_map[i] for i in wi] for wi in self.word_indices]
        return nwi

    def get_all_words(self):
        return self.dictionary.get_words(self.all_indices)

    def as_sequences(self):
        docs = []
        for wi, wc in zip(self.word_indices, self.word_counts):
            docs.append([])
            for i, c in zip(wi, wc):
                docs[-1].extend([self.dictionary.words[i]] * c)
        return docs

    def __len__(self):
        return len(self.word_indices)

    def max_tokens(self):
        return max(self.n_tokens) if self.n_tokens else 0

    def __getitem__(self, key):
        try:
            key = list(key)
            word_indices = [self.word_indices[i] for i in key]
            word_counts = [self.word_counts[i] for i in key]
        except TypeError:
            word_indices = [self.word_indices[key]]
            word_counts = [self.word_counts[key]]

        return BowCorpus(word_indices, word_counts, self.dictionary)

    def __iter__(self):
        return iter(zip(self.word_indices, self.word_counts))

    @staticmethod
    def _idx_list_to_corpus(self):
        pass

    def train_test_split(self, pct_train, random_state):
        train_word_indices = []
        train_word_counts = []

        test_word_indices = []
        test_word_counts = []

        for wi, wc in zip(self.word_indices, self.word_counts):
            indices = [i for i, c in zip(wi, wc) for j in range(c)]
            p = random_state.permutation(indices)
            n_train = int(pct_train * len(p))
            train, test = p[:n_train], p[n_train:]
            train = Counter(train)
            train_wi, train_wc = zip(*six.iteritems(train))

            test = Counter(test)
            test_wi, test_wc = zip(*six.iteritems(test))

            train_word_indices.append(list(train_wi))
            train_word_counts.append(list(train_wc))

            test_word_indices.append(list(test_wi))
            test_word_counts.append(list(test_wc))

        return (
            BowCorpus(train_word_indices, train_word_counts, self.dictionary),
            BowCorpus(test_word_indices, test_word_counts, self.dictionary))

    def __add__(self, other):
        if other.is_dummy():
            return BowCorpus(self.word_indices,
                             self.word_counts,
                             self.dictionary)
        elif self.is_dummy():
            return BowCorpus(other.word_indices,
                             other.word_counts,
                             other.dictionary)
        else:
            assert isinstance(other, BowCorpus)
            assert self.dictionary is other.dictionary, (
                "Can't concatenate corpuses with different dictionaries.")
            return BowCorpus(self.word_indices + other.word_indices,
                             self.word_counts + other.word_counts,
                             self.dictionary)

    @staticmethod
    def dummy(dictionary=None):
        return BowCorpus([], [], dictionary)

    def is_dummy(self):
        return (
            len(self.word_indices) == 0 and
            len(self.word_counts) == 0 and
            self.dictionary is None)


class Dictionary(object):
    def __init__(self):
        self.words = []
        self.word_indices = {}

    def encode(self, docs, to_hashable=None, name=None):
        if to_hashable is None:
            to_hashable = lambda x: x

        indices, counts = [], []
        for d in docs:
            if len(d) == 0:
                indices.append([])
                counts.append([])
                continue

            e = defaultdict(int)

            for w in d:
                hw = to_hashable(w)
                word_idx = self.word_indices.get(hw, None)
                if word_idx is None:
                    word_idx = len(self.word_indices)
                    self.word_indices[hw] = word_idx
                    self.words.append(w)

                e[word_idx] += 1

            i, c = zip(*list(six.iteritems(e)))
            indices.append(list(i))
            counts.append(list(c))

        return BowCorpus(indices, counts, self)

    def get_words(self, indices):
        return [self.words[i] for i in indices]


class MultitaskSequenceDataset(object):
    """ An object containing data for multiple tasks.

    Can be used as the ``X`` in the Dataset classes (and the ``y`` as well
    for the case of supervised learning).

    Either set of indices can be omitted. In that case, the core datasets
    are automatically given indices range(len(core)), and transfer datasets
    are given indices beginning at one more than the largest core index.

    The difference between core and transfer is only really relevant at
    training time.

    Parameters
    ----------
    core: array-like
        Core data.
    core_indices: list of int
        Indices/labels for the datasets in ``core``.
    transfer: array-like
        transfer data.
    transfer_indices: list of int
        Indices/labels for the datasets in ``transfer``.
    context: dict
        Added as attributes.

    """
    def __init__(
            self, core=None, core_indices=None,
            transfer=None, transfer_indices=None, **context):

        for k, v in six.iteritems(context):
            setattr(self, k, v)

        self.core_data = BowCorpus.dummy() if core is None else core

        if core_indices is None:
            core_indices = range(len(self.core_data))
        self.core_indices = core_indices
        assert len(self.core_data) == len(self.core_indices)

        self.transfer_data = (
            BowCorpus.dummy() if transfer is None else transfer)
        if transfer_indices is None:
            start = (max(self.core_indices) + 1) if self.core_indices else 0
            transfer_indices = range(start, start + len(self.transfer_data))
        self.transfer_indices = transfer_indices
        assert len(self.transfer_data) == len(self.transfer_indices)

        self.data = self.core_data + self.transfer_data
        self.indices = self.core_indices + self.transfer_indices

        self.context = context

        self.shape = (self.data.max_tokens(),)

        self._train_set, self._test_set = None, None

    @property
    def X(self):
        """ Having this property makes this class act like a ``Dataset``. """
        return self

    @property
    def y(self):
        """ Having this property makes this class act like a ``Dataset``. """
        return None

    def __len__(self):
        """ Number of samples in the first task of the dataset.
            Could be improved. Done this way to make it possible to use it with
            sklearn cross validation. """
        return self.shape[0]

    def __getitem__(self, key):
        # Super hacky, this is all just to support creation of training and
        # test sets, not general purpose indexing.

        # NB: We get rid of transfer data at this point. This is only really
        # called by the sklearn cross-validation code to obtain data for the
        # folds, and we don't really want to have transfer data during the
        # cross-validation step (the transfer tasks shouldn't affect our choice
        # of hyper-parameters; we should just learn a hypothesis that we think
        # will transfer well no matter what the new tasks are).
        assert hasattr(key, '__getitem__')
        if len(key) == 0:
            return MultitaskSequenceDataset(**self.context)
        assert isinstance(key[0], int)

        if self._test_set is None:
            # Indices for training.
            pct = float(len(key)) / len(self)
            random_state = np.random.RandomState(hash(tuple(key)) % (2**32-1))

            self._train_set, self._test_set = (
                self.core_data.train_test_split(pct, random_state))
            return MultitaskSequenceDataset(
                core=self._train_set, core_indices=self.core_indices,
                **self.context)
        else:
            test_set = self._test_set
            self._train_set, self._test_set = None, None
            return MultitaskSequenceDataset(
                core=test_set, core_indices=self.core_indices,
                **self.context)

    @property
    def n_core(self):
        return len(self.core)

    @property
    def n_transfer(self):
        return len(self.transfer)

    @property
    def n_tasks(self):
        return self.core + self.n_tasks

    @property
    def all(self):
        return zip(self.indices, self.data)

    @property
    def core(self):
        return zip(self.core_indices, self.core_data)

    @property
    def transfer(self):
        return zip(self.transfer_indices, self.transfer_data)


# Tasks:
# 1. Multitask learning/document completion
# 2. Transfer learning/completing documents never seen in the dataset
# 3. Probability assigned to documents in the corpus that we've never seen
#    before. Not sure if there is a good multi-task learning analog of this.
#    It would be like predicting the output of a new task without having seen
#    that task at all. The ability to do this will probably never be
#    practically useful, but it at least gives us a good measure of the
#    quality of the model.
@six.add_metaclass(abc.ABCMeta)
class MultitaskPredictor(object):
    task_indices = set()

    def record_indices(self, task_indices):
        """ Store indices of tasks that this estimator has learned about.

        Parameters
        ----------
        indices: list-like
            List of integers giving indices of tasks this est has learned.

        """
        if not self.task_indices:
            self.task_indices = set()

        self.task_indices |= set(task_indices)

    def predictor_for_task(self, task_idx):
        if task_idx not in self.task_indices:
            raise IndexError(
                "Have not learned a task with index %d." % task_idx)
        return self._predictor_for_task(task_idx)

    @abc.abstractmethod
    def _predictor_for_task(self, task_idx):
        raise NotImplementedError()

    def predict(self, X):
        """
        Going to assume that we're not allowed to use the data in X to do any
        training. So we have to have seen all the tasks in X before, either
        in a multitask or transfer learning capacity.

        Prediction takes the form of a distribution over next observations at
        each point in each sequence. Each prediction actually contains
        n_symbols + 1 entries, the extra entry being the probability of
        halting.

        """
        predictions = []
        for idx, data in X.core + X.transfer:
            predictor = self.predictor_for_task(idx)

            task_predictions = []
            for sequence in data:
                predictor.reset()
                seq_predictions = []

                for symbol in sequence:
                    seq_predictions.append(predictor.get_obs_dist())
                    predictor.update(symbol)

                seq_predictions.append(predictor.get_obs_dist())

                task_predictions.append(seq_predictions)
            predictions.append(task_predictions)

        return predictions


class GenericMultitaskPredictor(MultitaskPredictor):
    def __init__(
            self, predictors, task_indices=None,
            name="GenericMultitaskPredictor"):

        if task_indices is None:
            task_indices = range(len(predictors))

        self.record_indices(task_indices)
        self._predictors = {i: p for i, p in zip(task_indices, predictors)}
        self.name = name

    def _predictor_for_task(self, task_idx):
        return self._predictors[task_idx]


class SequentialLDA(MultitaskPredictor):
    def __init__(self, alpha, generators, gamma=None, name="SequentialLDA"):
        """ Can supply ``gamma`` for some tasks. If theta is not supplied
        for a given task index, it will be generated randomly from
        Dirichlet(alpha).

        """
        self.alpha_ = alpha
        self.base_generators_ = generators
        self.n_topics = len(generators)
        self.name = name

        if gamma is None:
            self.gamma_ = np.zeros((0, self.n_topics))
        else:
            self.gamma_ = gamma
            self.record_indices(range(gamma.shape[0]))

    def log_topics(self, words, prefix):
        """ Get topic-conditional word propabilities in log space
            for all topics in the current model. """

        _log_topics = np.zeros((len(self.base_generators_), len(words)))
        for i, generator in enumerate(self.base_generators_):
            for j, word in enumerate(words):
                _log_topics[i, j] = (
                    generator.prefix_prob(word, log=True) if prefix
                    else generator.string_prob(word, log=True))
        return _log_topics

    def _predictor_for_task(self, idx):
        gamma = self.gamma_[idx, :].copy()
        theta = normalize(gamma, ord=1)
        return MixtureSeqGen(theta, self.base_generators_)

    def generate_data(
            self,
            horizon=10,
            n_train_docs=100,
            n_transfer_docs=0,
            n_test_docs=0,
            n_train_wpd=100,
            n_test_wpd=0,
            random_state=None):
        random_state = check_random_state(random_state)

        n_symbols = self.base_generators_[0].n_observations
        n_words_per_doc = n_train_wpd + n_test_wpd

        n_docs = n_train_docs + n_transfer_docs + n_test_docs
        n_current_docs = self.gamma_.shape[0]

        if n_docs > n_current_docs:
            new_gamma = random_state.dirichlet(
                self.alpha_ * np.ones(self.n_topics), n_docs - n_current_docs)
            self.gamma_ = np.concatenate((self.gamma_, new_gamma))
            self.record_indices(range(n_current_docs, n_docs))

        docs = []
        for d in sorted(self.task_indices)[:n_docs]:
            docs.append([])
            for i in range(n_words_per_doc):
                topic = sample_multinomial(
                    normalize(self.gamma_[d], ord=1, axis=1), random_state)
                generator = self.base_generators_[topic]
                docs[-1].append(
                    generator.sample_episode(
                        horizon=horizon, random_state=random_state))

        train_data, test_data = generate_multitask_sequence_data(
            docs, n_train_tasks=n_train_docs, n_transfer_tasks=n_transfer_docs,
            train_split=(n_train_wpd, n_test_wpd),
            transfer_split=(n_train_wpd, n_test_wpd),
            context=dict(learn_halt=horizon == np.inf, n_symbols=n_symbols),
            random_state=random_state)

        return (
            train_data, test_data,
            dict(
                true_model=self,
                n_topics=self.n_topics,
                max_topics=2*self.n_topics,
                n_symbols=n_symbols,
                max_states=max(g.n_states for g in self.base_generators_),
                n_components=self.base_generators_[0].n_states))


class LDA(object):
    def __init__(self, alpha, log_topics, gamma=None, name="LDA"):
        self.alpha_ = alpha
        self.log_topics_ = log_topics
        self.gamma_ = (
            np.zeros((0, self.log_topics_.shape[0]))
            if gamma is None else gamma)
        self.name = name

    @property
    def n_topics(self):
        return self.log_topics_.shape[0]

    def _predictor_for_task(self, task_idx):
        raise NotImplementedError()

    def generate_data(
            self,
            n_train_docs=100,
            n_test_docs=0,
            n_words_per_doc=100,
            random_state=None):

        random_state = check_random_state(random_state)

        n_docs = n_train_docs + n_test_docs
        n_current_docs = self.gamma_.shape[0]

        if n_docs > n_current_docs:
            new_gamma = random_state.dirichlet(
                self.alpha_ * np.ones(self.n_topics), n_docs - n_current_docs)
            self.gamma_ = np.concatenate((self.gamma_, new_gamma))
            self.record_indices(range(n_current_docs, n_docs))

        docs = []
        for d in sorted(self.task_indices)[:n_docs]:
            docs.append([])
            for i in range(n_words_per_doc):
                topic = sample_multinomial(
                    normalize(self.gamma_[d], ord=1, axis=1), random_state)
                word = sample_multinomial(
                    self.log_prob_w[topic, :], random_state)
                docs[-1].append(word)

        train_data, test_data = generate_multitask_sequence_data(
            docs, n_train_tasks=n_train_docs, random_state=random_state)

        return (
            train_data, test_data,
            dict(
                true_model=self,
                n_topics=self.n_topics,
                max_states=max(g.n_states for g in self.base_generators_),
                n_components=self.base_generators_[0].n_states))

    def log_topics(self, words=None):
        return self.log_topics_


def generate_multitask_sequence_data(
        tasks, n_train_tasks=0, n_transfer_tasks=0,
        train_split=None, transfer_split=None, test_wpt=None,
        to_hashable=None, context=None, random_state=None):
    """ Generate multitask sequence data from a set of collections of sequences.
    Also does the work of taking the task data, encoded directly as a list of
    sequences, and converting it to BOW format.

    Allows at least three types of datasets to be generated:
        multitask learning, transfer learning, and model evaluation

    Parameters
    ----------
    tasks: list of list of str
        Tasks to generate from.
    n_train_tasks: int (optional)
    n_transfer_tasks: int (optional)
        Number of tasks to mark for use in transfer learning; algorithm
        will see an initial portion of these tasks, and have to predict the
        remainder.
    train_split: 2-tuple
    transfer_split: 2-tuple
    symbols: list (optional)
        Set of symbols used in the corpus. If not provided, set of symbols
        will be inferred from the task.
    context: dict (optional)
    random_state:

    """
    train_split = train_split if train_split else (None, 0)
    transfer_split = transfer_split if transfer_split else (None, 0)
    assert len(train_split) == 2
    assert len(transfer_split) == 2

    train_data = []
    transfer_data = []
    test_data = []
    for i, task in enumerate(tasks):
        if i < n_train_tasks:
            train, test = train_test_split(
                task, train_size=train_split[0], test_size=train_split[1])
            train_data.append(train)
        elif i < n_transfer_tasks + n_train_tasks:
            transfer, test = train_test_split(
                task, train_size=transfer_split[0],
                test_size=transfer_split[1])
            transfer_data.append(transfer)
        else:
            test = task if test_wpt is None else test[:test_wpt]

        test_data.append(test)

    dictionary = Dictionary()

    train_data = dictionary.encode(train_data, to_hashable=to_hashable)
    transfer_data = dictionary.encode(transfer_data, to_hashable=to_hashable)
    test_data = dictionary.encode(test_data, to_hashable=to_hashable)

    context = context or {}

    train_data = MultitaskSequenceDataset(
        train_data, transfer=transfer_data,
        dictionary=dictionary, **context)

    test_data = MultitaskSequenceDataset(
        test_data, dictionary=dictionary, **context)

    return train_data, test_data


def read_gamma(dir_name):
    filename = os.path.join(dir_name, 'final.gamma')
    with open(filename) as f:
        n_docs = int(f.readline().split(' ')[1])
        gamma = []

        for d in range(n_docs):
            g = np.array([float(g) for g in f.readline().split(',')[:-1]])
            gamma.append(g)
    return gamma


def read_results_markov_lda(dir_name):
    filename = os.path.join(dir_name, 'final.model')
    with open(filename) as f:
        int(f.readline().split(' ')[1])  # n_word_types
        n_topics = int(f.readline().split(' ')[1])
        n_symbols = int(f.readline().split(' ')[1])
        learn_halt = bool(int(f.readline().split(' ')[1]))
        alpha = float(f.readline().split(' ')[1])
        topics = []

        for k in range(n_topics):
            log_init_dist = np.array(
                [float(p) for p in f.readline().split(',')[:-1]])
            log_T = np.zeros((n_symbols, n_symbols))
            for i in range(n_symbols):
                log_T[i, :] = [float(p) for p in f.readline().split(',')[:-1]]

            if learn_halt:
                log_halt = np.array(
                    [float(p) for p in f.readline().split(',')[:-1]])
                T = np.exp(log_T)
                T = normalize(T, ord=1, axis=1)
                topics.append(
                    AdjustedMarkovChain(
                        np.exp(log_init_dist), T, np.exp(log_halt)))
            else:
                f.readline()
                topics.append(
                    AdjustedMarkovChain(np.exp(log_init_dist), np.exp(log_T)))

    gamma = read_gamma(dir_name)
    return SequentialLDA(alpha, topics), gamma


def read_results_lda(dir_name):
    filename = os.path.join(dir_name, 'final.model')
    with open(filename) as f:
        n_topics = int(f.readline().split(' ')[1])
        int(f.readline().split(' ')[1])  # n_word_types
        alpha = float(f.readline().split(' ')[1])
        topics = []

        for k in range(n_topics):
            log_beta_k = np.array(
                [float(p) for p in f.readline().split(',')[:-1]])
            topics.append(log_beta_k)
        topics = np.array(topics)

    gamma = read_gamma(dir_name)
    return LDA(alpha, topics), gamma


def process_results_markov_lda(log_init, log_T, log_halt, learn_halt):
    n_topics = log_init.shape[0]
    topics = []

    for k in range(n_topics):
        if learn_halt:
            T = np.exp(log_T[k])

            # Need to normalize if learning halt because C++ includes
            # halting probability in normalization.
            T = normalize(T, ord=1, axis=1)

            topics.append(
                AdjustedMarkovChain(
                    np.exp(log_init[k]), T, np.exp(log_halt[k])))
        else:
            topics.append(
                AdjustedMarkovChain(np.exp(log_init[k]), np.exp(log_T[k])))
    return topics


def _process_settings(settings):
    if settings is None:
        settings = EST_SETTINGS
    elif isinstance(settings, str):
        pass
    else:
        assert isinstance(settings, dict)
        assert "filename" in settings
        write_settings(**settings)
        settings = settings['filename']
    return settings


def write_settings(
        filename, var_max_iter=20, var_tol=1e-6, em_max_iter=100, em_tol=1e-4):

    with open(filename, 'w') as f:
        f.write("var max iter %d\n" % var_max_iter)
        f.write("var convergence %f\n" % var_tol)
        f.write("em max iter %d\n" % em_max_iter)
        f.write("em convergence %f\n" % em_tol)


def fit_lda(
        bow_corpus, directory, n_topics, n_word_types,
        estimate_alpha=True, settings=None,
        start="random", log_name=""):

    settings = _process_settings(settings)
    nwi = bow_corpus.normalized_word_indices()

    alpha, log_topics, gamma = seq_lda._seq_lda.run_em(
        start, directory, log_name, settings, n_topics, n_word_types,
        int(estimate_alpha), nwi, bow_corpus.word_counts)

    return alpha, log_topics, gamma


def fit_markov_lda(
        bow_corpus, directory, n_topics, n_word_types, n_symbols,
        learn_halt, estimate_alpha=True,
        settings=None, start="random", log_name=""):

    settings = _process_settings(settings)
    nwi = bow_corpus.normalized_word_indices()

    alpha, log_init, log_T, log_halt, gamma = seq_lda._seq_lda.run_em_markov(
        start, directory, log_name, settings, n_topics, n_word_types,
        n_symbols, int(estimate_alpha), int(learn_halt),
        nwi, bow_corpus.word_counts, bow_corpus.get_all_words())

    markov_chains = process_results_markov_lda(
        log_init, log_T, log_halt, learn_halt)

    return alpha, markov_chains, gamma


def fit_callback_lda(
        bow_corpus, directory, n_topics, n_word_types, callback,
        estimate_alpha=True, settings=None, start="random",
        log_name=""):

    settings = _process_settings(settings)
    nwi = bow_corpus.normalized_word_indices()

    alpha, log_topics, gamma = seq_lda._seq_lda.run_em_callback(
        start, directory, log_name, settings, n_topics, n_word_types,
        int(estimate_alpha), nwi, bow_corpus.word_counts, callback)
    return alpha, log_topics, gamma


def lda_inference(
        bow_corpus, directory, n_topics, n_word_types, alpha,
        log_topics, settings=INF_SETTINGS, log_name=""):

    settings = _process_settings(settings)
    nwi = bow_corpus.normalized_word_indices()

    likelihood, gamma, phi = seq_lda._seq_lda.inference(
        directory, log_name, settings, n_topics, n_word_types, alpha,
        log_topics, nwi, bow_corpus.word_counts)

    return likelihood, gamma, phi


def word_correct_rate(estimator, X, y=None):
    """ ``X`` must be an instance of ``MultitaskSequenceDataset``. """

    sequences = X.data.as_sequences()
    sequences = [s for s in sequences if len(sequences)]

    total_error = 0.0
    total_predictions = 0

    for task_idx, task_sequences in zip(X.indices, sequences):
        if len(task_sequences) == 0:
            continue

        predictor = estimator.predictor_for_task(task_idx)
        error = predictor.WER(task_sequences)
        n_predictions = sum(len(ts) + 1 for ts in task_sequences)

        total_error += error * n_predictions
        total_predictions += n_predictions

    return 1 - total_error / total_predictions


def log_likelihood_score(estimator, X, y=None, string=True):
    """ ``X`` must be an instance of ``MultitaskSequenceDataset``. """
    sequences = X.data.as_sequences()

    total_log_likelihood = 0.0

    for task_idx, task_sequences in zip(X.indices, sequences):
        if len(task_sequences) == 0:
            continue

        predictor = estimator.predictor_for_task(task_idx)
        log_likelihood = (
            predictor.mean_log_likelihood(task_sequences, string=string))
        total_log_likelihood += log_likelihood * len(task_sequences)

    return total_log_likelihood / sum(len(ts) for ts in sequences)


def one_norm_score(estimator, X, y=None):
    """ one norm error with the delta distribution which puts
        probability mass of 1 at the actual observation.

    """
    sequences = X.data.as_sequences()

    total_error = 0.0
    total_predictions = 0

    for task_idx, task_sequences in zip(X.indices, sequences):
        if len(task_sequences) == 0:
            continue

        predictor = estimator.predictor_for_task(task_idx)
        error = predictor.mean_one_norm_error(task_sequences)
        n_predictions = sum(len(s) + 1 for s in task_sequences)

        total_error += error * n_predictions
        total_predictions += n_predictions

    return -total_error / total_predictions


def RMSE_score(estimator, X, y=None):
    """ ``X`` must be an instance of ``MultitaskSequenceDataset``.

    Only works for continuous data.

    """
    sequences = X.data.as_sequences()
    sequences = [s for s in sequences if len(sequences)]

    total_error = 0.0
    total_predictions = 0

    for task_idx, task_sequences in zip(X.indices, sequences):
        if len(task_sequences) == 0:
            continue

        predictor = estimator.predictor_for_task(task_idx)
        error = predictor.RMSE(task_sequences)
        error = error**2
        n_predictions = sum(len(ts) for ts in task_sequences)

        total_error += error * n_predictions
        total_predictions += n_predictions

    return -np.sqrt(total_error / total_predictions)
