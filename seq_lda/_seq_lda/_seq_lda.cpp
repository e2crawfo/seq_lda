#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_10_API_VERSION
#include <numpy/arrayobject.h>

#include <iostream>

#include "words.hpp"
#include "lda.hpp"
#include "markov_lda.hpp"
#include "callback_lda.hpp"

static char module_docstring[] = "TODO";

static char run_em_docstring[] = 
    "Run Variational EM to solve an LDA instance.\n\n"
    "Parameters\n"
    "----------\n"
    "    start: str\n"
    "        String specifying type of initialization. "
    "        Can be: \"seeded\", \"random\", or the name of a file storing a model.\n"
    "    directory: str\n"
    "        Location to store the learned model.\n"
    "    settings: str\n"
    "        Name of file storing settings for run.\n"
    "    n_topics: int >= 0\n"
    "        Number of topics to use for inference.\n"
    "    n_word_types: int >= 0\n"
    "        Number of word types in the corpus.\n"
    "    word_indices: list of list of int\n"
    "        Each sublist corresponds to a doc.\n"
    "        Each entry of each sublist gives the word type index.\n"
    "    word_counts: list of list of int\n"
    "        Each sublist corresponds to a doc.\n"
    "        Each entry of each sublist gives the count of the word type\n"
    "        at the same index in `word_indices`.";

static char run_em_markov_docstring[] = "TODO";
static char run_em_callback_docstring[] = "TODO";

static char inference_docstring[] = 
    "Infer variational parameters for a corpus of document "
    "and evaluate a lower bound on the log-likelihood.";

extern "C" PyObject* _seq_lda_run_em(PyObject *self, PyObject *args);
extern "C" PyObject* _seq_lda_run_em_markov(PyObject *self, PyObject *args);
extern "C" PyObject* _seq_lda_run_em_callback(PyObject *self, PyObject *args);
extern "C" PyObject* _seq_lda_inference(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
    {"run_em", _seq_lda_run_em, METH_VARARGS, run_em_docstring},
    {"run_em_markov", _seq_lda_run_em_markov, METH_VARARGS, run_em_markov_docstring},
    {"run_em_callback", _seq_lda_run_em_callback, METH_VARARGS, run_em_callback_docstring},
    {"inference", _seq_lda_inference, METH_VARARGS, inference_docstring},
    {NULL, NULL, 0, NULL}
};


PyMODINIT_FUNC init_seq_lda(void)
{
    PyObject *m = Py_InitModule3("_seq_lda", module_methods, module_docstring);
    if (m == NULL)
        return;

    /* Load `numpy` functionality. */
    import_array();
}

class PythonException: public runtime_error{
public:
    PythonException():runtime_error(""){};
    PythonException(const string& message):runtime_error(message){};
};

static Dictionary build_dictionary(int n_word_types, int n_symbols, PyObject *words){
    int length_words = PyList_Size(words);
    if(length_words != n_word_types){
        PyErr_SetString(PyExc_TypeError, "``n_word_types`` does not equal length of ``words``.");
        throw PythonException();
    }

    Dictionary dictionary(n_word_types, n_symbols);

    int word_length, i, j, is_list, symbol;
    PyObject *py_symbols, *py_symbol;
    Word word;

    // Build a vector of vectors of ints from a list of list of ints
    for (i = 0; i < n_word_types; i++){
        py_symbols = PyList_GetItem(words, i);
        is_list = PyList_Check(py_symbols);
        if(!is_list){
            PyErr_SetString(PyExc_TypeError, "Entry in ``words`` is not a list.");
            throw PythonException();
        }

        word_length = PyList_Size(py_symbols);
        word = Word();
        word.reserve(word_length);

        for(j = 0; j < word_length; j++){
            py_symbol = PyList_GetItem(py_symbols, j);
            symbol = PyInt_AsLong(py_symbol);
            if(symbol == -1 && PyErr_Occurred()){
                throw PythonException();
            }
            word.push_back(symbol);
        }

        dictionary.add_word(move(word));
    }

    return dictionary;
}

static Corpus build_corpus(
        PyObject* word_indices, PyObject* word_counts,
        int n_word_types, int n_symbols=0, PyObject* words=NULL){

    int n_docs = PyList_Size(word_indices);
    if(PyList_Size(word_counts) != n_docs){
        PyErr_SetString(
            PyExc_TypeError,
            "``word_indices`` and ``word_counts`` have different lengths.");
        throw PythonException();
    }

    int i, j, is_list, word_idx, word_count, n_word_types_for_doc;
    PyObject *py_words, *py_counts, *py_word, *py_count;

    Document doc;
    Corpus corpus(n_docs, n_word_types, n_symbols);

    // Build a corpus from the passed-in list.
    for (i = 0; i < n_docs; i++){
        py_words = PyList_GetItem(word_indices, i);
        is_list = PyList_Check(py_words);
        if(!is_list){
            PyErr_SetString(PyExc_TypeError, "Entry in word_indices is not a list.");
            throw PythonException();
        }

        py_counts = PyList_GetItem(word_counts, i);
        is_list = PyList_Check(py_counts);
        if(!is_list){
            PyErr_SetString(PyExc_TypeError, "Entry in word_counts is not a list.");
            throw PythonException();
        }

        n_word_types_for_doc = PyList_Size(py_words);

        doc = Document(n_word_types_for_doc);
        for(j = 0; j < n_word_types_for_doc; j++){
            py_word = PyList_GetItem(py_words, j);
            word_idx = PyInt_AsLong(py_word);
            if(word_idx == -1 && PyErr_Occurred()){
                throw PythonException();
            }

            py_count = PyList_GetItem(py_counts, j);
            word_count = PyInt_AsLong(py_count);
            if(word_count == -1 && PyErr_Occurred()){
                throw PythonException();
            }

            doc.add_word(word_idx, word_count);
        }

        corpus.add_doc(move(doc));
    }

    if(words != NULL){
        corpus.dictionary = build_dictionary(n_word_types, n_symbols, words);
    }

    return corpus;
}

static PyObject* build_return_value(const LDA& model, const Corpus& corpus){
    // Create numpy arrays to return values in.
    npy_intp log_prob_shape[2], gamma_shape[2];
    PyArrayObject *py_log_prob, *py_gamma;
    PyObject *item;

    log_prob_shape[0] = model.n_topics;
    log_prob_shape[1] = model.n_word_types;
    py_log_prob = (PyArrayObject*)(PyArray_ZEROS(2, log_prob_shape, NPY_DOUBLE, 0));

    gamma_shape[0] = corpus.n_docs;
    gamma_shape[1] = model.n_topics;
    py_gamma = (PyArrayObject*)(PyArray_ZEROS(2, gamma_shape, NPY_DOUBLE, 0));

    int success;
    // Transfer data from model into numpy arrays
    for (int k = 0; k < model.n_topics; k++){
        for (int w = 0; w < model.n_word_types; w++){
            item = Py_BuildValue("d", model.log_prob_w[k][w]);
            success = PyArray_SETITEM(py_log_prob, (char*)(PyArray_GETPTR2(py_log_prob, k, w)), item);
            Py_DECREF(item);
        }
    }
    for (int d = 0; d < corpus.n_docs; d++){
        for (int k = 0; k < model.n_topics; k++){
            item = Py_BuildValue("d", model.gamma[d][k]);
            success = PyArray_SETITEM(py_gamma, (char*)(PyArray_GETPTR2(py_gamma, d, k)), item);
            Py_DECREF(item);
        }
    }

    PyObject* py_alpha = Py_BuildValue("d", model.alpha);

    PyObject* ret = PyTuple_New(3);
    success = PyTuple_SetItem(ret, 0, py_alpha);
    success = PyTuple_SetItem(ret, 1, (PyObject*)(py_log_prob));
    success = PyTuple_SetItem(ret, 2, (PyObject*)(py_gamma));

    return ret;
}

static PyObject* build_markov_return_value(const MarkovLDA& model, const Corpus& corpus){
    // Create numpy arrays to return values in.
    npy_intp init_shape[2], trans_shape[3], halt_shape[2], gamma_shape[2];
    PyArrayObject *py_init, *py_trans, *py_halt, *py_gamma;
    PyObject *item;

    init_shape[0] = model.n_topics;
    init_shape[1] = model.n_symbols;
    py_init = (PyArrayObject*)(PyArray_ZEROS(2, init_shape, NPY_DOUBLE, 0));

    trans_shape[0] = model.n_topics;
    trans_shape[1] = model.n_symbols;
    trans_shape[2] = model.n_symbols;
    py_trans = (PyArrayObject*)(PyArray_ZEROS(3, trans_shape, NPY_DOUBLE, 0));

    halt_shape[0] = model.n_topics;
    halt_shape[1] = model.n_symbols;
    py_halt = (PyArrayObject*)(PyArray_ZEROS(2, halt_shape, NPY_DOUBLE, 0));

    gamma_shape[0] = corpus.n_docs;
    gamma_shape[1] = model.n_topics;
    py_gamma = (PyArrayObject*)(PyArray_ZEROS(2, gamma_shape, NPY_DOUBLE, 0));

    // Transfer data from model into numpy arrays
    int success;

    // init
    for (int k = 0; k < model.n_topics; k++){
        for (int s = 0; s < model.n_symbols; s++){
            item = Py_BuildValue("d", model.log_init[k][s]);
            success = PyArray_SETITEM(py_init, (char*)(PyArray_GETPTR2(py_init, k, s)), item);
            Py_DECREF(item);
        }
    }

    // trans
    for (int k = 0; k < model.n_topics; k++){
        for (int s = 0; s < model.n_symbols; s++){
            for (int r = 0; r < model.n_symbols; r++){
                item = Py_BuildValue("d", model.log_transitions[k][s][r]);
                success = PyArray_SETITEM(py_trans, (char*)(PyArray_GETPTR3(py_trans, k, s, r)), item);
                Py_DECREF(item);
            }
        }
    }

    // halt
    for (int k = 0; k < model.n_topics; k++){
        for (int s = 0; s < model.n_symbols; s++){
            item = Py_BuildValue("d", model.log_halt[k][s]);
            success = PyArray_SETITEM(py_halt, (char*)(PyArray_GETPTR2(py_halt, k, s)), item);
            Py_DECREF(item);
        }
    }

    // gamma
    for (int d = 0; d < corpus.n_docs; d++){
        for (int k = 0; k < model.n_topics; k++){
            item = Py_BuildValue("d", model.gamma[d][k]);
            success = PyArray_SETITEM(py_gamma, (char*)(PyArray_GETPTR2(py_gamma, d, k)), item);
            Py_DECREF(item);
        }
    }

    PyObject* py_alpha = Py_BuildValue("d", model.alpha);

    PyObject* ret = PyTuple_New(5);
    success = PyTuple_SetItem(ret, 0, py_alpha);
    success = PyTuple_SetItem(ret, 1, (PyObject*)(py_init));
    success = PyTuple_SetItem(ret, 2, (PyObject*)(py_trans));
    success = PyTuple_SetItem(ret, 3, (PyObject*)(py_halt));
    success = PyTuple_SetItem(ret, 4, (PyObject*)(py_gamma));

    return ret;
}

extern "C" PyObject *_seq_lda_run_em(PyObject *self, PyObject *args)
{
    const char *start, *directory, *log, *settings;
    int n_topics, n_word_types, estimate_alpha;
    PyObject *word_indices, *word_counts;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "ssssiiiOO", &start, &directory, &log, &settings,
                          &n_topics, &n_word_types, &estimate_alpha,
                          &word_indices, &word_counts)){
        return NULL;
    }

    ofstream log_file;
    if(string(log).length() > 0){
        log_file.open(string(directory) + "/" + string(log));
        LOG.rdbuf(log_file.rdbuf());
    }

    try{
        const Corpus corpus = build_corpus(word_indices, word_counts, n_word_types);
        LDA model(n_topics, estimate_alpha, start, directory, settings);
        model.fit(corpus);

        return build_return_value(model, corpus);
    }catch(const PythonException& e){
        return NULL;
    }
}

extern "C" PyObject *_seq_lda_run_em_markov(PyObject *self, PyObject *args)
{
    const char *start, *directory, *log, *settings;
    int n_topics, n_word_types, n_symbols, estimate_alpha, learn_halt;
    PyObject *word_indices, *word_counts, *words;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(
            args, "ssssiiiiiOOO", &start, &directory, &log, &settings,
            &n_topics, &n_word_types, &n_symbols, &estimate_alpha, &learn_halt,
            &word_indices, &word_counts, &words)){
        return NULL;
    }

    ofstream log_file;
    if(string(log).length() > 0){
        log_file.open(string(directory) + "/" + string(log));
        LOG.rdbuf(log_file.rdbuf());
    }

    try{
        Corpus corpus = build_corpus(
            word_indices, word_counts, n_word_types, n_symbols, words);
        MarkovLDA model(
            n_topics, estimate_alpha, learn_halt, start, directory, settings);
        model.fit(corpus);

        return build_markov_return_value(model, corpus);
    }catch(const PythonException& e){
        return NULL;
    }
}

extern "C" PyObject *_seq_lda_run_em_callback(PyObject *self, PyObject *args)
{
    const char *start, *directory, *log, *settings;
    int n_topics, n_word_types, estimate_alpha;
    PyObject *word_indices, *word_counts, *callback;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "ssssiiiOOO", &start, &directory, &log, &settings,
                          &n_topics, &n_word_types, &estimate_alpha,
                          &word_indices, &word_counts, &callback)){
        return NULL;
    }

    if (!PyCallable_Check(callback)) {
        PyErr_SetString(PyExc_TypeError, "Parameter ``callback`` must be callable.");
        return NULL;
    }
    Py_INCREF(callback);

    ofstream log_file;
    if(string(log).length() > 0){
        log_file.open(string(directory) + "/" + string(log));
        LOG.rdbuf(log_file.rdbuf());
    }

    PyObject *class_word, *arglist;
    try{
        const Corpus corpus = build_corpus(word_indices, word_counts, n_word_types);

        npy_intp shape[2];
        shape[0] = n_topics;
        shape[1] = n_word_types;
        class_word = PyArray_ZEROS(2, shape, NPY_DOUBLE, 0);
        arglist = PyTuple_New(1);
        int success = PyTuple_SetItem(arglist, 0, (PyObject*)(class_word));

        CallbackLDA model(n_topics, estimate_alpha, callback, class_word, arglist, start, directory, settings);
        model.fit(corpus);

        Py_XDECREF(callback);
        // Py_XDECREF(class_word); Not sure why we don't need this. Apparently PyArray_ZEROS returns a borrowed reference.
        Py_XDECREF(arglist);

        return build_return_value(model, corpus);
    }catch(const PythonException& e){
        // We use the convention that if the PythonException has a message, then
        // we should set the python exception string to that. Otherwise, we'll assume
        // that python has already set it for us. So the PythonException is just a mechanism
        // for jumping back up to this level quickly (avoids having to return NULL for
        // every function).

        Py_XDECREF(callback);
        // Py_XDECREF(class_word); Not sure why we don't need this. Apparently PyArray_ZEROS returns a borrowed reference.
        Py_XDECREF(arglist);

        return NULL;
    }
}

// Returns 0 on sucess, -1 on failure.
int callback_m_step(
        PyObject* py_callback, PyObject* _py_class_word,
        PyObject* py_arglist, vector<vector<double>>& log_prob_w,
        const vector<vector<double>>& class_word){

    PyArrayObject* py_class_word = (PyArrayObject*)(_py_class_word);
    PyObject* item;
    int success;
    int n_topics = log_prob_w.size();
    int n_word_types = log_prob_w[0].size();

    for(int k = 0; k < n_topics; k++){
        for(int w = 0; w < n_word_types; w++){
            item = Py_BuildValue("d", class_word[k][w]);
            success = PyArray_SETITEM(py_class_word, (char*)(PyArray_GETPTR2(py_class_word, k, w)), item);
            Py_DECREF(item);
        }
    }

    PyArrayObject *result = (PyArrayObject*)(PyObject_CallObject(py_callback, py_arglist));

    if(result == NULL){
        // The function that we call should have already set the exception,
        // so leave the message blank.
        throw PythonException();
    }

    for(int k = 0; k < n_topics; k++){
        for(int w = 0; w < n_word_types; w++){
            item = PyArray_GETITEM(result, (char*)(PyArray_GETPTR2(result, k, w)));
            log_prob_w[k][w] = PyFloat_AsDouble(item);
            Py_DECREF(item);
        }
    }

    Py_DECREF(result);
    return 0;
}

extern "C" PyObject *_seq_lda_inference(PyObject *self, PyObject *args)
{
    const char *settings, *directory, *log;
    double alpha;
    int n_topics, n_word_types, success;
    PyObject *word_indices, *word_counts;
    PyArrayObject *py_log_prob_w;

    // Parse the input tuple
    if (!PyArg_ParseTuple(args, "sssiidOOO", &directory, &log, &settings,
                          &n_topics, &n_word_types, &alpha,
                          &py_log_prob_w, &word_indices, &word_counts)){
        return NULL;
    }

    ofstream log_file;
    if(string(log).length() > 0){
        log_file.open(string(directory) + "/" + string(log));
        LOG.rdbuf(log_file.rdbuf());
    }

    const Corpus corpus = build_corpus(word_indices, word_counts, n_word_types);

    // log_prob_w = PyArray_FROM_OTF(_log_prob_w, NPY_DOUBLE, NPY_IN_ARRAY);
    double* log_prob_w = (double*)(PyArray_DATA(py_log_prob_w));

    // Create a model
    LDA model(n_topics, false, "random", directory, settings);
    model.allocate_storage(corpus);
    model.alpha = alpha;

    // Populate models log_prob_w array
    for (int k = 0; k < n_topics; k++){
        for (int w = 0; w < n_word_types; w++){
            model.log_prob_w[k][w] = log_prob_w[w + k * n_word_types];
        }
    }

    // Perform inference
    model.e_step(corpus);

    // Create numpy arrays to return values in.
    npy_intp likelihood_shape[1], gamma_shape[2], phi_shape[3];
    PyArrayObject *py_gamma, *py_phi, *py_likelihood;
    PyObject *item;

    likelihood_shape[0] = corpus.n_docs;
    py_likelihood = (PyArrayObject*)(PyArray_ZEROS(1, likelihood_shape, NPY_DOUBLE, 0));

    gamma_shape[0] = corpus.n_docs;
    gamma_shape[1] = n_topics;
    py_gamma = (PyArrayObject*)(PyArray_ZEROS(2, gamma_shape, NPY_DOUBLE, 0));

    phi_shape[0] = corpus.n_docs;
    phi_shape[1] = corpus.max_doc_length();
    phi_shape[2] = n_topics;
    py_phi = (PyArrayObject*)(PyArray_ZEROS(3, phi_shape, NPY_DOUBLE, 0));

    // Transfer data from model into numpy arrays
    for (int d = 0; d < corpus.n_docs; d++){
        for (int w = 0; w < corpus.docs[d].n_word_types; w++){
            for (int k = 0; k < n_topics; k++){
                item = Py_BuildValue("d", model.phi[d][w][k]);
                success = PyArray_SETITEM(py_phi, (char*)(PyArray_GETPTR3(py_phi, d, w, k)), item);
                Py_DECREF(item);
            }
        }

        for (int k = 0; k < n_topics; k++){
            item = Py_BuildValue("d", model.gamma[d][k]);
            success = PyArray_SETITEM(py_gamma, (char*)(PyArray_GETPTR2(py_gamma, d, k)), item);
            Py_DECREF(item);
        }

        item = Py_BuildValue("d", model.likelihood[d]);
        success = PyArray_SETITEM(py_likelihood, (char*)(PyArray_GETPTR1(py_likelihood, d)), item);
        Py_DECREF(item);
    }

    PyObject* ret = PyTuple_New(3);
    success = PyTuple_SetItem(ret, 0, (PyObject*)(py_likelihood));
    success = PyTuple_SetItem(ret, 1, (PyObject*)(py_gamma));
    success = PyTuple_SetItem(ret, 2, (PyObject*)(py_phi));

    return ret;
}
