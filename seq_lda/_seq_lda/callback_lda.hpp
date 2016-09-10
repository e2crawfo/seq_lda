#pragma once

#include <Python.h>

#include "lda.hpp"

// Makes a call-back to python for its M-step
class CallbackLDA: public LDA{
public:
    CallbackLDA(
        int n_topics, bool estimate_alpha, PyObject* callback, PyObject* class_word, PyObject* arglist,
        const string& start, const string& directory, const string& settings)
    :LDA(n_topics, estimate_alpha, start, directory, settings),
     callback(callback), class_word(class_word), arglist(arglist){};

    CallbackLDA(const CallbackLDA&) = delete;
    CallbackLDA(CallbackLDA&&) = delete;
    CallbackLDA& operator=(const CallbackLDA&) = delete;
    CallbackLDA& operator=(CallbackLDA&&) = delete;

    virtual void m_step(bool _estimate_alpha=true);

    // Parameters
    PyObject* callback;
    PyObject* arglist;
    PyObject* class_word;

    // Data-dependent
    int n_symbols;
};

int callback_m_step(
    PyObject* py_callback, PyObject* _py_class_word,
    PyObject* py_arglist, vector<vector<double>>& log_prob_w,
    const vector<vector<double>>& class_word);
