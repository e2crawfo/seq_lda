#pragma once

#include <vector>
#include <iostream>
#include <random>

using namespace std;

struct LdaSuffstats{
    LdaSuffstats() = default;
    LdaSuffstats(int n_topics, int n_word_types);

    void init_zero();
    void init_random();
    // void init_from_corpus(const Corpus& corpus);

    int n_docs;
    int n_topics;
    int n_word_types;

    default_random_engine rng;
    uniform_real_distribution<double> dist;

    double alpha_suffstats;

    vector<vector<double>> class_word;
    vector<double> class_total;
};
