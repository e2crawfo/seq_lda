#pragma once

#include "lda.hpp"
#include "words.hpp"

class OptWords{
public:
    OptWords(){};
    OptWords(const Dictionary& dictionary);

    int n_word_types;
    int n_symbols;

    vector<vector<int>> i_idx;
    vector<vector<int>> j_idx;
    vector<vector<int>> transition_counts;
    vector<int> n_unique_transitions;
    vector<int> first_symbols;
    vector<int> last_symbols;
};

class MarkovLDA: public LDA{
public:
    MarkovLDA(
        int n_topics, bool estimate_alpha, bool learn_halt,
        const string& start, const string& directory, const string& settings)
    :LDA(n_topics, estimate_alpha, start, directory, settings), learn_halt(learn_halt){};

    virtual void m_step(bool _estimate_alpha=true);
    virtual void allocate_storage(const Corpus& corpus);
    virtual void save(const string& name);

    // Parameters
    int learn_halt;

    // Data-dependent
    int n_symbols;

    OptWords opt_words;

    vector<vector<double>> log_init; // n_topics x n_symbols
    vector<vector<vector<double>>> log_transitions; // n_topics x n_symbols x n_symbols
    vector<vector<double>> log_halt; // n_topics x n_symbols
};
