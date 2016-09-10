#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <memory>

#include "words.hpp"
#include "suff_stats.hpp"
#include "alpha.hpp"
#include "utils.hpp"

using namespace std;

extern double VAR_CONVERGED;
extern int VAR_MAX_ITER;

extern double EM_CONVERGED;
extern int EM_MAX_ITER;

extern double INITIAL_ALPHA;

extern ostream LOG;

class LDA{
public:
    LDA(int n_topics, bool estimate_alpha, const string& start,
        const string& directory, const string& settings)
    :n_topics(n_topics), estimate_alpha(estimate_alpha),
    start(start), directory(directory), settings(settings),
    n_docs(0), n_word_types(0), alpha(INITIAL_ALPHA){};

    virtual double e_step(const Corpus& corpus);
    virtual void m_step(bool _estimate_alpha=true);

    virtual void allocate_storage(const Corpus& corpus);

    void fit(const Corpus& corpus);
    void inference(const Corpus& corpus);

    virtual void save(const string& name);
    virtual void load(){};

    virtual string to_string() const;

    void read_settings(const string& filename);

    friend ostream& operator << (ostream &out, const LDA& lda){
        out << lda.to_string() << endl;
        return out;
    }

    // Parameters
    int n_topics;
    bool estimate_alpha;

    string start;
    string directory;
    string settings;

    // Data-dependent
    int n_docs;
    int n_word_types;
    double alpha;

    vector<double> likelihood;
    vector<vector<double>> log_prob_w;
    vector<vector<vector<double>>> phi;
    vector<vector<double>> gamma;

    LdaSuffstats ss;
};

// Forward declaration of functions from variational_inference.hpp
double lda_inference(
        const Document& doc, const LDA& model,
        vector<double>& gamma, vector<vector<double>>& phi);
double compute_likelihood(
        const Document& doc, const LDA& model,
        const vector<double>& gamma, const vector<vector<double>>& phi);
