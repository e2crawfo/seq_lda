#include "markov_lda.hpp"

void MarkovLDA::allocate_storage(const Corpus& corpus){
    LDA::allocate_storage(corpus);

    n_symbols = corpus.n_symbols;

    opt_words = OptWords(corpus.dictionary);

    log_init.resize(n_topics, vector<double>(n_symbols, 0.0));
    log_halt.resize(n_topics, vector<double>(n_symbols, 0.0));
    log_transitions.resize(
        n_topics, vector<vector<double>>(n_symbols,
            vector<double>(n_symbols, 0.0)));
}

/*
 * compute MLE lda markov model from sufficient statistics
 *
 */
void MarkovLDA::m_step(bool _estimate_alpha){

    int k, i, j, w, s, symbol_idx, length;
    double sum, log_sum, val, scale;
    vector<int> counts, i_idx, j_idx;

    for (k = 0; k < n_topics; k++)
    {
        fill(log_init[k].begin(), log_init[k].end(), 0.0);

        // compute log_init - but not in log space yet
        sum = 0;
        for (w = 0; w < n_word_types; w++)
        {
            symbol_idx = opt_words.first_symbols[w];
            log_init[k][symbol_idx] += ss.class_word[k][w];
        }

        // normalize log_init and transform to log space
        log_sum = log(ss.class_total[k]);
        for (s = 0; s < n_symbols; s++)
        {
            log_init[k][s] = normalized_log(log_init[k][s], log_sum);
        }

        // zero-out log_transitions and log_halt
        fill(log_transitions[k].begin(), log_transitions[k].end(), 
             vector<double>(n_symbols, 0.0));
        fill(log_halt[k].begin(), log_halt[k].end(), 0.0);

        // compute log_transitions - but not in log space yet
        for (w = 0; w < n_word_types; w++)
        {
            scale = ss.class_word[k][w];
            i_idx = opt_words.i_idx[w];
            j_idx = opt_words.j_idx[w];
            counts = opt_words.transition_counts[w];
            length = opt_words.n_unique_transitions[w];

            for (s = 0; s < length; s++){
                i = i_idx[s];
                j = j_idx[s];

                log_transitions[k][i][j] += counts[s] * scale;
            }

            symbol_idx = opt_words.last_symbols[w];
            log_halt[k][symbol_idx] += ss.class_word[k][w];
        }

        // normalize transitions and transform to log space
        for (i = 0; i < n_symbols; i++){
            sum = 0.0;
            for (j = 0; j < n_symbols; j++){
                sum += log_transitions[k][i][j];
            }

            // If learning halting probs, we also normalize over halting probs
            sum += learn_halt ? log_halt[k][i] : 0;

            log_sum = log(sum);

            for (j = 0; j < n_symbols; j++){
                val = log_transitions[k][i][j];
                log_transitions[k][i][j] = normalized_log(val, log_sum);
            }

            if(learn_halt){
                log_halt[k][i] = normalized_log(log_halt[k][i], log_sum);
            }
        }
    }

    // Compute log_prob from log_transitions
    double log_prob;
    for (k = 0; k < n_topics; k++)
    {
        for (w = 0; w < n_word_types; w++)
        {
            i_idx = opt_words.i_idx[w];
            j_idx = opt_words.j_idx[w];
            counts = opt_words.transition_counts[w];
            length = opt_words.n_unique_transitions[w];

            log_prob = log_init[k][opt_words.first_symbols[w]];
            for (s = 0; s < length; s++){
                i = i_idx[s];
                j = j_idx[s];
                log_prob += counts[s] * log_transitions[k][i][j];
            }

            if (learn_halt){
                log_prob += log_halt[k][opt_words.last_symbols[w]];
            }

            log_prob_w[k][w] = log_prob;
        }

        // If not learning halting probabilities, set the halting probability to 0.
        if (!learn_halt){
            for (s = 0; s < n_symbols; s++){
                log_halt[k][s] = -100;
            }
        }
    }

    if (_estimate_alpha and estimate_alpha){
        alpha = opt_alpha(ss.alpha_suffstats, n_docs, n_topics);
    }
}

void MarkovLDA::save(const string& name){
    string model_name(directory + "/" + name + ".model");
    ofstream mf(model_name);
    mf << "n_word_types " << n_word_types << endl;
    mf << "n_topics " << n_topics << endl;
    mf << "n_symbols " << n_symbols << endl;
    mf << "learn_halt " << learn_halt << endl;
    mf << "alpha " << alpha << endl;

    for(int k = 0; k < n_topics; k++)
    {
        for (int i = 0; i < n_symbols; i++){
            mf << log_init[k][i] << ",";
        }
        mf << endl;
        for (int i = 0; i < n_symbols; i++) {
            for (int j = 0 ; j < n_symbols; j++){
                mf << log_transitions[k][i][j] << ",";
            }
            mf << endl;
        }
        for (int i = 0; i < n_symbols; i++){
            mf << log_halt[k][i] << ",";
        }
        mf << endl;
    }

    string gamma_name(directory + "/" + name + ".gamma");
    ofstream gf(gamma_name);
    gf << "n_docs " << n_docs << endl;

    for (int d = 0; d < n_docs; d++){
        for (int k = 0; k < n_topics; k++){
            gf << gamma[d][k] << ",";
        }
        gf << endl;
    }
    gf.close();
}

OptWords::OptWords(const Dictionary& dict)
{
    n_word_types = dict.n_word_types;
    n_symbols = dict.n_symbols;

    int w, s, pre, post, n_unique;
    vector<int> counts(n_symbols*n_symbols, 0.0);

    i_idx.resize(n_word_types);
    j_idx.resize(n_word_types);
    transition_counts.resize(n_word_types);

    n_unique_transitions.resize(n_word_types);
    first_symbols.resize(n_word_types);
    last_symbols.resize(n_word_types);

    w = 0;
    for (const Word& word: dict.get_words()){
        fill(counts.begin(), counts.end(), 0.0);
        first_symbols[w] = word.front();
        last_symbols[w] = word.back();

        n_unique = 0;
        for (s = 0; s < word.size() - 1; s++)
        {
            pre = word[s];
            post = word[s+1];
            if(counts[post + n_symbols * pre] == 0){
                n_unique++;
            }

            counts[post + n_symbols * pre] += 1;
        }
        n_unique_transitions[w] = n_unique;

        i_idx[w].resize(n_unique);
        j_idx[w].resize(n_unique);
        transition_counts[w].resize(n_unique);

        s = 0;
        for (pre = 0; pre < n_symbols; pre++){

            for (post = 0; post < n_symbols; post++){

                if(counts[post + n_symbols * pre] > 0){
                    i_idx[w][s] = pre;
                    j_idx[w][s] = post;
                    transition_counts[w][s] = counts[post + n_symbols * pre];
                    s++;
                }
            }
        }

        w++;
    }
}
