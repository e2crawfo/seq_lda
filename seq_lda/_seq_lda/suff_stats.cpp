#include "suff_stats.hpp"


LdaSuffstats::LdaSuffstats(int n_topics, int n_word_types)
:n_docs(0), n_topics(n_topics), n_word_types(n_word_types), dist(0.0, 1.0), alpha_suffstats(0.0){
    rng.seed(1);

    class_word.resize(n_topics);
    class_total.resize(n_topics, 0.0);

    for(int k = 0; k < n_topics; k++){
        class_word[k].resize(n_word_types, 0.0);
    }
}

void LdaSuffstats::init_zero() {
    fill(class_total.begin(), class_total.end(), 0.0);

    for(auto& cw: class_word){
        fill(cw.begin(), cw.end(), 0.0);
    }

    n_docs = 0;
    alpha_suffstats = 0;
}

void LdaSuffstats::init_random() {
    for (int k = 0; k < n_topics; k++)
    {
        class_total[k] = 0.0;
        for (int w = 0; w < n_word_types; w++)
        {
            class_word[k][w] = 1.0/n_word_types + dist(rng);
            class_total[k] += class_word[k][w];
        }
    }
}
