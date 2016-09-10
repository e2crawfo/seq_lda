#include "callback_lda.hpp"

void CallbackLDA::m_step(bool _estimate_alpha){
    callback_m_step(callback, class_word, arglist, log_prob_w, ss.class_word);

    if (_estimate_alpha and estimate_alpha){
        alpha = opt_alpha(ss.alpha_suffstats, n_docs, n_topics);
    }
}
