// (C) Copyright 2004, David M. Blei (blei [at] cs [dot] cmu [dot] edu)

// This file is part of LDA-C.

// LDA-C is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your
// option) any later version.

// LDA-C is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// for more details.

// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA

#include "variational_inference.hpp"

/*
 * variational inference
 *
 */

double lda_inference(
        const Document& doc, const LDA& model,
        vector<double>& gamma, vector<vector<double>>& phi){

    double phi_sum = 0, likelihood = 0;
    double likelihood_old = 0, old_phi[model.n_topics];
    double digamma_gam[model.n_topics];

    // compute posterior dirichlet
    for (int k = 0; k < model.n_topics; k++){
        gamma[k] = model.alpha + (doc.n_word_tokens/((double) model.n_topics));
        digamma_gam[k] = digamma(gamma[k]);
        for (int n = 0; n < doc.n_word_types; n++){
            phi[n][k] = 1.0/model.n_topics;
        }
    }

    int var_iter = 0;
    double converged = 1;

    while ((converged > VAR_CONVERGED) &&
           ((var_iter < VAR_MAX_ITER) || (VAR_MAX_ITER == -1))){
        var_iter++;
        for (int n = 0; n < doc.n_word_types; n++){
            phi_sum = 0;
            for (int k = 0; k < model.n_topics; k++){
                old_phi[k] = phi[n][k];
                phi[n][k] = digamma_gam[k] + model.log_prob_w[k][doc.word_indices[n]];

                if (k > 0){
                    phi_sum = log_sum(phi_sum, phi[n][k]);
                }else{
                    phi_sum = phi[n][k]; // note, phi is in log space
                }
            }

            for (int k = 0; k < model.n_topics; k++){
                phi[n][k] = exp(phi[n][k] - phi_sum);
                gamma[k] = gamma[k] + doc.word_counts[n]*(phi[n][k] - old_phi[k]);
                digamma_gam[k] = digamma(gamma[k]);
            }
        }

        likelihood = compute_likelihood(doc, model, gamma, phi);
        assert(!std::isnan(likelihood));
        converged = (likelihood_old - likelihood) / likelihood_old;
        likelihood_old = likelihood;
    }

    return(likelihood);
}


/*
 * compute likelihood bound
 *
 */

double compute_likelihood(
        const Document& doc, const LDA& model,
        const vector<double>& gamma, const vector<vector<double>>& phi){

    double likelihood = 0, digsum = 0, gamma_sum = 0, dig[model.n_topics];

    for (int k = 0; k < model.n_topics; k++){
        dig[k] = digamma(gamma[k]);
        gamma_sum += gamma[k];
    }
    digsum = digamma(gamma_sum);

    likelihood =
        lgamma(model.alpha * model . n_topics)
        - model.n_topics * lgamma(model.alpha)
        - (lgamma(gamma_sum));

    for (int k = 0; k < model.n_topics; k++){
        likelihood +=
            (model.alpha - 1)*(dig[k] - digsum) + lgamma(gamma[k])
            - (gamma[k] - 1)*(dig[k] - digsum);

        for (int n = 0; n < doc.n_word_types; n++){
            if (phi[n][k] > 0){
                likelihood += doc.word_counts[n]*
                    (phi[n][k]*((dig[k] - digsum) - log(phi[n][k])
                                + model.log_prob_w[k][doc.word_indices[n]]));
            }
        }
    }

    return(likelihood);
}
