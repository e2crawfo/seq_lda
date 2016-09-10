#include "lda.hpp"

ostream LOG(cout.rdbuf());

int VAR_MAX_ITER = 0;
double VAR_CONVERGED = 0.0;

int EM_MAX_ITER = 0;
double EM_CONVERGED = 0.0;

double INITIAL_ALPHA = 1.0;

double LDA::e_step(const Corpus& corpus){
    double total_likelihood = 0.0;
    ss.init_zero();
    int d = 0;

    for (auto& doc: corpus.docs){
        // posterior inference
        likelihood[d] = lda_inference(doc, *this, gamma[d], phi[d]);
        total_likelihood += likelihood[d];

        // update sufficient statistics
        double gamma_sum = 0;
        for (int k = 0; k < n_topics; k++)
        {
            gamma_sum += gamma[d][k];
            ss.alpha_suffstats += digamma(gamma[d][k]);
        }
        ss.alpha_suffstats -= n_topics * digamma(gamma_sum);

        for (int k = 0; k < n_topics; k++){
            for (int n = 0; n < doc.n_word_types; n++){
                ss.class_word[k][doc.word_indices[n]] += doc.word_counts[n]*phi[d][n][k];
                ss.class_total[k] += doc.word_counts[n]*phi[d][n][k];
            }
        }

        d++;
    }

    ss.n_docs = n_docs;

    return(total_likelihood);
}

void LDA::m_step(bool _estimate_alpha){
    for (int k = 0; k < n_topics; k++){
        double log_class_total = log(ss.class_total[k]);

        for (int w = 0; w < n_word_types; w++){
            log_prob_w[k][w] = normalized_log(ss.class_word[k][w], log_class_total);
        }
    }

    if (_estimate_alpha and estimate_alpha){
        alpha = opt_alpha(ss.alpha_suffstats, n_docs, n_topics);
    }
}

void LDA::allocate_storage(const Corpus& corpus){
    n_docs = corpus.n_docs;
    n_word_types = corpus.n_word_types;

    ss = LdaSuffstats(n_topics, n_word_types);

    log_prob_w.resize(n_topics, vector<double>(corpus.n_word_types, 0.0));

    likelihood.resize(corpus.n_docs, 0.0);
    gamma.resize(corpus.n_docs, vector<double>(n_topics, 0.0));

    phi.reserve(n_docs);
    for(const Document& doc: corpus.docs){
        phi.push_back(
            vector<vector<double>>(
                doc.n_word_types, vector<double>(n_topics, 0.0)));
    }
}

void LDA::fit(const Corpus& corpus){
    LOG << "Reading settings in " << settings << "..." << endl;
    read_settings(settings);

    LOG << "Allocating storage for model..." << endl;
    allocate_storage(corpus);

    LOG << "Initializing model..." << endl;
    if (start.compare("seeded")==0) {
        // TODO - use init-from-corpus
        ss.init_zero();
        m_step(false);
    }else if (start.compare("random")==0) {
        ss.init_random();
        m_step(false);
    }else {
        //load(start);
    }

    LOG << "Fitting model..." << endl;
    // run expectation maximization
    int i = 0;
    double likelihood, old_likelihood = 0, converged = 1000;

    vector<vector<double>> old_log_prob_w, old_gamma;
    double old_alpha;

    while ((converged > EM_CONVERGED) && (i <= EM_MAX_ITER)) {
        old_gamma = gamma;
        old_likelihood = likelihood;
        likelihood = e_step(corpus);

        old_log_prob_w = log_prob_w;
        old_alpha = alpha;
        m_step();

        // check for convergence
        converged = (old_likelihood - likelihood) / (old_likelihood);
        // if (converged < 0){
        //     VAR_MAX_ITER = VAR_MAX_ITER * 2;
        // }

        LOG << "Likelihood: " << likelihood
             << ", converged: " << converged
             << ", iteration: " << i << endl;
        i++;
    }

    LOG << "Fit complete." << endl;
    if(converged < 0){
        LOG << "Rolling back..." << endl;
        gamma = old_gamma;
        likelihood = old_likelihood;
        log_prob_w = old_log_prob_w;
        alpha = old_alpha;
    }

    LOG << "Saving results..." << endl;
    save("final");
}

void LDA::save(const string& name){
    string model_name(directory + "/" + name + ".model");
    ofstream mf(model_name);

    mf << "n_topics " << n_topics << endl;
    mf << "n_word_types " << n_word_types << endl;
    mf << "alpha " << alpha << endl;

    for (int k = 0; k < n_topics; k++){
        for (int w = 0; w < n_word_types; w++){
            mf << log_prob_w[k][w] << ",";
        }
        mf << endl;
    }
    mf.close();

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

string LDA::to_string() const{
    stringstream stream;
    stream << "LDA(" << endl;
    stream << "    n_topics=" << n_topics << endl;
    stream << "    estimate_alpha=" << estimate_alpha << endl;
    stream << "    start=" << start << endl;
    stream << "    directory=" << directory << endl;
    stream << "    n_docs=" << n_docs << endl;
    stream << "    n_word_types=" << n_word_types << endl;
    stream << "    alpha=" << alpha << endl;
    stream << "    likelihood=" << endl;
    stream << "    ";
    for(int d = 0; d < n_docs; d++){
        stream << likelihood[d] << ",";
    }
    stream << endl;
    stream << "    log_prob_w=" << endl;
    for(int k = 0; k < n_topics; k++){
        stream << "    ";
        for(int w = 0; w < n_word_types; w++){
            stream << log_prob_w[k][w] << ",";
        }
        stream << endl;
    }
    stream << "    gamma=" << endl;
    for(int d = 0; d < n_topics; d++){
        stream << "    ";
        for(int k = 0; k < n_word_types; k++){
            stream << gamma[d][k] << ",";
        }
        stream << endl;
    }
    stream << ")" << endl;

    return stream.str();
}

void LDA::read_settings(const string& filename)
{
    FILE* fileptr = fopen(filename.c_str(), "r");

    safe_fscanf(fileptr, "var max iter %d\n", &VAR_MAX_ITER);
    safe_fscanf(fileptr, "var convergence %lf\n", &VAR_CONVERGED);
    safe_fscanf(fileptr, "em max iter %d\n", &EM_MAX_ITER);
    safe_fscanf(fileptr, "em convergence %lf\n", &EM_CONVERGED);
    fclose(fileptr);
}