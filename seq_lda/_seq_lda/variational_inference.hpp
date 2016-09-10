#pragma once

#include <cmath>
#include <cassert>
#include <vector>

#include "utils.hpp"
#include "words.hpp"
#include "lda.hpp"

double lda_inference(
        const Document& doc, const LDA& model,
        vector<double>& gamma, vector<vector<double>>& phi);
double compute_likelihood(
        const Document& doc, const LDA& model,
        const vector<double>& gamma, const vector<vector<double>>& phi);
