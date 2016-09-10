#pragma once

#include <cmath>
#include <iostream>

#include "utils.hpp"

using namespace std;

#define NEWTON_THRESH 1e-5
#define MAX_ALPHA_ITER 1000

double alhood(double a, double ss, int D, int K);
double d_alhood(double a, double ss, int D, int K);
double d2_alhood(double a, int D, int K);

double opt_alpha(double ss, int D, int K);
