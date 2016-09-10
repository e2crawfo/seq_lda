#pragma once

#include <cmath>
#include <iostream>
#include <sys/stat.h>
#include <stdio.h>
#include <stdarg.h>

double normalized_log(double val, double log_norm);
double log_sum(double log_a, double log_b);
double trigamma(double x);
double digamma(double x);
double log_gamma(double x);
void make_directory(char* name);
int argmax(double* x, int n);

void safe_fscanf(FILE* fileptr, const char* format, ...);
