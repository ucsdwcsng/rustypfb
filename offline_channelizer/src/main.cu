#include "../include/cinterface.cuh"
#include "../include/offline_channelizer.cuh"
#include <stdio.h>
#include <cmath>
#include <complex>
using namespace std::complex_literals;

using std::complex;

int main()
{
    complex<float>* filter_coeff = new complex<float> [10];
    filter_coeff[0] = 2.022 * 1e-20;
    filter_coeff[1] = 9.462 * 1e-3;
    filter_coeff[2] = 1.23 * 1e-1;
    filter_coeff[3] = 4.92 * 1e-1;
    filter_coeff[4] = 9.26 * 1e-1;
    filter_coeff[5] = 9.26 * 1e-1;
    filter_coeff[6] = 4.92 * 1e-1;
    filter_coeff[7] = 1.23 * 1e-1;
    filter_coeff[8] = 9.46 * 1e-3;
    filter_coeff[9] = 2.02 * 1e-20;
    chann* p_chann = chann_create(5, 4, 2, filter_coeff);

    complex<float>* input = new complex<float>[20]{static_cast<complex<float>>(1.0 + 2.0i), 
    static_cast<complex<float>>(1.0 + 2.0i),
    static_cast<complex<float>>(1.0 + 2.0i),
    static_cast<complex<float>>(1.0 + 2.0i),
    static_cast<complex<float>>(1.0 + 2.0i),
    static_cast<complex<float>>(1.0 + 2.0i),
    static_cast<complex<float>>(1.0 + 2.0i),
    static_cast<complex<float>>(1.0 + 2.0i),
    static_cast<complex<float>>(1.0 + 2.0i),
    static_cast<complex<float>>(1.0 + 2.0i),
    static_cast<complex<float>>(1.0 + 2.0i),
    static_cast<complex<float>>(1.0 + 2.0i),
    static_cast<complex<float>>(1.0 + 2.0i),
    static_cast<complex<float>>(1.0 + 2.0i),
    static_cast<complex<float>>(1.0 + 2.0i),
    static_cast<complex<float>>(1.0 + 2.0i),
    static_cast<complex<float>>(1.0 + 2.0i),
    static_cast<complex<float>>(1.0 + 2.0i),
    static_cast<complex<float>>(1.0 + 2.0i)
    };
    cufftComplex* output = new cufftComplex [20];

    chann_process(p_chann, input, output);

    for (int i=0; i<15; i++)
        printf("%f %f\n", output[i].x, output[i].y);

    chann_destroy(p_chann);
    delete [] filter_coeff;
    delete [] input;
    delete [] output;
}