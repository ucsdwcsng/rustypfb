#include "../include/cinterface.cuh"
#include "../include/offline_channelizer.cuh"
#include <stdio.h>
#include <cmath>
#include <complex>
#include <chrono>
#include <iostream>
using namespace std::complex_literals;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::cyl_bessel_if;

using std::complex;

float sinc(float x)
{
    return (x == 0.0) ? 1.0 : float(sin(x)/x);
}

int main()
{
    int Nsamples = 100000000;
    int Nch   = 1024;
    int Nslice = 1024*128;
    int Nproto = 100;
    float kbeta=9.6;
    vector<complex<float>> filter_function;
    for (int i=0; i<Nch*Nproto; i++)
    {
        float arg = Nproto / 2 + static_cast<float>(i + 1) / Nch;
        filter_function.push_back(complex<float>(sinc(arg) * cyl_bessel_if(0.0, kbeta * sqrt(1-pow((static_cast<float>(2 * i) / Nch*Nproto - 1.0), 2.0))) / cyl_bessel_if(0.0, kbeta), 0.0));
    }
    

    // chann* p_chann = chann_create(Nch, Nslice, Nproto, &filter_function[0]);
    auto obj_chann = channelizer(Nch, Nslice, Nproto, &filter_function[0]);

    complex<float>* input = new complex<float>[Nch*Nslice];
    for (int k=0; k<Nsamples; k++)
    {
        complex<float> t(k, 2.0 *k);
        input[k] = t;
    }
    complex<float>* output = new complex<float> [Nch*Nslice];

    double total_duration = 0.0;
    int ntimes = 100;

    for (int i=0; i<ntimes; i++)
    {
        auto start = high_resolution_clock::now();
        obj_chann.process(input, output);
        auto end = high_resolution_clock::now();
        double f = duration<double, std::milli>(end-start).count();
        total_duration += f;
    }
    std::cout << "Time taken in milliseconds to process " << Nsamples <<" samples into 1024 channels is " << (total_duration / ntimes) << std::endl;

    // chann_destroy(p_chann);
    delete [] input;
    delete [] output;
}