// #include "../include/cinterface.cuh"
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
using std::cout;
using std::endl;

using std::complex;

float sinc(float x)
{
    return (x == 0.0) ? 1.0 : float(sin(x)/x);
}

int main()
{
    // auto inp = new int [32];
    // auto outp = new int [32];
    // for (int i=0; i<32; i++)
    // {
    //     inp[i] = i;
    // }

    // for (int i=0; i<32; i++)
    // {
    //     cout << outp[i] << endl;
    // }
    // cudaMemcpy2D(outp + 6, 8*sizeof(int), inp+6, 8*sizeof(int), 2*sizeof(int), 4, cudaMemcpyHostToHost);
    // cout << "After strided copying" << endl;
    // for (int i=0; i<32; i++)
    // {
    //     cout << outp[i] << endl;
    // }
    // cudaMemcpy2D(outp + 4, 8*sizeof(int), inp+4, 8*sizeof(int), 2*sizeof(int), 4, cudaMemcpyHostToHost);
    // cout << "After second strided copying" << endl;
    // for (int i=0; i<32; i++)
    // {
    //     cout << outp[i] << endl;
    // }
    int Nsamples = 100000000;
    int Nch   = 1024;
    int Nslice = 1024*128;
    int Nproto = 100;
    float kbeta=9.6;
    vector<complex<float>> filter_function;
    for (int j=0; j<Nch*Nproto; j++)
    {
        float arg = Nproto / 2 + static_cast<float>(j + 1) / Nch;
        float darg = static_cast<float>(2 * j) / static_cast<float>(Nch*Nproto) - 1.0;
        float carg = kbeta * sqrt(1-darg*darg);
        try{
        float earg = cyl_bessel_if(0.0, carg) / cyl_bessel_if(0.0, kbeta);
        filter_function.push_back(complex<float>(earg, 0.0));
        }
        catch(int num)
        {
            cout << "Exception occured " << j << endl;
        }
        // cout << arg << " " << sinc(arg) << " " << darg << " " << carg << " " << earg <<endl;
        // float barg = sinc(arg) * cyl_bessel_if(0.0, ) / cyl_bessel_if(0.0, kbeta);cd
        
    }

    for (int k=0; k< 1280; k++)
    {
        cout << filter_function[k].real() << " " << filter_function[k].imag() << endl;
    }    

    // chann* p_chann = chann_create(Nch, Nslice, Nproto, &filter_function[0]);
    auto obj_chann = channelizer(&filter_function[0]);

    complex<float>* input = new complex<float>[Nch*Nslice];
    for (int k=0; k<Nsamples; k++)
    {
        complex<float> t(sin(k), sinc(2.0 *k));
        input[k] = t;
        // cout << input[k].real() << " " << input[k].imag() << endl;
    }
    // complex<float>* output = new complex<float> [Nch*Nslice];
    // complex<float>* output;
    complex<float>* output = new complex<float>[10];

    double total_duration = 0.0;
    int ntimes = 100;

    for (int i=0; i<ntimes; i++)
    {
        auto start = high_resolution_clock::now();
        obj_chann.process(input);
        auto end = high_resolution_clock::now();
        double f = duration<double, std::milli>(end-start).count();
        cudaMemcpy(output, obj_chann.output_buffer, sizeof(cufftComplex)*10, cudaMemcpyDeviceToHost);
        for (int i=0; i<10; i++)
    {
        cout << output[i].real() << " " << output[i].imag() << endl;  
    }
        total_duration += f;
    }
    std::cout << "Time taken in milliseconds to process " << Nsamples <<" samples into 1024 channels is " << (total_duration / ntimes) << std::endl;

    // for (int i=0; i<1000; i++)
    // {
    //     cout << output[i].real() << " " << output[i].imag() << endl;  
    // }

    // chann_destroy(p_chann);
    delete [] input;
    // delete [] output;
    delete [] output;
}