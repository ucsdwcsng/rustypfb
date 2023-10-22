// #include "../include/cinterface.cuh"
#include "../include/offline_channelizer.cuh"
#include "../include/offline_chann_C_interface.cuh"
#include <stdio.h>
#include <cmath>
#include <complex>
#include <chrono>
#include <iostream>
using namespace std::complex_literals;
using std::chrono::high_resolution_clock;
using std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::cyl_bessel_if;
using std::cout;
using std::endl;
using std::milli;
using std::complex;

float sinc(float x)
{
    return (x == 0.0) ? 1.0 : float(sin(x)/x);
}

int main()
{
    int Nsamples = 100000000;
    int Nch   = 1024;
    int Nslice = 2*1024*128;
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
    }
    chann* p_chann = chann_create(&filter_function[0], Nproto, Nch, Nslice);
    float* input = new float [Nch*(Nslice)];
    cufftComplex* inp_c = new cufftComplex [Nch * Nslice / 2];
    cufftComplex* output_gpu;
    cufftComplex* output_cpu;
    output_cpu = new cufftComplex [Nch*Nslice];
    cudaMalloc((void **)&output_gpu, sizeof(cufftComplex) * Nch * Nslice);
    for (int k=0; k<2*Nsamples; k++)
    {
        float inp_arg = static_cast<float>(k / 2);
        if (k%2 == 0)
        {
            input[k] = sin(inp_arg);
        }
        else 
        {
            input[k] = sinc(2.0*inp_arg);
        }
    }
    cout << "---------------------------------------" << endl;
    float total_duration = 0.0;
    float total_duration_cpy = 0.0;
    int ntimes = 100;
    float milliseconds;
    for (int i=0; i<10;i++)
    {
        chann_process(p_chann, input, output_gpu, i);
        total_duration += reinterpret_cast<channelizer*>(p_chann)->time;
        total_duration_cpy += reinterpret_cast<channelizer*>(p_chann)->time_cpy;
    }
    transfer(output_gpu, output_cpu, 10);
    //Check that results are legitimate.
    for (int i=0; i< 10; i++)
    {
        cout << output_cpu[i].x << " " << output_cpu[i].y << endl;
    }
    std::cout << "Time taken in milliseconds to process " << Nsamples <<" samples into 1024 channels is " << (total_duration / 10) << " and time taken to copy is " << (total_duration_cpy / 10) << std::endl;
    chann_destroy(p_chann);
    delete [] input;
    delete [] output_cpu;
    cudaFree(output_gpu);
}