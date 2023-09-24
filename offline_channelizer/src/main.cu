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
    }
  
    auto obj_chann = channelizer(&filter_function[0]);
    // complex<float>* input = new complex<float>[Nch*Nslice];
    float* input = new float [Nch*Nslice*2];
    complex<float>* output_gpu;
    cudaMalloc((void **)&output_gpu, sizeof(complex<float>) * Nch * Nslice*2);
    for (int k=0; k<2*Nsamples; k++)
    {
        float complex_id = float(k / 2);
        if (k%2 == 0)
        {
        input[k] = sin(complex_id);
        }
        else
        {
        input[k] = sinc(2.0*complex_id);
        }  
    }
    // for (int k=0; k<Nsamples; k++)
    // {
    //     complex<float> t(sin(k), sinc(2.0*k));
    //     input[k] = t;
    // }

    complex<float>* output_cpu = new complex<float>[10];
    double total_duration = 0.0;
    int ntimes = 100;
    for (int i=0; i<ntimes; i++)
    {
        auto start = high_resolution_clock::now();
        obj_chann.process(input, output_gpu);
        auto end = high_resolution_clock::now();
        double f = duration<double>(end-start).count();
        cudaMemcpy(output_cpu, output_gpu, sizeof(complex<float>)*10, cudaMemcpyDeviceToHost);
        for (int i=0; i<10; i++)
    {
        cout << output_cpu[i].real() << " " << output_cpu[i].imag() << endl;  
    }
        cout << "-----------------------------------------" << endl;
        total_duration += f;
    }
    std::cout << "Time taken in seconds to process " << Nsamples <<" samples into 1024 channels is " << (total_duration / ntimes) << std::endl;
    delete [] input;
    cudaFree(output_gpu);
    delete [] output_cpu;
    // float* input = new float [2*Nsamples];
    // for (int k=0; k<2*Nsamples; k++)
    // {
    //     if (k%2 == 0)
    //     {
    //     input[k] = sin(k);
    //     }
    //     else
    //     {
    //     input[k] = sinc(2.0*k);
    //     }  
    // }
    // float* input_gpu;
    // cudaMalloc((void**)&input_gpu, 2*sizeof(float)*Nsamples);
    // cudaMemcpy(input_gpu, input, sizeof(float)*2*Nsamples, cudaMemcpyHostToDevice);
    // cufftComplex* output;
    // cudaMalloc((void**)&output, sizeof(cufftComplex)*Nsamples);
    // club<<<2*Nslice, Nch>>>(input_gpu, output, 2*Nsamples);
    // cufftComplex* output_cpu = new cufftComplex [10];
    // cudaMemcpy(output_cpu, output, sizeof(cufftComplex)*10, cudaMemcpyDeviceToHost);
    // for (int i=0; i<10; i++)
    // {
    //     cout << output_cpu[i].x << " " << output_cpu[i].y << endl;
    // }
    // delete [] input;
    // cudaFree(input_gpu);
    // cudaFree(output);
    // delete [] output_cpu;
}