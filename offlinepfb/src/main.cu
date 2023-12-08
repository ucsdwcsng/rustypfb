#include "../include/offline_channelizer.cuh"
#include "../include/offline_chann_C_interface.cuh"
#include "../include/revert.cuh"
// #include "/opt/asmlib/asmlib.h"
// #include <string.h>
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

void time_test(chann* p_chann, float* input, cufftComplex* output, int ntimes, float &time)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float duration;
    cudaEventRecord(start);
    for (int i=0; i < ntimes; i++)
    {
        chann_process(p_chann, input, output);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&duration, start, stop);
    time += duration;
}

int main()
{
    // int Nsamples = 100000000;
    // const int Nch   = 1024;
    // const int Nslice = 2*1024*128;
    // int Nproto = 100;
    // float kbeta=9.6;
    // vector<complex<float>> filter_function;
    // for (int j=0; j<Nch*Nproto; j++)
    // {
    //     float arg = Nproto / 2 + static_cast<float>(j + 1) / Nch;
    //     float darg = static_cast<float>(2 * j) / static_cast<float>(Nch*Nproto) - 1.0;
    //     float carg = kbeta * sqrt(1-darg*darg);
    //     try{
    //     float earg = cyl_bessel_if(0.0, carg) / cyl_bessel_if(0.0, kbeta);
    //     filter_function.push_back(complex<float>(earg, 0.0));
    //     }
    //     catch(int num)
    //     {
    //         cout << "Exception occured " << j << endl;
    //     }
    // }
    // chann* p_chann = chann_create(&filter_function[0], Nproto, Nch, Nslice);
    // float* input = new float [Nch*(Nslice)];
    // // cufftComplex* inp_c = new cufftComplex [Nch * Nslice / 2];
    // cufftComplex* output_gpu;
    // cufftComplex* output_cpu;
    // output_cpu = new cufftComplex [Nch*Nslice];
    // cudaMalloc((void **)&output_gpu, sizeof(cufftComplex) * Nch * Nslice);
    // cudaHostRegister(input, sizeof(float)*Nch*Nslice, cudaHostRegisterMapped);
    // for (int k=0; k<2*Nsamples; k++)
    // {
    //     float inp_arg = static_cast<float>(k / 2);
    //     if (k%2 == 0)
    //     {
    //         input[k] = sin(inp_arg);
    //     }
    //     else 
    //     {
    //         input[k] = sinc(2.0*inp_arg);
    //     }
    // }
    // cout << "---------------------------------------" << endl;
    // float time;
    // time_test(p_chann, input, output_gpu, 50, time);
    // cout << "Channelization of " << Nsamples << " into 1024 channels takes " << time / 50 << " in milliseconds" << endl; 
    // chann_destroy(p_chann);
    // delete [] input;
    // delete [] output_cpu;
    // cudaFree(output_gpu);

    // int Nproto = 128;
    // float kbeta = 10.2;
    // int Nch = 1024;
    // auto now = steady_clock::now();
    // for (int i=0; i<100; i++)
    // {
    //     float arg = Nproto / 2 + static_cast<float>(i + 1) / Nch;
    //     float darg = static_cast<float>(2 * i) / static_cast<float>(Nch*Nproto) - 1.0;
    //     float carg = kbeta * sqrt(1-darg*darg);
    //     float earg = cyl_bessel_if(0.0, carg) / cyl_bessel_if(0.0, kbeta);
    // }
    // auto end = steady_clock::now();
    // auto elapsed_time = duration<float, milli>(end - now).count();

    // cout << elapsed_time / 100 << endl;

    // auto box_collection = new box [16];

    // box* box_gpu;
    // cudaMalloc((void**)&box_gpu, sizeof(box)*16);

    // cudaMemcpy(box_gpu, box_collection, sizeof(box)*16, cudaMemcpyHostToDevice);

    // test<<<4, 4>>>(box_gpu);
    // cudaMemcpy(box_collection, box_gpu, sizeof(box)*16, cudaMemcpyDeviceToHost);

    // for (int i=0; i< 16; i++)
    // {
    //     cout << box_collection[i].start_time << " " << box_collection[i].start_chann << " " << box_collection[i].stop_time << " " << box_collection[i].stop_chann << endl;
    // }

    cufftHandle* plans;
    plans = new cufftHandle [100];
    auto istrides = new int [100];
    auto ostrides = new int [100];
    auto idists   = new int [100];
    auto odists   = new int [100];
    auto batches  = new int [100];
    auto n = new int [100];
    
    for (int i=0; i < 100; i++)
    {
        istrides[i] = 2*i;
        ostrides[i] = 2*i;
        idists[i] = 4*i;
        odists[i] = 4*i;
        batches[i] = 1;
        n[i] = 2*(i%10+1);
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float my_duration;
    cudaEventRecord(start);
    for (int i=0; i<100; i++)
    {
        cufftPlanMany(&plans[i], 1, n + i, n+i, istrides[i], idists[i], n+i, ostrides[i], odists[i], CUFFT_C2C, batches[i]);
    }

    for (int i=0; i<100; i++)
    {
        cufftDestroy(plans[i]);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&my_duration, start, stop);

    cout << "Initializing 100 plans and destroying them takes " << my_duration / 100 << " in milliseconds" << endl;
    
    delete [] plans;
    delete [] istrides;
    delete [] ostrides;
    delete [] idists;
    delete [] odists;
    delete [] batches;
    delete [] n;

}