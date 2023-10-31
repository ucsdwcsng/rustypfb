#include "../include/offline_channelizer.cuh"
#include "../include/offline_chann_C_interface.cuh"
#include "/opt/asmlib/asmlib.h"
#include <stdio.h>
#include <cmath>
#include <complex>
#include <chrono>
#include <iostream>
#include <fstream>
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
using std::ifstream;
using std::ofstream;

float sinc(float x)
{
    return (x == 0.0) ? 1.0 : float(sin(x)/x);
}

void time_test(chann* p_chann, float* input, cufftComplex* output, int ntimes, float &time)
{
    time = 0.0;
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

void write_to_file(cufftComplex* inp, int size)
{
    ofstream myfile;
    myfile.open("../prototype_filter.32cf");
    // for (int j=0; j<size; j++)
    // {
    //     myfile << inp[j].x << "\n";
    //     myfile << inp[j].y << "\n";
    // }
    myfile.write((char*)inp, sizeof(float)*size*2);
    myfile.close();
}

int main()
{
    /*
     * Example 1 
     */
    int Nsamples = 20000000;
    int Nch      = 1024;
    int Nslice   = 2*1024*128;
    ifstream file;
    file.open("../../busyBand/DSSS.32cf");
    float* input_cpu;
    input_cpu = new float [Nch*Nslice];
    file.read((char*) input_cpu, sizeof(float)*Nsamples*2);
    file.close();

    // for (int i=0; i<100;i++)
    // {
    //     cout << input_cpu[i] << endl;
    // }
    // int Nsamples = 1024;
    // const int Nch   = 32;
    // const int Nslice = 32;
    int   Nproto = 128;
    float kbeta  = 10.0;
    vector<complex<float>> filter_function;
    for (int j=0; j<Nch*Nproto; j++)
    {
        float arg  =  - Nproto / 2 + static_cast<float>(j + 1) / Nch;
        float darg = static_cast<float>(2 * j) / static_cast<float>(Nch*Nproto) - 1.0;
        float carg = kbeta * sqrt(1-darg*darg);
        try{
        float earg = cyl_bessel_if(0.0, carg) / cyl_bessel_if(0.0, kbeta) * sinc(arg);
        filter_function.push_back(complex<float>(earg, 0.0));
        }
        catch(int num)
        {
            cout << "Exception occured " << j << endl;
        }
    }

    chann* p_chann = chann_create(&filter_function[0], Nproto, Nch, Nslice);
    // float* input = new float [Nch*(Nslice)];
    cufftComplex* output_gpu;
    cudaMalloc((void **)&output_gpu, sizeof(cufftComplex) * Nch * Nslice);

    float* input_buffer = new float [Nch*Nslice];
    cudaHostRegister(input_buffer, sizeof(float)*Nch*Nslice, cudaHostRegisterMapped);

    for (int i=0; i< 100; i++)
    {
        cout << input_cpu[i] << endl;
    }
    cout << "---------------------" << endl;

    memcpy(input_buffer, input_cpu, sizeof(float)*Nsamples*2);

    // for (int i=0; i< 100; i++)
    // {
    //     cout << input_buffer[i] << endl;
    // }

    cufftComplex* output_cpu;
    output_cpu = new cufftComplex [Nch*Nslice];

    float time;
    time_test(p_chann, input_buffer, output_gpu, 20, time);

    cout << "Time taken to process 2.68 x 10^8 samples is " << (time / 20) << endl;

    // chann_process(p_chann, input_buffer, output_gpu);
    // transfer(output_gpu, output_cpu, Nch*Nslice);

    // cout << "----------------------------" << endl;
    // for (int i=0; i< 100; i++)
    // {
    //     cout << output_cpu[i].x << " " << output_cpu[i].y << endl;
    // }

    // write_to_file(output_cpu, Nch*Nslice);

    chann_destroy(p_chann);

    cudaFree(output_gpu);
    delete [] output_cpu;

    cudaHostUnregister(input_buffer);
    delete [] input_buffer;

    delete [] input_cpu;

    // write_to_file(reinterpret_cast<cufftComplex*>(&filter_function[0]), Nch*Nproto);

    /* Reshape Test*/
    // cufftComplex* test = new cufftComplex [30*10]; 
    // cufftComplex* test_gpu;
    // cufftComplex* test_gpu_output;

    // for (int i=0; i<300; i++)
    // {
    //     test[i] = make_cuComplex(i, i*i);
    // }
    // cufftComplex* test_cpu = new cufftComplex [30*10];

    // cudaMalloc((void**)&test_gpu, sizeof(cufftComplex)*30*10);
    // cudaMalloc((void**)&test_gpu_output, sizeof(cufftComplex)*30*10);

    // cudaMemcpy(test_gpu, test, sizeof(cufftComplex)*300, cudaMemcpyHostToDevice);
    // dim3 block(5,5);
    // dim3 grid(1,12);
    // reshape<<<grid, block>>>(test_gpu, test_gpu_output, 10, 60);
    // auto err = cudaGetLastError();
    // cout << cudaGetErrorString(err) << endl;
    // cudaMemcpy(test_cpu, test_gpu_output, sizeof(cufftComplex)*300, cudaMemcpyDeviceToHost);

    // for (int i=0; i<5; i++)
    // {
    //     cout << "---------------------------" << endl;
    //     for(int j=0; j<60; j++)
    //     {
    //         cout << test_cpu[i*60+j].x << " " << test_cpu[i*60+j].y << endl;
    //     }
    // }

    // delete [] test_cpu;
    // delete [] test;
    // cudaFree(test_gpu);
    // cudaFree(test_gpu_output);
}