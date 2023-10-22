#include "../include/offline_chann_C_interface.cuh"
#include <stdio.h>
#include <cmath>
#include <cstring>
#include <iostream>
using std::cout;
using std::endl;
using std::cyl_bessel_if;
// using std::memcpy;

extern "C"
{
    chann* chann_create(complex<float>* coeff_arr, int nprot, int nchann, int nsl)
    {
        return reinterpret_cast<chann*>(new channelizer(coeff_arr, nprot, nchann, nsl));
    }

    void chann_destroy(chann* inp)
    {
        delete reinterpret_cast<channelizer*>(inp);
    }

    void chann_process(chann* chann, float* input, cufftComplex* rhs, int j)
    {
        // int n = reinterpret_cast<channelizer*>(chann)->nchannel;
        // int s = reinterpret_cast<channelizer*>(chann)->nslice;
        // input[0] = static_cast<float>(j);
        // memcpy(reinterpret_cast<channelizer*>(chann)->input_buffer, input, sizeof(float)*n*s);
        // for(int i=0; i<30;i++)
        // {
        //     cout << (reinterpret_cast<channelizer*>(chann)->input_buffer)[i].x << " " << (reinterpret_cast<channelizer*>(chann)->input_buffer)[i].y << "---------------" << input[2*i] << " " << input[2*i + 1] << endl;
        // }
        // cout << "---------------------------------" << endl;
        reinterpret_cast<channelizer*>(chann)->process(input, rhs);
    }

    cufftComplex* memory_allocate_device(int size)
    {
        // printf("Memory is getting allocated in C\n");
        cufftComplex* output;
        cudaMalloc((void**)&output, sizeof(cufftComplex)*size);
        return output;
    }

    void memory_deallocate_device(cufftComplex* inp)
    {
        cudaFree(inp);
    }

    complex<float>* memory_allocate_cpu(int size)
    {
        complex<float>* output = new complex<float> [size];
        return output;
    }

    void memory_deallocate_cpu(complex<float>* inp)
    {
        delete [] inp;
    }

    float bessel_func(float x)
    {
        return cyl_bessel_if(0.0, x);
    }

    void transfer(cufftComplex* in, cufftComplex* out, int size)
    {
        cudaMemcpy(out, in, sizeof(cufftComplex)*size, cudaMemcpyDeviceToHost);
    }
}