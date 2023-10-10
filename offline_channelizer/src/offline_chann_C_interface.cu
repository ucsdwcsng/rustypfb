#include "../include/offline_chann_C_interface.cuh"
#include <stdio.h>
#include <cmath>
using std::cyl_bessel_if;

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

    void chann_process(chann* chann, float* lhs, complex<float>* rhs)
    {
        reinterpret_cast<channelizer*>(chann)->process(lhs, rhs);
    }

    complex<float>* memory_allocate(int size)
    {
        // printf("Memory is getting allocated in C\n");
        complex<float>* output;
        cudaMalloc((void**)&output, sizeof(complex<float>)*size);
        return output;
    }

    void memory_deallocate(complex<float>* inp)
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

    void transfer(complex<float>* in, complex<float>* out, int size)
    {
        cudaMemcpy(out, in, sizeof(complex<float>)*size, cudaMemcpyDeviceToHost);
    }
}