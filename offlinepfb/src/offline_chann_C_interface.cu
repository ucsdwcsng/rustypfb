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

    void chann_process(chann* chann, float* input, cufftComplex* rhs)
    {
        reinterpret_cast<channelizer*>(chann)->process(input, rhs);
    }

    void chann_set_revert_filter(chann* chann, complex<float>* filt)
    {
        reinterpret_cast<channelizer*>(chann)->set_revert_filter(filt);
    }

    void chann_revert(chann* chann, cufftComplex* input, cufftComplex* output)
    {
        reinterpret_cast<channelizer*>(chann)->revert(input, output);
    }

    // cufftComplex* memory_allocate_device(int size)
    // {
    //     // printf("Memory is getting allocated in C\n");
    //     cufftComplex* output;
    //     cudaMalloc((void**)&output, sizeof(cufftComplex)*size);
    //     return output;
    // }

    // void memory_deallocate_device(cufftComplex* inp)
    // {
    //     cudaFree(inp);
    // }

    // complex<float>* memory_allocate_cpu(int size)
    // {
    //     complex<float>* output = new complex<float> [size];
    //     return output;
    // }

    // void memory_deallocate_cpu(complex<float>* inp)
    // {
    //     delete [] inp;
    // }

    // float bessel_func(float x)
    // {
    //     return cyl_bessel_if(0.0, x);
    // }

    // void transfer(cufftComplex* in, cufftComplex* out, int size)
    // {
    //     cudaMemcpy(out, in, sizeof(cufftComplex)*size, cudaMemcpyDeviceToHost);
    // }

    synth* synth_create(int chann, int tap, int slice)
    {   
        return reinterpret_cast<synth*>(new synthesizer(chann, tap, slice));
    }

    void synth_destroy(synth* inp)
    {
        delete reinterpret_cast<synthesizer*>(inp);
    }

    // c_box* box_create(int a, int b, int c, int d, int e)
    // {
    //     return reinterpret_cast<c_box*>(new box(a, b, c, d, e));
    // }

    // void synth_revert(synth* inp, cufftComplex* input, box* Box, cufftComplex* scratch, cufftComplex* output, int taps, int nboxes)
    // {
    //     reinterpret_cast<synthesizer*>(inp)->revert(input, Box, scratch, output, nboxes);
    // }
}

    
