#ifndef _CCHANNELIZER_
#define _CCHANNELIZER_
#include "offline_channelizer.cuh"
#include "cufft.h"

struct chann;
typedef struct chann chann;

extern "C"
{
    chann* chann_create(complex<float>*, int, int, int);
    void chann_destroy(chann*);
    // void chann_process(chann*, float*, float*, complex<float>*);
    void chann_process(chann*, float*, cufftComplex*);
    cufftComplex* memory_allocate_device(int);
    void memory_deallocate_device(cufftComplex*);
    complex<float>* memory_allocate_cpu(int);
    void memory_deallocate_cpu(complex<float>*);
    float bessel_func(float);
    void transfer(cufftComplex*, cufftComplex*, int);
}

#endif