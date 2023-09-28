#ifndef _CCHANNELIZER_
#define _CCHANNELIZER_
#include "offline_channelizer.cuh"
#include "cufft.h"

struct chann;
typedef struct chann chann;

extern "C"
{
    chann* chann_create(complex<float>*);
    void chann_destroy(chann*);
    void chann_process(chann*, float*, complex<float>*);
    complex<float>* memory_allocate(int);
    void memory_deallocate(complex<float>*);
    complex<float>* memory_allocate_cpu(int);
    void memory_deallocate_cpu(complex<float>*);
    float bessel_func(float);
    void transfer(complex<float>*, complex<float>*, int);
}

#endif