#ifndef _CCHANNELIZER_
#define _CCHANNELIZER_
#include "offline_channelizer.cuh"

struct chann;
typedef struct chann chann;

extern "C"
{
    chann* chann_create(complex<float>*);
    void chann_destroy(chann*);
    void chann_process(chann*, float*, complex<float>*);
    complex<float>* memory_allocate(int);
}

#endif