#ifndef _PFB_
#define _PFB_
#include "offline_channelizer.cuh"
#include "revert.cuh"
#include "cufft.h"

struct chann;
typedef struct chann chann;

struct synth;
typedef struct synth synth;

// struct c_box;
// typedef struct c_box c_box;

extern "C"
{
    chann* chann_create(complex<float>*, int, int, int);
    void chann_destroy(chann*);
    // void chann_process(chann*, float*, float*, complex<float>*);

    void chann_process(chann*, float*, cufftComplex*);
    void chann_set_revert_filter(chann*, complex<float>*);
    void chann_revert(chann*, cufftComplex*, cufftComplex*);

    // cufftComplex* memory_allocate_device(int);
    // void memory_deallocate_device(cufftComplex*);
    // complex<float>* memory_allocate_cpu(int);
    // void memory_deallocate_cpu(complex<float>*);

    // float bessel_func(float);
    // void transfer(cufftComplex*, cufftComplex*, int);

    synth* synth_create(int, int, int);
    void synth_destroy(synth*);

    // c_box* box_create(int, int, int, int, int);

    void synth_revert(synth*, cufftComplex*, box*, cufftComplex*, cufftComplex*, int, int);

}

#endif