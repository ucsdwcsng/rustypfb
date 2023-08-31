#ifndef _CCHANNELIZER_
#define _CCHANNELIZER_

#include "offline_channelizer.cuh"

struct chann;
typedef struct chann chann;
extern "C"{
chann* chann_create(int, int, int, complex<float>*);
void chann_destroy(chann*);
void chann_process(chann*, complex<float>*, cufftComplex*);
}

#endif
