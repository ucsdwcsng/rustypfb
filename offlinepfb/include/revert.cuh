#include "offline_channelizer.cuh"
#include <cufft.h>


struct box {
    int start_time;
    int stop_time;
    int start_chann;
    int stop_chann;
    int box_id;

    box(int, int, int, int, int);
    box();
};

void __global__ test(box*);

void __global__ revert(cufftComplex*, box*, cufftComplex*);