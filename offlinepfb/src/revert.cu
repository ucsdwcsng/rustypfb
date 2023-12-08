#include "../include/revert.cuh"

box::box(int a, int b, int c, int d, int e)
: start_time{a}, stop_time{b}, start_chann{c}, stop_chann{d}, box_id{e}
{}

box::box()
: box(0, 0, 0, 0, 0){}

void __global__ test(box* inp)
{
    int id = blockDim.x*blockIdx.x + threadIdx.x;
    inp[id].start_time += 1;
    inp[id].stop_time  += 2;
    inp[id].start_chann += 3;
    inp[id].stop_chann  += 4;
}

float __device__ filter_value(int index)
{
    return 0.0;
}

void __global__ revert(cufftComplex* input, box *Box, cufftComplex* output)
{
    int inp_chann_id = blockDim.z * blockIdx.z + threadIdx.z;
    int inp_slice_id = blockDim.x * blockIdx.x + threadIdx.x;
    int outp_slice_id = blockDim.y * blockIdx.y + threadIdx.y;

    // unsigned long long int x, y, z;
    // atomicAdd(&x, y);
    atomicAdd(&output[outp_slice_id].x, filter_value(outp_slice_id - inp_slice_id)*input[inp_slice_id].x);
    atomicAdd(&output[outp_slice_id].y, filter_value(outp_slice_id - inp_slice_id)*input[inp_slice_id].y);
}