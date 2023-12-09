#include "../include/revert.cuh"

box::box(int a, int b, int c, int d, int e)
    : start_time{a}, stop_time{b}, start_chann{c}, stop_chann{d}, box_id{e}
{
}

box::box()
    : box(0, 0, 0, 0, 0) {}

// void __global__ test(box* inp)
// {
//     int id = blockDim.x*blockIdx.x + threadIdx.x;
//     inp[id].start_time += 1;
//     inp[id].stop_time  += 2;
//     inp[id].start_chann += 3;
//     inp[id].stop_chann  += 4;
// }

synthesizer::synthesizer(int chann, int tap, int slice)
: nchannel{chann}, ntaps{tap}, nslice{slice}
{
    plans = new cufftHandle [chann / 2];
    int istride = nslice;
    int ostride = nslice;
    int idist = 1;
    int odist = 1;

    for (int i=0; i<chann / 2; i++)
    {
        cufftPlanMany(&plans[i], 1, &i, &i, istride, idist, &i, ostride, odist, CUFFT_C2C, 1);
    }
}

synthesizer::~synthesizer()
{
    for (int ind=0; ind < nchannel / 2; ind++)
    {
        cufftDestroy(plans[ind]);
    }
    delete [] plans;
}

float __device__ filter_value(int index)
{
    return 0.0;
}

void synthesizer::revert(cufftComplex *input, box *Box, cufftHandle* plan, cufftComplex* scratch, cufftComplex *output, int taps)
{
    
}

void __global__ sythesize(cufftComplex *input, box *Box, cufftHandle* plan, cufftComplex* scratch, cufftComplex *output, int taps)
{
    int inp_chann_id    = blockDim.z * blockIdx.z + threadIdx.z;
    int inp_slice_id    = blockDim.x * blockIdx.x + threadIdx.x;
    int outp_slice_id   = blockDim.y * blockIdx.y + threadIdx.y;

    if (inp_slice_id <= outp_slice_id)
    {
        atomicAdd(&output[outp_slice_id].x, filter_value(outp_slice_id - inp_slice_id) * input[inp_slice_id].x);
        atomicAdd(&output[outp_slice_id].y, filter_value(outp_slice_id - inp_slice_id) * input[inp_slice_id].y);
    }
}