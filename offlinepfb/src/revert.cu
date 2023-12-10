#include "../include/revert.cuh"
#include <vector>
using std::vector;

box::box(int a, int b, int c, int d, int e)
    : start_time{a}, stop_time{b}, start_chann{c}, stop_chann{d}, box_id{e}
{
}

box::box()
    : box(0, 0, 0, 0, 0) {}

synthesizer::synthesizer(int chann, int tap, int slice)
    : nchannel{chann}, ntaps{tap}, nslice{slice}
{
    small_plans = new cufftHandle[chann / 2 - 2];
    large_plans = new cufftHandle[36];
    int istride = nslice;
    int ostride = nslice;
    int idist = 1;
    int odist = 1;
    vector<int> channel_vec{chann / 32, chann / 16, chann / 8, chann / 4, chann / 2, chann};
    vector<int> slice_vec{slice / 32, slice / 16, slice / 8, slice / 4, slice / 2, slice};
    for (int i = 2; i < chann / 2; i++)
    {
        cufftPlanMany(&small_plans[i], 1, &i, &i, istride, idist, &i, ostride, odist, CUFFT_C2C, 1);
    }

    for (int chann_dim = 0; chann_dim < 6; chann_dim++)
    {
        for (int slice_dim = 0; slice_dim < 6; slice_dim++)
        {
            cufftPlanMany(&large_plans[6 * chann_dim + slice_dim], 1, &channel_vec[chann_dim], &channel_vec[chann_dim], istride, idist, &channel_vec[chann_dim], ostride, odist, CUFFT_C2C, slice_vec[slice_dim]);
        }
    }
}

synthesizer::~synthesizer()
{
    for (int ind = 0; ind < nchannel / 2 - 1; ind++)
    {
        cufftDestroy(small_plans[ind]);
    }
    for (int ind = 0; ind < 36; ind++)
    {
        cufftDestroy(large_plans[ind]);
    }
    delete [] small_plans;
    delete [] large_plans;
}

float __device__ filter_value(int index, int nchannel, int taps)
{
    return cyl_bessel_i0f(static_cast<float>(index));
}

void synthesizer::revert(cufftComplex *input, box *Box, cufftHandle *plan, cufftComplex *scratch, cufftComplex *output, int taps, int nboxes)
{
    for (int boxind=0; boxind < nboxes; boxind++)
    {
        auto curr_box = Box[boxind];

        auto start_channel = curr_box.start_chann;
        auto end_channel   = curr_box.stop_chann;
        auto start_time    = curr_box.start_time;
        auto end_time      = curr_box.stop_time;

        int scratch_start_chann = start_channel;
        if (start_channel == 0)
        {
            scratch_start_chann = 1;
        }
        cudaMemcpy2D(scratch + nslice*scratch_start_chann, nslice, input+start_channel*nslice, nslice, (end_time - start_time)*sizeof(cufftComplex), end_channel - start_channel, cudaMemcpyDeviceToDevice);
    }
}

void __global__ synthesize(cufftComplex *input, box *Box, cufftComplex *output, int taps)
{
    int inp_chann_id    = blockDim.z * blockIdx.z + threadIdx.z;
    int inp_slice_id    = blockDim.x * blockIdx.x + threadIdx.x;
    int outp_slice_id   = blockDim.y * blockIdx.y + threadIdx.y;

    int nchannel = Box->stop_chann - Box->start_chann;
    int nslice   = Box->stop_time  - Box->start_time;

    if (inp_slice_id <= outp_slice_id)
    {
        atomicAdd(&output[outp_slice_id].x, filter_value(outp_slice_id - inp_slice_id, nchannel, taps) * input[inp_slice_id].x);
        atomicAdd(&output[outp_slice_id].y, filter_value(outp_slice_id - inp_slice_id, nchannel, taps) * input[inp_slice_id].y);
    }
}