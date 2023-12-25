#include "../include/revert.cuh"
#include <vector>
#include <cmath>
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
    input_plans = new cufftHandle[36];
    downconvert_plans = new cufftHandle[36];
    int istride = nslice;
    int ostride = nslice;
    int idist = 1;
    int odist = 1;
    vector<int> channel_vec{chann / 8, chann / 4, chann / 2};
    vector<int> slice_vec{slice / 8, slice / 4, slice / 2};
    for (int chann_dim = 0; chann_dim < 6; chann_dim++)
    {
        for (int slice_dim = 0; slice_dim < 6; slice_dim++)
        {
            cufftPlanMany(&input_plans[6 * chann_dim + slice_dim], 1, &slice_vec[slice_dim], &slice_vec[slice_dim], 1, 
            slice_vec[slice_dim], &slice_vec[slice_dim], 1, slice_vec[slice_dim], CUFFT_C2C, channel_vec[chann_dim]);
        }
    }
    for (int chann_dim = 0; chann_dim < 6; chann_dim++)
    {
        for (int slice_dim = 0; slice_dim < 6; slice_dim++)
        {
            cufftPlanMany(&downconvert_plans[6 * chann_dim + slice_dim], 1, &channel_vec[chann_dim], &channel_vec[chann_dim], slice_vec[slice_dim], 
            1, &channel_vec[chann_dim], channel_vec[chann_dim], 1, CUFFT_C2C, slice_vec[slice_dim]);
        }
    }

}

synthesizer::~synthesizer()
{
    for (int ind = 0; ind < 36; ind++)
    {
        cufftDestroy(input_plans[ind]);
        cufftDestroy(downconvert_plans[ind]);
    }
    delete[] input_plans;
    delete[] downconvert_plans;
}

float __device__ filter_value(int index, int nchannel, int taps)
{
    return cyl_bessel_i0f(static_cast<float>(index));
}

void synthesizer::revert(cufftComplex *input, box* Box, cufftComplex *scratch, cufftComplex *output, int taps, int nboxes)
{
    // auto start_channel = curr_box.start_chann;
    // auto end_channel = curr_box.stop_chann;
    // auto start_time = curr_box.start_time;
    // auto end_time = curr_box.stop_time;

    // auto area = (end_time - start_time)*(end_channel - start_channel);

    // int padded_channel = (int)(log2(((32 * (end_channel - start_channel)) / nchannel) + 1));
    // int padded_slice = (int)(log2(((32 * (end_time - start_time)) / nslice) + 1));

    // auto full_channel = (int)pow(2, padded_channel);
    // int scratch_start_chann = (full_channel-(end_channel - start_channel)) / 2;

    // cudaMemcpy2D(scratch + nslice * scratch_start_chann, nslice, input + start_channel * nslice, nslice, (end_time - start_time) * sizeof(cufftComplex), end_channel - start_channel, cudaMemcpyDeviceToDevice);
    // cufftExecC2C(large_plans[6*padded_channel + padded_slice], scratch, scratch, CUFFT_FORWARD);
    // synthesize<<<end_time - start_time, area, full_channel>>>(scratch, Box+boxind, output, taps);
}


void __global__ synthesize(cufftComplex *input, box *Box, cufftComplex *output, int taps)
{
    int inp_chann_id = blockDim.z * blockIdx.z + threadIdx.z;
    int inp_slice_id = blockDim.x * blockIdx.x + threadIdx.x;
    int outp_slice_id = blockDim.y * blockIdx.y + threadIdx.y;

    int nchannel = Box->stop_chann - Box->start_chann;
    int nslice = Box->stop_time - Box->start_time;

    if (inp_slice_id <= outp_slice_id)
    {
        atomicAdd(&output[outp_slice_id].x, filter_value(outp_slice_id - inp_slice_id, nchannel, taps) * input[inp_slice_id].x);
        atomicAdd(&output[outp_slice_id].y, filter_value(outp_slice_id - inp_slice_id, nchannel, taps) * input[inp_slice_id].y);
    }
}