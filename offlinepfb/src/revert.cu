#include "../include/revert.cuh"
#include "../include/offline_channelizer.cuh"
#include <vector>
#include <cmath>
using std::vector;
using std::cyl_bessel_if;

void make_synth_coeff_matrix(cufftComplex* gpu, int nproto, int nchannel, int nslice) {
    int nchannelhalf = nchannel / 2;
    for (int id = 0; id < nchannel * nproto; id++)
    {
        int tap_id = id / nchannel;
        int chann_id = id % nchannel;
        float arg = nproto / 2 + static_cast<float>(id + 1) / nchannel;
        float sinc_val = (arg == 0.0 ? 1.0 : sinf(2.0* M_PI * arg) / (2.0*arg));
        float darg = static_cast<float>(2 * id) / static_cast<float>(nchannel*nproto) - 1.0;
        float carg = 10.0 * sqrtf(1-darg*darg);
        float earg = sinc_val* cyl_bessel_if(0.0, carg) / cyl_bessel_if(0.0, 10.0);
        cufftComplex earg_ = make_cuComplex(earg, 0.0);
        if (chann_id < nchannelhalf)
        {
            cudaMemcpy(gpu + 2 * tap_id  + chann_id * nslice, &earg_, sizeof(cufftComplex), cudaMemcpyHostToDevice);
        }
        else 
        {
            cudaMemcpy(gpu + 2 * tap_id + 1 + chann_id * nslice, &earg_, sizeof(cufftComplex), cudaMemcpyHostToDevice);   
        }
    }
    int istride = 1;
    int ostride = 1;
    int idist = nslice;
    int odist = nslice;
    int batch = nchannel;
    int* n = new int [1];
    *n = nslice;
    int* inembed = n;
    int* onembed = n;
    cufftHandle plan;
    cufftPlanMany(&plan, 1, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch);
    cufftExecC2C(plan, gpu, gpu, CUFFT_FORWARD);
    cufftDestroy(plan);
    delete [] n;
}

box::box(int a, int b, int c, int d, int e)
    : time_start{a}, time_stop{b}, chann_start{c}, chann_stop{d}, box_id{e}
{
}

box::box()
    : box(0, 0, 0, 0, 0) {}

synthesizer::synthesizer(int chann, int tap, int slice)
    : nchannel{chann}, ntaps{tap}, nslice{slice}
{
    input_plans = new cufftHandle[25];
    downconvert_plans = new cufftHandle[25];
    filters = new cufftComplex* [25];

    vector<int> channel_vec{chann / 32, chann / 16, chann / 8, chann / 4, chann / 2};
    vector<int> slice_vec{slice / 32, slice / 16, slice / 8, slice / 4, slice / 2};

    for (int chann_dim = 0; chann_dim < 5; chann_dim++)
    {
        for (int slice_dim = 0; slice_dim < 5; slice_dim++)
        {
            cufftPlanMany(&input_plans[5 * chann_dim + slice_dim], 1, &slice_vec[slice_dim], &slice_vec[slice_dim], 1, 
            slice, &slice_vec[slice_dim], 1, slice, CUFFT_C2C, channel_vec[chann_dim]);
            cufftPlanMany(&downconvert_plans[5 * chann_dim + slice_dim], 1, &channel_vec[chann_dim], &channel_vec[chann_dim], slice, 
            1, &channel_vec[chann_dim],slice, 1, CUFFT_C2C, slice_vec[slice_dim]);
            cudaMalloc((void**)&filters[chann_dim * 5 + slice_dim], sizeof(cufftComplex)*channel_vec[chann_dim]*slice_vec[slice_dim]);
            make_synth_coeff_matrix(filters[chann_dim * 5 + slice_dim], ntaps, channel_vec[chann_dim], slice_vec[slice_dim]);
            cufftExecC2C(input_plans[5 * chann_dim + slice_dim], filters[chann_dim * 5 + slice_dim], filters[chann_dim * 5 + slice_dim], CUFFT_FORWARD);
            // cufftPlanMany(plan,)
        }
    }
    cudaMalloc((void**)&scratch_space, sizeof(cufftComplex)*chann*slice);
    cudaMalloc((void**)&multiply_scratch_space, sizeof(cufftComplex)*chann*slice);
    cudaMalloc((void**)&output, sizeof(cufftComplex)*chann*slice);
}

void synthesizer::reconstruct(cufftComplex *input, box Box, cufftComplex *output)
{
    auto start_channel = Box.chann_start;
    auto end_channel = Box.chann_stop;
    auto start_time = Box.time_start;
    auto end_time = Box.time_stop;

    // auto area = (end_time - start_time)*(end_channel - start_channel);
    int padded_channel_factor = (int)(log2(32 * (end_channel - start_channel) / nchannel));
    int padded_channel_index = padded_channel_factor ? padded_channel_factor + 1 : 0;
    int padded_slice_factor = (int)(log2(32 * (end_time - start_time) / nslice));
    int padded_slice_index = padded_slice_factor ? padded_slice_factor + 1 : 0;

    int offset = (channel_vec[padded_channel_index] - (end_channel - start_channel)) / 2;

    // auto full_channel = (int)pow(2, padded_channel);
    // int scratch_start_chann = (full_channel-(end_channel - start_channel)) / 2;

    // cudaMemcpy2D(scratch + nslice * scratch_start_chann, nslice, input + start_channel * nslice, nslice, (end_time - start_time) * sizeof(cufftComplex), end_channel - start_channel, cudaMemcpyDeviceToDevice);
    // cufftExecC2C(large_plans[6*padded_channel + padded_slice], scratch, scratch, CUFFT_FORWARD);
    // synthesize<<<end_time - start_time, area, full_channel>>>(scratch, Box+boxind, output, taps);
}

synthesizer::~synthesizer()
{
    for (int ind = 0; ind < 25; ind++)
    {
        cufftDestroy(input_plans[ind]);
        cufftDestroy(downconvert_plans[ind]);
        cudaFree(filters[ind]);
    }
    cudaFree(scratch_space);
    cudaFree(multiply_scratch_space);
    delete[] filters;
    delete[] input_plans;
    delete[] downconvert_plans;
}
