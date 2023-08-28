#include "offline_channelizer.cuh"

void __global__ create_polyphase_input(cufftComplex *inp, cufftComplex *outp, int nchannel, int nslice)
{
    int slice_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (slice_id < nslice)
    {
        for (int channel_id = 0; channel_id < nchannel; channel_id++)
        {
            outp[channel_id * nslice + slice_id] = inp[(1 + slice_id) * nchannel - 1 - channel_id];
        }
    }
}

void __global__ multiply(cufftComplex *inp, cufftComplex *coeff, cufftComplex *outp, int nsamples)
{
    int sample_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_id < nsamples)
        outp[sample_id] = make_cuComplex(inp[sample_id].x * coeff[sample_id].x - inp[sample_id].y * coeff[sample_id].y, inp[sample_id].x * coeff[sample_id].y + inp[sample_id].y * coeff[sample_id].x);
}