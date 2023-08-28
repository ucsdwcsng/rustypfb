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

channelizer::channelizer(int nchann, int nsl, complex<float>* coeff_arr)
{
    nchannel = nchann;
    nslice   = nsl;

    // Allocate GPU memory for filter coefficients.
    cudaMalloc((void**) &coeff_fft_polyphaseform, sizeof(cufftComplex)*nchannel*nslice);

    // Allocate GPU memory for internal buffer.
    cudaMalloc((void**) &internal_buffer, sizeof(cufftComplex)*nchannel*nslice);

    /*
     * Plan 1 : Take FFT along each row. There are nslice elements in each row.
     * There are nchannel rows.
     */
    istride_1 = 1;
    idist_1 = nslice;
    batch_1 = nchannel;
    ostride_1 = 1;
    odist_1 = nslice;
    n_1 = new int [1];
    *n_1 = nslice;
    inembed_1 = n_1;
    onembed_1 = n_1;

    /*
     * Plan 2 : Take IFFT along each row. There are nslice elements in each row.
     * There are nchannel rows.
     */
    // istride_2 = 1;
    // idist_2 = nslice;
    // batch_2 = nchannel;
    // ostride_2 = 1;
    // odist_2 = nslice;

    /*
     * Plan 3 : Take IFFT along each column. There are nslice elements in each row.
     * There are nchannel rows.
     */
    istride_2 = nslice;
    idist_2 = 1;
    batch_2 = nslice;
    ostride_2 = nslice;
    odist_2 = 1;
    n_2 = new int [1];
    *n_2 = nchannel;
    inembed_2 = n_2;
    onembed_2 = n_2;

    cufftPlanMany(&plan_1, rank, n_1, inembed_1, istride_1, idist_1, onembed_1, ostride_1, odist_1, CUFFT_C2C, batch_1);
    cufftPlanMany(&plan_2, rank, n_2, inembed_2, istride_2, idist_2, onembed_2, ostride_2, odist_2, CUFFT_C2C, batch_2);
}