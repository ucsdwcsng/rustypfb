#include "../include/offline_channelizer.cuh"
#include <iostream>
#include <omp.h>
#include <stdio.h>
using std::cout;
using std::endl;
using std::make_unique;


const int NCHANNEL = 1024;
const int NCHANNELHALF = 512;
const int NSLICE   = 2*131072;
const int NPROTO   = 100;
const int BLOCKCHANNELS = 32;
const int BLOCKSLICES = 32;
const int GRIDCHANNELS = 32;
const int GRIDSLICES = 8192;
const int HALFSUBCHANNELS = 16;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


/*
 * Create the FFT version of the coefficient filters for non-maximal decimation.
 */
void make_coeff_matrix(cufftComplex* gpu, complex<float>* inp) {
    for (int id = 0; id < NCHANNEL * NPROTO; id++)
    {
        int tap_id = id / NCHANNEL;
        int chann_id = id % NCHANNEL;
        if (chann_id < NCHANNELHALF)
        {
            cudaMemcpy(gpu + 2 * tap_id  + chann_id * NSLICE, inp + id, sizeof(cufftComplex), cudaMemcpyHostToDevice);
        }
        else 
        {
            cudaMemcpy(gpu + 2 * tap_id + 1 + chann_id * NSLICE, inp + id, sizeof(cufftComplex), cudaMemcpyHostToDevice);   
        }
        // auto err_0 = cudaGetLastError();
    }
    int istride = 1;
    int ostride = 1;
    int idist = NSLICE;
    int odist = NSLICE;
    int batch = NCHANNEL;
    int* n = new int [1];
    *n = NSLICE;
    int* inembed = n;
    int* onembed = n;
    cufftHandle plan;
    cufftPlanMany(&plan, 1, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch);
    cufftExecC2C(plan, gpu, gpu, CUFFT_FORWARD);
    cufftDestroy(plan);
    delete [] n;

}

void __global__ multiply(cufftComplex* inp, cufftComplex* coeff, cufftComplex* output)
{
    int half = blockIdx.y;
    int input_xcoord = blockIdx.x * blockDim.x + threadIdx.x;
    int input_ycoord;
    int inp_id;
    // int output_id = (NCHANNEL - input_ycoord) * NSLICE + input_xcoord;
    int coeff_id;
    if (half < HALFSUBCHANNELS)
    {
        input_ycoord = blockIdx.y * blockDim.y + threadIdx.y;
        inp_id = input_ycoord*NSLICE + input_xcoord;
        coeff_id = inp_id;
    }
    else
    {
        input_ycoord = (blockIdx.y - HALFSUBCHANNELS) * blockDim.y + threadIdx.y;
        inp_id = input_ycoord*NSLICE + input_xcoord;
        coeff_id = NCHANNELHALF*NSLICE + inp_id;
    }
    cufftComplex lhs = inp[inp_id];
    cufftComplex rhs = coeff[coeff_id];
    output[inp_id] = make_cuComplex(lhs.x* rhs.x - lhs.y * rhs.y, lhs.x * rhs.y + lhs.y * rhs.x);
}

void __global__ scale(cufftComplex* inp, bool row)
{
    int inp_id = (blockIdx.y * blockDim.y + threadIdx.y)*NSLICE + blockIdx.x * blockDim.x + threadIdx.x;
    if (row)
    {
        inp[inp_id] = make_cuComplex(inp[inp_id].x / static_cast<float>(NSLICE), inp[inp_id].y / static_cast<float>(NSLICE));
    }
    else
    {
        inp[inp_id] = make_cuComplex(inp[inp_id].x / static_cast<float>(NCHANNEL), inp[inp_id].y / static_cast<float>(NCHANNEL));
    }
}

channelizer::channelizer(complex<float> *coeff_arr)
{
    // Allocate GPU memory for filter coefficients.
    cudaMalloc((void **)&coeff_fft_polyphaseform, sizeof(cufftComplex) * NCHANNEL * NSLICE);

    // Allocate Pagelocked memory for input buffer on host
    cudaMallocHost((void **)&locked_buffer, sizeof(cufftComplex) * NCHANNELHALF * NSLICE);

    // Allocate GPU memory for output buffer.
    cudaMalloc((void **)&output_buffer, sizeof(cufftComplex) * NCHANNEL * NSLICE);

    // Allocate GPU memory for scratch buffer.
    cudaMalloc((void **)&scratch_buffer, sizeof(cufftComplex) * NCHANNELHALF * NSLICE);

    /*
     * Initial FFT of input
     */
    istride_1 = NCHANNELHALF;
    ostride_1 = 1;
    idist_1 = 1;
    odist_1 = NSLICE;
    batch_1 = NCHANNELHALF;
    n_1 = new int [1];
    *n_1 = NSLICE;
    inembed_1 = n_1;
    onembed_1 = n_1;
    cufftPlanMany(&plan_1, 1, n_1, inembed_1, istride_1, idist_1, onembed_1, ostride_1, odist_1, CUFFT_C2C, batch_1);

    /*
     * IFFT before downconversions
     */
    istride_0 = 1;
    ostride_0 = 1;
    idist_0 = NSLICE;
    odist_0 = NSLICE;
    batch_0 = NCHANNEL;
    n_0 = new int [1];
    *n_0 = NCHANNEL;
    inembed_0 = n_0;
    onembed_0 = n_0;
    cufftPlanMany(&plan_0, 1, n_0, inembed_0, istride_0, idist_0, onembed_0, ostride_0, odist_0, CUFFT_C2C, batch_0);

    /*
     * Final Downconversion IFFT
     */
    istride = NSLICE;
    ostride = NSLICE;
    idist = 1;
    odist = 1;
    batch = NSLICE;
    n = new int [1];
    *n = NCHANNEL;
    inembed = n;
    onembed = n;
    cufftPlanMany(&plan, 1, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch);
    make_coeff_matrix(coeff_fft_polyphaseform, coeff_arr);
}

void channelizer::process(complex<float>* input)
{
    dim3 dimBlockMultiply(BLOCKSLICES, BLOCKCHANNELS);
    dim3 dimGridMultiply(GRIDSLICES, GRIDCHANNELS);
    memcpy(locked_buffer, input, sizeof(cufftComplex)*NCHANNELHALF*NSLICE);
    cufftExecC2C(plan_1, locked_buffer, scratch_buffer, CUFFT_FORWARD);
    multiply<<<dimGridMultiply, dimBlockMultiply>>>(scratch_buffer, coeff_fft_polyphaseform, output_buffer);
    cufftExecC2C(plan_0, output_buffer, output_buffer, CUFFT_INVERSE);
    scale<<<dimGridMultiply, dimBlockMultiply>>>(output_buffer, true);
    cufftExecC2C(plan, output_buffer, output_buffer, CUFFT_INVERSE);
    scale<<<dimGridMultiply, dimBlockMultiply>>>(output_buffer, false);
}

channelizer::~channelizer()
{
    // cufftDestroy(plan_1);
    cufftDestroy(plan);
    cufftDestroy(plan_0);
    delete [] n;
    delete [] n_0;
    cudaFree(coeff_fft_polyphaseform);
    cudaFree(scratch_buffer);
    cudaFreeHost(locked_buffer);
    cudaFree(output_buffer);
}

unique_ptr<channelizer> create_chann(vector<complex<float>> coeff_arr)
{
    return make_unique<channelizer>(&coeff_arr[0]);
}