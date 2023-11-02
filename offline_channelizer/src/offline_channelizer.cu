#include "../include/offline_channelizer.cuh"
#include "/opt/asmlib/asmlib.h"
#include <iostream>
#include <vector>
#include <omp.h>
#include <stdio.h>
// #include <string.h>
#include <chrono>
// #include <immintrin.h>
using std::cout;
using std::endl;
using std::make_unique;
using std::vector;
using std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::milli;


// const int NCHANNEL = 1024;
// const int NCHANNELHALF = 512;
// const int NSLICE   = 2*131072;
// const int NPROTO   = 100;
const int BLOCKCHANNELS = 32;
const int BLOCKSLICES = 32;
// const int GRIDCHANNELS = 32;
// const int GRIDSUBCHANNELS = 16;
// const int GRIDSLICES = 8192;
// const int HALFSUBCHANNELS = 16;

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
void make_coeff_matrix(cufftComplex* gpu, complex<float>* inp, int nproto, int nchannel, int nslice) {
    int nchannelhalf = nchannel / 2;
    for (int id = 0; id < nchannel * nproto; id++)
    {
        int tap_id = id / nchannel;
        int chann_id = id % nchannel;
        if (chann_id < nchannelhalf)
        {
            cudaMemcpy(gpu + 2 * tap_id  + chann_id * nslice, inp + id, sizeof(cufftComplex), cudaMemcpyHostToDevice);
        }
        else 
        {
            cudaMemcpy(gpu + 2 * tap_id + 1 + chann_id * nslice, inp + id, sizeof(cufftComplex), cudaMemcpyHostToDevice);   
        }
        // auto err_0 = cudaGetLastError();
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

void __global__ multiply(cufftComplex* inp, cufftComplex* coeff, cufftComplex* output, int nchannel, int nslice, int griddim)
{
    int half         = blockIdx.y;
    int input_xcoord = blockIdx.x * blockDim.x + threadIdx.x;
    int raw_ycoord   = blockIdx.y * blockDim.y + threadIdx.y;
    int output_id    = raw_ycoord * nslice + input_xcoord;
    // int coeff_id     = output_id;
    int input_ycoord;
    int input_id;
    if (half < (griddim / 2))
    {
        input_ycoord = raw_ycoord;
        // coeff_id     = input_ycoord*nslice + input_xcoord;
    }
    else
    {
        input_ycoord = raw_ycoord - (griddim / 2)*blockDim.y; //(blockIdx.y - HALFSUBCHANNELS) * blockDim.y + threadIdx.y;
        // coeff_id     = (nchannel*nslice / 2) + input_ycoord*nslice + input_xcoord;
    }
    input_id         = (nchannel / 2 - input_ycoord)*nslice + input_xcoord;
    cufftComplex lhs = inp[input_id];
    cufftComplex rhs = coeff[output_id];
    output[output_id] = make_cuComplex(lhs.x*rhs.x - lhs.y*rhs.y, lhs.x*rhs.y + lhs.y*rhs.x);
}

void __global__ scale(cufftComplex* inp, bool row, int nchannel, int nslice)
{
    int inp_id = (blockIdx.y * blockDim.y + threadIdx.y)*nslice + blockIdx.x * blockDim.x + threadIdx.x;
    if (row)
    {
        inp[inp_id] = make_cuComplex(inp[inp_id].x / static_cast<float>(nslice), inp[inp_id].y / static_cast<float>(nslice));
    }
    else
    {
        inp[inp_id] = make_cuComplex(inp[inp_id].x / static_cast<float>(nchannel), inp[inp_id].y / static_cast<float>(nchannel));
    }
}

void __global__ alias(cufftComplex* inp, int nslice)
{
    int x_coord = blockIdx.x * blockDim.x + threadIdx.x;
    int y_coord = blockIdx.y * blockDim.y + threadIdx.y;
    int id = y_coord * nslice + x_coord;
    int signx = (1 - 2*(x_coord %2));
    bool signy = (y_coord % 2 == 0);
    if (signy)
    {
        if (signx != 1)
        {
            inp[id] = make_cuComplex(-inp[id].x, -inp[id].y);
        }
    }
}

// This reshape function has been tested to be correct
void __global__ reshape(cufftComplex* inp, cufftComplex* output, int nchannel, int nslice)
{
    __shared__ cufftComplex tile[BLOCKCHANNELS][BLOCKSLICES];
    int input_x_coord = blockIdx.x * blockDim.x + threadIdx.x;
    int input_y_coord = blockIdx.y * blockDim.y + threadIdx.y;
    auto inter = inp + (nchannel / 2)*input_y_coord + input_x_coord;
    tile[threadIdx.x][threadIdx.y] = *inter;
    __syncthreads();
    int output_grid_y_coord = (blockIdx.x * blockDim.x + threadIdx.y) * nslice;
    int output_grid_x_coord = blockIdx.y * blockDim.y + threadIdx.x;
    auto outer = output + output_grid_y_coord + output_grid_x_coord;
    (*outer) = tile[threadIdx.y][threadIdx.x];
}

void __global__ fft_shift(cufftComplex* inp, int nslice, bool row)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int id = idy*nslice + idx;
    int sign = row ? (1-2*(idx % 2)) : (1-2*(idy % 2));
    inp[id] = make_cuComplex(sign*inp[id].x, sign*inp[id].y);
}

// // This is a very slow CUDA kernel, because of strided access
// void __global__ club(float* inp, cufftComplex* output, int size)
// {
//     int id = blockIdx.x * blockDim.x + threadIdx.x;
//     int out_id = static_cast<int>(id / 2);
//     // printf("%d\n", id);
//     if (id < size)
//     {
//         // printf("Inside Kernel function\n");
//         if (id%2 == 0)
//         {
//             output[out_id].x = inp[id];
//         }
//         // printf("%f %f\n", output[out_id].x, output[out_id].y);
//         else 
//         {
//             output[out_id].y = inp[id];
//         }
//     }
// }

// void __global__ club_fromstream(float* real, float* imag, cufftComplex* output, int size)
// {
//     int id = blockIdx.x * blockDim.x + threadIdx.x;
//     if (id < size)
//     {
//         output[id] = make_cuComplex(real[id], imag[id]);
//     }
// }

// void __global__ declub_fromstream(float* input, float* real, float* imag, int size, int* mask)
// {
//     int id = blockIdx.x * blockDim.x + threadIdx.x;
//     if (id < size)
//     {
//         if (id%2 == 0)
//         {
//             real[mask[id]] = input[id];
//         }
//         else
//         {
//             imag[mask[id]] = input[id];
//         }
//     }
// }

channelizer::channelizer(complex<float> *coeff_arr, int npr, int nchan, int nsl)
: nchannel{nchan}, nslice{nsl}, nproto{npr}, gridchannels{nchan / BLOCKCHANNELS}, gridslices{nsl / BLOCKSLICES} //, input_buffer(nchannel*nslice / 2)
{
    // Create CUDA events to calculate the running time on the default stream.
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);

    // Allocate GPU memory for filter coefficients.
    cudaMalloc((void **)&coeff_fft_polyphaseform, sizeof(cufftComplex) * nchannel * nslice);

    // Allocate GPU memory to hold the input on the GPU.
    cudaMalloc((void**)&input_buffer, sizeof(cufftComplex)*nchannel*nslice/2);
    
    // Allocate GPU memory to hold reshaped input.
    cudaMalloc((void **)&reshaped_buffer, sizeof(cufftComplex) * (nchannel / 2) * nslice);

    // Allocate GPU memory to hold channelizer outputs.
    cudaMalloc((void **)&output_buffer, sizeof(cufftComplex) * nchannel * nslice);

    // Allocate GPU memory to hold intermediate scratch results.
    cudaMalloc((void **)&scratch_buffer, sizeof(cufftComplex) * (nchannel / 2) * nslice);

    /*
     * Initial FFT of input
     */
    istride_0 = 1;
    ostride_0 = 1;
    idist_0 = nslice;
    odist_0 = nslice;
    batch_0 = (nchannel / 2);
    n_0 = new int [1];
    *n_0 = nslice;
    inembed_0 = n_0;
    onembed_0 = n_0;
    cufftPlanMany(&plan_0, 1, n_0, inembed_0, istride_0, idist_0, onembed_0, ostride_0, odist_0, CUFFT_C2C, batch_0);

    /*
     * Channel IFFT of samples which have had the filter FFT multiplied to them.
     */
    istride_1 = 1;
    ostride_1 = 1;
    idist_1 = nslice;
    odist_1 = nslice;
    batch_1 = nchannel;
    n_1 = new int [1];
    *n_1 = nslice;
    inembed_1 = n_1;
    onembed_1 = n_1;
    cufftPlanMany(&plan_1, 1, n_1, inembed_1, istride_1, idist_1, onembed_1, ostride_1, odist_1, CUFFT_C2C, batch_1);


    /*
     * Final Downconversion IFFT. This applies an IFFT along the channel dimension.
     */
    istride_2 = nslice;
    ostride_2 = nslice;
    idist_2 = 1;
    odist_2 = 1;
    batch_2 = nslice;
    n_2 = new int [1];
    *n_2 = nchannel;
    inembed_2 = n_2;
    onembed_2 = n_2;
    cufftPlanMany(&plan_2, 1, n_2, inembed_2, istride_2, idist_2, onembed_2, ostride_2, odist_2, CUFFT_C2C, batch_2);
    make_coeff_matrix(coeff_fft_polyphaseform, coeff_arr, nproto, nchannel, nslice);
}

// input will be page locked on Host
void channelizer::process(float* input, cufftComplex* output)
{
    dim3 dimBlock(BLOCKCHANNELS, BLOCKSLICES);
    dim3 dimGridMultiply(gridslices, gridchannels);
    dim3 dimGridReshape(gridchannels / 2, gridslices);
    // float duration_;
    // time = 0.0;
    // cudaEventRecord(start,0);
    cudaMemcpy(input_buffer, input, sizeof(cufftComplex)*nchannel*nslice / 2, cudaMemcpyHostToDevice);
    reshape<<<dimGridReshape, dimBlock>>>(input_buffer, reshaped_buffer, nchannel, nslice);
    cufftExecC2C(plan_0, reshaped_buffer, scratch_buffer, CUFFT_FORWARD);
    multiply<<<dimGridMultiply, dimBlock>>>(scratch_buffer, coeff_fft_polyphaseform, output_buffer, nchannel, nslice, gridchannels);
    cufftExecC2C(plan_1, output_buffer, output_buffer, CUFFT_INVERSE);
    scale<<<dimGridMultiply, dimBlock>>>(output_buffer, true, nchannel, nslice);
    // fft_shift<<<dimGridMultiply, dimBlock>>>(output_buffer, nslice, false);
    cufftExecC2C(plan_2, output_buffer, output_buffer, CUFFT_INVERSE);
    scale<<<dimGridMultiply, dimBlock>>>(output_buffer, false, nchannel, nslice);
    alias<<<dimGridMultiply, dimBlock>>>(output_buffer, nslice);
    cudaMemcpy(output, output_buffer, sizeof(cufftComplex)*nslice*nchannel, cudaMemcpyDeviceToDevice);
    // cudaEventRecord(stop,0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&duration_, start, stop);
    // time += duration_;
}

channelizer::~channelizer()
{
    cufftDestroy(plan_0);
    cufftDestroy(plan_1);
    cufftDestroy(plan_2);
    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);
    delete [] n_0;
    delete [] n_1;
    delete [] n_2;
    cudaFree(input_buffer);
    cudaFree(coeff_fft_polyphaseform);
    cudaFree(scratch_buffer);
    cudaFree(reshaped_buffer);
    cudaFree(output_buffer);
}