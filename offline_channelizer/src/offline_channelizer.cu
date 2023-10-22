#include "../include/offline_channelizer.cuh"
#include <iostream>
#include <vector>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <immintrin.h>
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

void naivetransfer(void* pdest, void* psource, size_t num_bytes)
{
    char* pDest = (char* )pdest;
    char* pSource = (char* )psource;
    for (int i=0; i< num_bytes; i++)
    {
        *pDest++ = *pSource++;
    }
}

void transfer_intrinsic(float* outp, float* inp, size_t num_samples)
{
    {
        for (int i=0; i<num_samples;i+=16)
        {
            __m512 a = _mm512_loadu_ps((__m512 *)(inp+i));
            _mm512_storeu_ps(outp + i,a);
        }
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

void initialize_mask(int* input_mask, int size)
{
    for (int i=0; i<size; i++)
    {
        input_mask[i] = i/2;
    }
}

void __global__ multiply(cufftComplex* inp, cufftComplex* coeff, cufftComplex* output, int nchannel, int nslice, int griddim)
{
    int half = blockIdx.y;
    int input_xcoord = blockIdx.x * blockDim.x + threadIdx.x;
    int raw_ycoord = blockIdx.y * blockDim.y + threadIdx.y;
    int output_id = (nchannel - raw_ycoord) * nslice + input_xcoord;
    int input_ycoord;
    int inp_id;
    int coeff_id;
    if (half < (griddim / 2))
    {
        input_ycoord = raw_ycoord;
        inp_id = input_ycoord*nslice + input_xcoord;
        coeff_id = inp_id;
    }
    else
    {
        input_ycoord = raw_ycoord - (griddim / 2)*blockDim.y; //(blockIdx.y - HALFSUBCHANNELS) * blockDim.y + threadIdx.y;
        inp_id = input_ycoord*nslice + input_xcoord;
        coeff_id = (nchannel*nslice / 2) + inp_id;
    }
    cufftComplex lhs = inp[inp_id];
    cufftComplex rhs = coeff[coeff_id];
    output[output_id] = make_cuComplex(lhs.x* rhs.x - lhs.y * rhs.y, lhs.x * rhs.y + lhs.y * rhs.x);
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
    // int output_x_coord = blockIdx.y * blockDim.y + threadIdx.x;
    // int output_y_coord = blockIdx.x * blockDim.x + threadIdx.y;
    auto outer = output + output_grid_y_coord + output_grid_x_coord;
    (*outer) = tile[threadIdx.y][threadIdx.x];
}

// This is a very slow CUDA kernel, because of strided access
void __global__ club(float* inp, cufftComplex* output, int size)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int out_id = static_cast<int>(id / 2);
    // printf("%d\n", id);
    if (id < size)
    {
        // printf("Inside Kernel function\n");
        if (id%2 == 0)
        {
            output[out_id].x = inp[id];
        }
        // printf("%f %f\n", output[out_id].x, output[out_id].y);
        else 
        {
            output[out_id].y = inp[id];
        }
    }
}

void __global__ club_fromstream(float* real, float* imag, cufftComplex* output, int size)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size)
    {
        output[id] = make_cuComplex(real[id], imag[id]);
    }
}

void __global__ declub_fromstream(float* input, float* real, float* imag, int size, int* mask)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size)
    {
        if (id%2 == 0)
        {
            real[mask[id]] = input[id];
        }
        else
        {
            imag[mask[id]] = input[id];
        }
    }
}

channelizer::channelizer(complex<float> *coeff_arr, int npr, int nchan, int nsl)
: nchannel{nchan}, nslice{nsl}, nproto{npr}, gridchannels{nchan / BLOCKCHANNELS}, gridslices{nsl / BLOCKSLICES} //, input_buffer(nchannel*nslice / 2)
{
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Allocate GPU memory for filter coefficients.
    cudaMalloc((void **)&coeff_fft_polyphaseform, sizeof(cufftComplex) * nchannel * nslice);
    
    // Allocate Pagelocked memory for input buffer on host
    cudaMalloc((void **)&locked_buffer, sizeof(cufftComplex) * (nchannel / 2) * nslice);

    // cudaMallocHost((void **)&input_buffer, sizeof(cufftComplex) * (nchannel / 2) * nslice);
    input_buffer = (complex<float>* )malloc(sizeof(cufftComplex)*(nchannel / 2)*nslice);
    auto err_t = cudaHostRegister(input_buffer, (nchannel / 2)*(nslice)*sizeof(cufftComplex), cudaHostRegisterMapped);
    // // auto err_tt = cudaHostGetDevicePointer((void **)&input_device_ptr, (void* )input_buffer, 0);
    cudaMalloc((void**)&input_buffer_device, sizeof(cufftComplex)*nchannel*nslice/2);
    cudaMalloc((void **)&output_buffer, sizeof(cufftComplex) * nchannel * nslice);
    // Allocate GPU memory for scratch buffer.
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
    // initialize_mask(mask, nchannel*nslice);
}

// void channelizer::process(float* input_real, float* input_imag, complex<float>* output)
void channelizer::process(float* input, cufftComplex* output)
{
    dim3 dimBlock(BLOCKCHANNELS, BLOCKSLICES);
    dim3 dimGridMultiply(gridslices, gridchannels);
    dim3 dimGridReshape(gridchannels / 2, gridslices);
    float duration_;
    time = 0.0;
    auto start_time = steady_clock::now();
    memcpy(input_buffer, input, sizeof(float)*nchannel*nslice);
    auto end_time = steady_clock::now();
    time += duration<float, milli>(end_time - start_time).count();
    time_cpy = time;

    cudaEventRecord(start,0);
    cudaMemcpy(input_buffer_device, input_buffer, sizeof(cufftComplex)*nchannel*nslice / 2, cudaMemcpyHostToDevice);
    // time_cpy = duration<float, milli>(end_du - start_du).count();
    // cudaMemcpy(input_buffer, input, sizeof(float)*nchannel*nslice, cudaMemcpyHostToHost);
    reshape<<<dimGridReshape, dimBlock>>>(input_buffer_device, locked_buffer, nchannel, nslice);
    // reshape<<<dimGridReshape, dimBlock>>>(input_buffer, locked_buffer, nchannel, nslice);
    // auto err_t1 = cudaGetLastError();
    // cout << cudaGetErrorString(err_t1) << endl;
    cufftExecC2C(plan_0, locked_buffer, scratch_buffer, CUFFT_FORWARD);
    // auto err_t0 = cudaGetLastError();
    // cout << cudaGetErrorString(err_t0) << endl;
    multiply<<<dimGridMultiply, dimBlock>>>(scratch_buffer, coeff_fft_polyphaseform, output_buffer, nchannel, nslice, gridchannels);
    // auto err_t1 = cudaGetLastError();
    // cout << cudaGetErrorString(err_t1) << endl;
    cufftExecC2C(plan_1, output_buffer, output_buffer, CUFFT_INVERSE);
    // auto err_t2 = cudaGetLastError();
    // cout << cudaGetErrorString(err_t2) << endl;
    scale<<<dimGridMultiply, dimBlock>>>(output_buffer, true, nchannel, nslice);
    // auto err_t3 = cudaGetLastError();
    // cout << cudaGetErrorString(err_t3) << endl;
    cufftExecC2C(plan_2, output_buffer, output_buffer, CUFFT_INVERSE);
    // auto err_t4 = cudaGetLastError();
    // cout << cudaGetErrorString(err_t4) << endl;
    scale<<<dimGridMultiply, dimBlock>>>(output_buffer, false, nchannel, nslice);
    // auto err_t5 = cudaGetLastError();
    // cout << cudaGetErrorString(err_t5) << endl;
    alias<<<dimGridMultiply, dimBlock>>>(output_buffer, nslice);
    // auto err_t6 = cudaGetLastError();
    // cout << cudaGetErrorString(err_t6) << endl;
    cudaMemcpy(output, output_buffer, sizeof(cufftComplex)*nslice*nchannel, cudaMemcpyDeviceToDevice);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&duration_, start, stop);
    time += duration_;
    // auto err_t7 = cudaGetLastError();
    // cout << cudaGetErrorString(err_t7) << endl;
}

channelizer::~channelizer()
{
    cufftDestroy(plan_0);
    cufftDestroy(plan_1);
    cufftDestroy(plan_2);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    delete [] n_0;
    delete [] n_1;
    delete [] n_2;
    cudaHostUnregister(input_buffer);
    free(input_buffer);
    // cudaFreeHost(input_buffer);
    cudaFree(input_buffer_device);
    cudaFree(coeff_fft_polyphaseform);
    cudaFree(scratch_buffer);
    cudaFree(locked_buffer);
    cudaFree(output_buffer);
}