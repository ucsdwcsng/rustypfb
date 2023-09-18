#include "../include/offline_channelizer.cuh"
#include <iostream>
#include <omp.h>
#include <stdio.h>
using std::cout;
using std::endl;
using std::make_unique;

const int NSTREAMS = 8;
const int NCHANNEL = 1024;
const int NSLICE   = 131072;
const int NPROTO   = 100;
const int SUBCHANNELS = 32;
const int SUBSLICES = 32;
const int NUMYBLOCKS = 4096;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

ProcessData::ProcessData(int istride_, int ostride_, int idist_, int odist_, int batch_, int n_, int rank_, shared_ptr<cudaStream_t> stream_)
:istride(istride_), ostride(ostride_), idist(idist_), odist(odist_), batch(batch_), stream(stream_), rank(rank_)
{   
    n = new int [1];
    *n = n_;
    inembed = n;
    onembed = n;
    cufftPlanMany(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch);
    cufftSetStream(plan, *stream.lock());
    // cout << "constructing processdata struct" << endl;
}

ProcessData::~ProcessData()
{
    // cout << "FFtData destroyed" << endl;
    cufftDestroy(plan);
    delete [] n;
}

/*
 * This is the main reshaping kernel function. It is designed to act across streams in separate streams.
 * 
 * At a given time, one block processes 32 channels, and 32 slices in each channel.
 * 
 * Shared memory amounts to be 64 kB per block which is supported by
 * A100 Ampere.
 */

void __global__ flipped_transpose(cufftComplex* inp, cufftComplex* outp)
{
   __shared__ cufftComplex tile[SUBCHANNELS][SUBSLICES];
   int sub_slice_id = blockIdx.y;
   int sub_channel_id = threadIdx.x;
   int tile_ylocation = threadIdx.y;
   auto inter = inp + SUBSLICES * sub_slice_id * NCHANNEL + sub_channel_id;
   tile[sub_channel_id][tile_ylocation] = *inter;
   __syncthreads();

   int output_offset = SUBSLICES * sub_slice_id * tile_ylocation;
   auto intermediate_value = tile[tile_ylocation][31 - sub_channel_id];
   *(outp + output_offset + sub_channel_id) = intermediate_value;
}

void make_coeff_matrix(cufftComplex* gpu, complex<float>* inp) {
    for (int id = 0; id < NCHANNEL * NPROTO; id++)
    {
        int tap_id = id / NCHANNEL;
        int chann_id = id % NCHANNEL;
        cudaMemcpy(gpu + tap_id  + chann_id * NSLICE, inp + id, sizeof(cufftComplex), cudaMemcpyHostToDevice);
        auto err_0 = cudaGetLastError();
    }
}

void __global__ multiply_per_stream(cufftComplex* inp, cufftComplex* coeff, cufftComplex* output, int start_channel_id)
{
    int slice_id = blockIdx.x * blockDim.x + threadIdx.x;
    int sub_channel_id   = blockIdx.y * blockDim.y + threadIdx.y;
    // printf("Inside multiply kernel\n");
    if ((sub_channel_id < SUBCHANNELS) && (slice_id < NSLICE))
    {
        int prod_id = start_channel_id + sub_channel_id*NSLICE + slice_id;
        int inp_id  = sub_channel_id*NSLICE + slice_id;
        cufftComplex lhs  = inp[inp_id];
        cufftComplex rhs  = coeff[prod_id];
        output[prod_id] =  make_cuComplex(lhs.x* rhs.x - lhs.y * rhs.y, lhs.x * rhs.y + lhs.y * rhs.x);
        // printf("%f %f\n", output[id].x, output[id].y);
    }
}

channelizer::channelizer(complex<float> *coeff_arr)
{
    // Allocate GPU memory for filter coefficients.
    cudaMalloc((void **)&coeff_fft_polyphaseform, sizeof(cufftComplex) * NCHANNEL * NSLICE);

    // Allocate Pagelocked memory for input buffer on host
    cudaMallocHost((void **)&locked_buffer, sizeof(cufftComplex) * NCHANNEL * NSLICE);

    // Allocate GPU memory for output buffer.
    cudaMalloc((void **)&output_buffer, sizeof(cufftComplex) * NCHANNEL * NSLICE);

    // Allocate GPU memory for scratch buffer.
    cudaMalloc((void **)&scratch_buffer, sizeof(cufftComplex) * SUBCHANNELS * NSLICE);


    /*
     * Plan 1 : Take FFT in slice dimension. 
     */
    istride_1 = 1;
    idist_1 = NSLICE;
    batch_1 = NCHANNEL;
    ostride_1 = 1;
    odist_1 = NSLICE;
    n_1 = new int[1];
    *n_1 = NSLICE;
    inembed_1 = n_1;
    onembed_1 = n_1;

    istride_2 = NSLICE;
    ostride_2 = NSLICE;
    idist_2 = 1;
    odist_2 = 1;
    batch_2 = NSLICE;
    n_2 = new int [1];
    *n_2 = NCHANNEL;
    inembed_2 = n_2;
    onembed_2 = n_2;

    streams = {};
    forward_process_fft_streams = {};
    for (int i=0; i<NSTREAMS; i++)
    {
        auto stream = shared_ptr<cudaStream_t>(new cudaStream_t);
        cudaStreamCreate(stream.get());
        streams.push_back(stream);
        auto process_data = make_shared<ProcessData>(1, 1, NSLICE, NSLICE, SUBCHANNELS, NSLICE, 1, stream);
        forward_process_fft_streams.push_back(process_data);
    }


    // make_coeff_matrix(coeff_fft_polyphaseform, coeff_arr, nchannel, ntaps, nslice);
    make_coeff_matrix(coeff_fft_polyphaseform, coeff_arr);

    // Hopefully faster GPU version.
    cufftExecC2C(plan_1, coeff_fft_polyphaseform, coeff_fft_polyphaseform, CUFFT_FORWARD);
}

void channelizer::process(complex<float>* input)
{
    
    dim3 dimBlockFlipTranspose(SUBCHANNELS, SUBSLICES);
    dim3 dimGridFlipTranspose(1, NUMYBLOCKS);
    dim3 dimBlockMultiply(SUBSLICES, SUBCHANNELS);
    dim3 dimGridMultiply(NUMYBLOCKS, 1);
    memcpy(locked_buffer, input, sizeof(cufftComplex)*NCHANNEL*NSLICE);
    for (int j=0; j< NSTREAMS; j++){
        
        auto stream_val = *(streams[j]);
        flipped_transpose<<<dimGridFlipTranspose, dimBlockFlipTranspose, 0, stream_val>>>(locked_buffer + NCHANNEL - 1 - j*SUBCHANNELS, scratch_buffer);
        cufftExecC2C(forward_process_fft_streams[j]->plan, scratch_buffer, scratch_buffer, CUFFT_FORWARD);
        multiply_per_stream<<<dimGridMultiply, dimBlockMultiply, 0, stream_val>>>(scratch_buffer, coeff_fft_polyphaseform, output_buffer, j*SUBCHANNELS);
        cufftExecC2C(forward_process_fft_streams[j]->plan, output_buffer + j*SUBCHANNELS*NSLICE, output_buffer + j*SUBCHANNELS*NSLICE, CUFFT_INVERSE);
    }
    // cudaMemcpy(output, input_buffer, sizeof(cufftComplex)*nslice*nchannel, cudaMemcpyDeviceToHost);
    auto err_t = cudaDeviceSynchronize();
    cufftExecC2C(plan_2, output_buffer, output_buffer, CUFFT_INVERSE);
    cout << cudaGetErrorString(err_t) << endl;
}

channelizer::~channelizer()
{
    cufftDestroy(plan_1);
    cufftDestroy(plan_2);
    delete [] n_1;
    delete [] n_2;
    cudaFree(coeff_fft_polyphaseform);
    cudaFree(scratch_buffer);
    cudaFreeHost(locked_buffer);
    cudaFree(output_buffer);
}

unique_ptr<channelizer> create_chann(vector<complex<float>> coeff_arr)
{
    return make_unique<channelizer>(&coeff_arr[0]);
}