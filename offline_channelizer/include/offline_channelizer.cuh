#ifndef _CHANNELIZER_
#define _CHANNELIZER_
#include <cufft.h>
#include <complex.h>
#include <cuComplex.h>
#include <memory>
#include <vector>

using std::complex;
using std::unique_ptr;
using std::vector;
using std::shared_ptr;
using std::unique_ptr;
using std::weak_ptr;
using std::make_unique;
using std::make_shared;


void make_coeff_matrix(cufftComplex*, complex<float>*, int, int, int);
void __global__ multiply(cufftComplex*, cufftComplex*, cufftComplex*, int, int, int);
void __global__ scale(cufftComplex* , bool, int, int);
void __global__ alias(cufftComplex*, int);
void __global__ club(float*, cufftComplex*, int);
void transfer_intrinsic(float*, float*, size_t);

class channelizer
{
    public:
    /*
     * Polyphase filter coefficients which have been filtered along slice dimension.
     */
    int nchannel;
    int nproto;
    int nslice;

    int gridslices;
    int gridchannels;

    cudaEvent_t start, stop;
    float time, time_cpy;
    /*
     * Plan for taking the initial FFT of input samples on host along slice dimension. 
     * The input is arranged as given by the call to process:
     * 
     * input = x0 x1 .....
     * 
     * We would have wanted the input to be provided in polyphase form, determined by 
     * half the number of channels (NCHANNELHALF)
     * 
     * Therefore, we set
     * 
     * istride_0 = NCHANNELHALF
     * 
     * This ensures successive input elements are taken with a gap of NCHANNELHALF.
     * 
     * Next, on the output side, we would like the successive FFT outputs to be contiguous (for ease of filtering and outputting).
     * 
     * Therefore, we set
     * 
     * ostride_0 = 1
     * 
     * Successive FFT batches on the input side start at elements that are successors in the input array.
     * 
     * Thus,
     * 
     * idist_0 = 1
     * 
     * However, successive FFT batches on the output side are separated by NSLICE elements. Therefore,
     * 
     * odist_0 = NSLICE
     * 
     * Finally, batch is the number of FFTS we want to take, which would be NCHANNELHALF
     * 
     * and n_0 is the size of each FFT which is NSLICE.
     * 
     */
    cufftHandle plan_0;
    int istride_0;
    int ostride_0;
    int idist_0;
    int odist_0;
    int batch_0;
    int* inembed_0;
    int* onembed_0;
    int* n_0;

    /*
     * Plan for taking IFFT along channels, after multiplying with FFT of filter coefficients.
     * This time, successive elements in each batch of the FFT are contiguous to one another,
     * 
     * istride_1 = 1
     * 
     * We want the same thing to be maintained on the output, so
     * 
     * ostride_1 = 1
     * 
     * Distance between batches, 
     * 
     * idist_1 = odist_1 = NSLICE
     * 
     * and number of batches
     * 
     * batch_1 = NCHANNEL (see the difference from the previous plan).
     * 
     * n_1 is NSLICE.
     */

    cufftHandle plan_1;
    int istride_1;
    int ostride_1;
    int idist_1;
    int odist_1;
    int batch_1;
    int* inembed_1;
    int* onembed_1;
    int* n_1;

    /*
     * Plan for FFT along channel dimensions for final downconversion.

     * Successive elements in each FFT batch are separated by NSLICE at both input and output sides, so

     * istride_2 = ostride_2 = NSLICE

     * Successive elements on the input and output side are in different batches, so

     * idist_2 = odist_2 = 1

     * There are NSLICE batches, and the FFT length of each is NCHANNEL.
     */
    cufftHandle plan_2;
    int istride_2;
    int ostride_2;
    int idist_2;
    int odist_2;
    int batch_2;
    int* inembed_2;
    int* onembed_2;
    int* n_2;

    /*
     * Holds the Coefficient filter coefficients.
     */
    cufftComplex* coeff_fft_polyphaseform;

    /*
     * Output buffer to hold results.
     */
    cufftComplex* output_buffer;

    /*
     * Internal scratch buffer on GPU
     */
    cufftComplex* scratch_buffer;

    /*
     * Page locked memory on host which is accessible to both 
     * device and host, used to hold the interleaved floats.
    //  */
    // float* locked_buffer_real;
    // float* locked_buffer_imag;
    cufftComplex* locked_buffer;

    complex<float>* input_buffer;
    cufftComplex* input_buffer_device;
    // cufftComplex* locked_buffer_unshaped;
    // float* locked_buffer_input;
    // int* mask;

    /*
     * Constructor
     */
    channelizer(complex<float>*, int, int, int);
    void process(float*, cufftComplex*);
    // void process(float*, float*, complex<float>*);
    ~channelizer();
};
#endif

