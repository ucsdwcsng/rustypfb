#pragma once
#include <cufft.h>
#include <complex.h>

using std::complex;

void __global__ create_polyphase_input(cufftComplex*, cufftComplex*, int, int);
void __global__ multiply(cufftComplex*, cufftComplex*, cufftComplex*, int);

class channelizer
{
    public:

    int nchannel;
    int nslice;

    /*
     * Polyphase filter coefficients which have been filtered along slice dimension.
     */
    cufftComplex* coeff_fft_polyphaseform;

    /*
     * Plan for taking FFT of polyphase inputs along slice dimension.
     */
    cufftHandle plan_1;
    int istride_1;
    int ostride_1;
    int idist_1;
    int odist_1;
    int batch_1;
    int inembed_1;
    int onembed_1;

    /*
     * Plan for taking IFFT (for convolution) along slice dimension.
     */
    cufftHandle plan_2;
    int istride_2;
    int ostride_2;
    int idist_2;
    int odist_2;
    int batch_2;
    int inembed_2;
    int onembed_2;

    /*
     * Plan for taking IFFT (for downconversion) along channel dimension.
     */
    cufftHandle plan_3;
    int istride_3;
    int ostride_3;
    int idist_3;
    int odist_3;
    int batch_3;
    int inembed_3;
    int onembed_3;

    /*
     * Buffer to hold coefficients
     */

    cufftComplex* coefficient_buffer;

    /*
     * Internal buffer to hold intermediate results.
     */
    cufftComplex* internal_buffer;

    /*
     * Constructor
     */
    channelizer(int, int, complex<float>*);
    void process(cufftComplex*, cufftComplex*);
    ~channelizer();




};