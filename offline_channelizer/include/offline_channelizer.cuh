#ifndef _CHANNELIZER_
#define _CHANNELIZER_
#include <cufft.h>
#include <complex.h>
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

void __global__ create_polyphase_input(cufftComplex*, cufftComplex*, int, int);
void __global__ multiply(cufftComplex*, cufftComplex*, cufftComplex*, int);
void __global__ make_coeff_matrix(complex<float>*, cufftComplex*, int, int, int);

/*
 * This struct wraps together CUFFT plans and the CUDA streams that those plans execute on.
 * Useful to launch CUFFT methods asynchronously on different streams.
 */

struct ProcessData {
    cufftHandle plan;
    int istride;
    int ostride;
    int idist;
    int odist;
    int batch;
    int* inembed;
    int* onembed;
    int* n;
    int rank;
    weak_ptr<cudaStream_t> stream;
    ProcessData(int, int, int, int, int, int, int,shared_ptr<cudaStream_t>);
    ~ProcessData();
};

class channelizer
{
    public:
    /*
     * nchannel is the number of channels.
     * nslice is the number of samples in each channel.
     * ntaps is the number of filter taps per channel.
     */
    // int nchannel;
    // int nslice;
    // int ntaps;


    /*
     * Polyphase filter coefficients which have been filtered along slice dimension.
     */
    cufftComplex* coeff_fft_polyphaseform;
    int rank = 1;

    // /*
    //  * Plan for taking FFT of polyphase inputs along slice dimension.
    //  */
    // cufftHandle plan_1;
    // int istride_1;
    // int ostride_1;
    // int idist_1;
    // int odist_1;
    // int batch_1;
    // int* n_1;
    // int* inembed_1;
    // int* onembed_1;

    // /*
    //  * Plan for taking IFFT (for convolution) along slice dimension.
    //  * Since this has the same dimensions as in plan_1, no need
    //  * to initiate similar variables.
    //  */
    // // cufftHandle plan_2;
    // // int istride_2;
    // // int ostride_2;
    // // int idist_2;
    // // int odist_2;
    // // int batch_2;
    // // int* inembed_2;
    // // int* onembed_2;

    // /*
    //  * Plan for taking IFFT (for downconversion) along channel dimension.
    //  */
    // cufftHandle plan_2;
    // int istride_2;
    // int ostride_2;
    // int idist_2;
    // int odist_2;
    // int batch_2;
    // int* inembed_2;
    // int* onembed_2;
    // int* n_2;

    /*
     * Plan for taking FFT along slice dimension. This requires no reshape of input.
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

    // /*
    //  * Plan for taking IFFT along channels.
    //  */

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
     * Output buffer to hold results.
     */
    cufftComplex* output_buffer;

    /*
     * Internal scratch buffer on GPU
     */
    cufftComplex* scratch_buffer;

    /*
     * Internal buffer to hold non-polyphase input on GPU.
     */
    // cufftComplex* input_buffer;

    /*
     * Page locked memory on host which is accessible to both 
     * device and host.
     */
    cufftComplex* locked_buffer;

    /*
     * Number of streams that this channelizer object acts on.
     */

    // int nstreams;
    // int subchannels;
    vector<shared_ptr<cudaStream_t>> streams;
    vector<shared_ptr<ProcessData>> forward_process_fft_streams;
    vector<shared_ptr<ProcessData>> down_convert_fft_streams;


    /*
     * Constructor
     */
    channelizer(complex<float>*);
    void process(complex<float>*);
    ~channelizer();
};

unique_ptr<channelizer> create_chann(int, vector<complex<float>>);

#endif

